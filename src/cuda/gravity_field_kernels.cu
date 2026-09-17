#include "physicsmade/cuda/field_grid.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace {

struct DeviceSource {
    double mass;
    double x;
    double y;
    double z;
};

__global__ void gravityFieldKernel(
    const DeviceSource* sources,
    int sourceCount,
    physicsmade::cuda::FieldSample* samples,
    int nx,
    int ny,
    int nz,
    physicsmade::math::Vector3 origin,
    physicsmade::math::Vector3 spacing,
    double gravitationalConstant) {
    const int flatIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int totalSamples = nx * ny * nz;
    if (flatIndex >= totalSamples) {
        return;
    }

    const int xyStride = nx * ny;
    const int iz = flatIndex / xyStride;
    const int remainder = flatIndex % xyStride;
    const int iy = remainder / nx;
    const int ix = remainder % nx;

    const physicsmade::math::Vector3 position{
        origin.x + (spacing.x * static_cast<double>(ix)),
        origin.y + (spacing.y * static_cast<double>(iy)),
        origin.z + (spacing.z * static_cast<double>(iz)),
    };

    physicsmade::math::Vector3 acceleration{};
    double potential = 0.0;

    for (int sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
        const DeviceSource source = sources[sourceIndex];
        const physicsmade::math::Vector3 delta{source.x - position.x, source.y - position.y, source.z - position.z};
        const double distanceSquared = delta.normSquared() + physicsmade::common::kEpsilon;
        const double distance = sqrt(distanceSquared);
        const double inverseDistanceCubed = 1.0 / (distanceSquared * distance);

        acceleration += delta * (gravitationalConstant * source.mass * inverseDistanceCubed);
        potential -= gravitationalConstant * source.mass / distance;
    }

    samples[flatIndex].acceleration = acceleration;
    samples[flatIndex].potential = potential;
}

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

namespace physicsmade::cuda {

FieldGrid evaluateGravityField(
    const std::vector<scene::ObjectState>& sources,
    const FieldGridSpec& spec,
    double gravitationalConstant) {
    if (spec.nx <= 0 || spec.ny <= 0 || spec.nz <= 0) {
        throw std::invalid_argument("field grid dimensions must be positive");
    }

    FieldGrid grid;
    grid.spec = spec;
    grid.samples.resize(spec.sampleCount());

    if (grid.samples.empty() || sources.empty()) {
        return grid;
    }

    std::vector<DeviceSource> hostSources;
    hostSources.reserve(sources.size());
    for (const auto& source : sources) {
        hostSources.push_back({source.mass, source.position.x, source.position.y, source.position.z});
    }

    DeviceSource* deviceSources = nullptr;
    FieldSample* deviceSamples = nullptr;

    checkCuda(cudaMalloc(&deviceSources, hostSources.size() * sizeof(DeviceSource)), "Failed to allocate source buffer");
    checkCuda(cudaMalloc(&deviceSamples, grid.samples.size() * sizeof(FieldSample)), "Failed to allocate sample buffer");

    try {
        checkCuda(
            cudaMemcpy(deviceSources, hostSources.data(), hostSources.size() * sizeof(DeviceSource), cudaMemcpyHostToDevice),
            "Failed to copy sources to device");

        constexpr int kThreadsPerBlock = 256;
        const int totalSamples = static_cast<int>(grid.samples.size());
        const int blocks = (totalSamples + kThreadsPerBlock - 1) / kThreadsPerBlock;

        gravityFieldKernel<<<blocks, kThreadsPerBlock>>>(
            deviceSources,
            static_cast<int>(hostSources.size()),
            deviceSamples,
            spec.nx,
            spec.ny,
            spec.nz,
            spec.origin,
            spec.spacing,
            gravitationalConstant);

        checkCuda(cudaGetLastError(), "Failed to launch gravity field kernel");
        checkCuda(cudaDeviceSynchronize(), "Failed to synchronize gravity field kernel");
        checkCuda(
            cudaMemcpy(grid.samples.data(), deviceSamples, grid.samples.size() * sizeof(FieldSample), cudaMemcpyDeviceToHost),
            "Failed to copy field samples back to host");
    } catch (...) {
        cudaFree(deviceSources);
        cudaFree(deviceSamples);
        throw;
    }

    cudaFree(deviceSources);
    cudaFree(deviceSamples);
    return grid;
}

bool cudaFieldSolverAvailable() noexcept {
    return true;
}

}  // namespace physicsmade::cuda

#include "physicsmade/cuda/gravity_body_forces.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace {

struct DeviceBody {
    double mass;
    double x;
    double y;
    double z;
};

__global__ void gravityBodyForceKernel(
    const DeviceBody* sources,
    int sourceCount,
    physicsmade::math::Vector3* forces,
    double gravitationalConstant) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= sourceCount) {
        return;
    }

    const DeviceBody self = sources[index];
    physicsmade::math::Vector3 accumulated{};

    for (int otherIndex = 0; otherIndex < sourceCount; ++otherIndex) {
        if (index == otherIndex) {
            continue;
        }

        const DeviceBody other = sources[otherIndex];
        const physicsmade::math::Vector3 delta{other.x - self.x, other.y - self.y, other.z - self.z};
        const double distanceSquared = delta.normSquared() + physicsmade::common::kEpsilon;
        const double distance = sqrt(distanceSquared);
        accumulated += delta * (gravitationalConstant * self.mass * other.mass / (distanceSquared * distance));
    }

    forces[index] = accumulated;
}

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

namespace physicsmade::cuda {

std::vector<math::Vector3> accumulateGravityForces(
    const std::vector<scene::ObjectState>& sources,
    double gravitationalConstant) {
    std::vector<math::Vector3> forces(sources.size(), {});
    if (sources.empty()) {
        return forces;
    }

    std::vector<DeviceBody> hostBodies;
    hostBodies.reserve(sources.size());
    for (const auto& source : sources) {
        hostBodies.push_back({source.mass, source.position.x, source.position.y, source.position.z});
    }

    DeviceBody* deviceBodies = nullptr;
    math::Vector3* deviceForces = nullptr;

    checkCuda(cudaMalloc(&deviceBodies, hostBodies.size() * sizeof(DeviceBody)), "Failed to allocate body buffer");
    checkCuda(cudaMalloc(&deviceForces, forces.size() * sizeof(math::Vector3)), "Failed to allocate force buffer");

    try {
        checkCuda(
            cudaMemcpy(deviceBodies, hostBodies.data(), hostBodies.size() * sizeof(DeviceBody), cudaMemcpyHostToDevice),
            "Failed to upload gravity bodies");

        constexpr int kThreadsPerBlock = 256;
        const int blocks = (static_cast<int>(forces.size()) + kThreadsPerBlock - 1) / kThreadsPerBlock;
        gravityBodyForceKernel<<<blocks, kThreadsPerBlock>>>(
            deviceBodies,
            static_cast<int>(hostBodies.size()),
            deviceForces,
            gravitationalConstant);

        checkCuda(cudaGetLastError(), "Failed to launch gravity body kernel");
        checkCuda(cudaDeviceSynchronize(), "Failed to synchronize gravity body kernel");
        checkCuda(
            cudaMemcpy(forces.data(), deviceForces, forces.size() * sizeof(math::Vector3), cudaMemcpyDeviceToHost),
            "Failed to download gravity body forces");
    } catch (...) {
        cudaFree(deviceBodies);
        cudaFree(deviceForces);
        throw;
    }

    cudaFree(deviceBodies);
    cudaFree(deviceForces);
    return forces;
}

bool cudaBodyForceSolverAvailable() noexcept {
    return true;
}

}  // namespace physicsmade::cuda

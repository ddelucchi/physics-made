#include "physicsmade/cuda/render_instances.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {

struct DeviceRenderSource {
    double positionX;
    double positionY;
    double positionZ;
    double orientationW;
    double orientationX;
    double orientationY;
    double orientationZ;
    double radius;
    double velocityX;
    double velocityY;
    double velocityZ;
    double angularVelocityX;
    double angularVelocityY;
    double angularVelocityZ;
    double mass;
    double charge;
    double temperatureKelvin;
    double emissiveIntensity;
};

__global__ void renderPackKernel(
    const DeviceRenderSource* sources,
    int sourceCount,
    physicsmade::cuda::RenderInstance* instances) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= sourceCount) {
        return;
    }

    const DeviceRenderSource source = sources[index];
    const physicsmade::math::Vector3 velocity{source.velocityX, source.velocityY, source.velocityZ};
    const physicsmade::math::Vector3 angularVelocity{source.angularVelocityX, source.angularVelocityY, source.angularVelocityZ};
    instances[index] = {
        {source.positionX, source.positionY, source.positionZ},
        {source.orientationW, source.orientationX, source.orientationY, source.orientationZ},
        source.radius,
        velocity.norm(),
        angularVelocity.norm(),
        source.mass,
        source.charge,
        source.temperatureKelvin,
        source.emissiveIntensity,
    };
}

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

namespace physicsmade::cuda {

RenderInstancePack buildRenderInstancePack(const std::vector<scene::ObjectState>& objects) {
    RenderInstancePack pack{};
    pack.instances.resize(objects.size());
    if (objects.empty()) {
        return pack;
    }

    std::vector<DeviceRenderSource> hostSources;
    hostSources.reserve(objects.size());
    for (const auto& object : objects) {
        hostSources.push_back({
            object.position.x,
            object.position.y,
            object.position.z,
            object.orientation.w,
            object.orientation.x,
            object.orientation.y,
            object.orientation.z,
            object.radius,
            object.velocity.x,
            object.velocity.y,
            object.velocity.z,
            object.angularVelocity.x,
            object.angularVelocity.y,
            object.angularVelocity.z,
            object.mass,
            object.charge,
            object.temperatureKelvin,
            object.emissiveIntensity,
        });
    }

    DeviceRenderSource* deviceSources = nullptr;
    RenderInstance* deviceInstances = nullptr;

    checkCuda(cudaMalloc(&deviceSources, hostSources.size() * sizeof(DeviceRenderSource)), "Failed to allocate render sources");
    checkCuda(cudaMalloc(&deviceInstances, pack.instances.size() * sizeof(RenderInstance)), "Failed to allocate render instances");

    try {
        checkCuda(
            cudaMemcpy(deviceSources, hostSources.data(), hostSources.size() * sizeof(DeviceRenderSource), cudaMemcpyHostToDevice),
            "Failed to upload render sources");

        constexpr int kThreadsPerBlock = 256;
        const int blocks = (static_cast<int>(pack.instances.size()) + kThreadsPerBlock - 1) / kThreadsPerBlock;
        renderPackKernel<<<blocks, kThreadsPerBlock>>>(deviceSources, static_cast<int>(hostSources.size()), deviceInstances);

        checkCuda(cudaGetLastError(), "Failed to launch render packing kernel");
        checkCuda(cudaDeviceSynchronize(), "Failed to synchronize render packing kernel");
        checkCuda(
            cudaMemcpy(pack.instances.data(), deviceInstances, pack.instances.size() * sizeof(RenderInstance), cudaMemcpyDeviceToHost),
            "Failed to download render instances");
    } catch (...) {
        cudaFree(deviceSources);
        cudaFree(deviceInstances);
        throw;
    }

    cudaFree(deviceSources);
    cudaFree(deviceInstances);

    pack.boundsMin = pack.instances.front().position;
    pack.boundsMax = pack.instances.front().position;
    for (const auto& instance : pack.instances) {
        pack.boundsMin.x = std::min(pack.boundsMin.x, instance.position.x);
        pack.boundsMin.y = std::min(pack.boundsMin.y, instance.position.y);
        pack.boundsMin.z = std::min(pack.boundsMin.z, instance.position.z);
        pack.boundsMax.x = std::max(pack.boundsMax.x, instance.position.x);
        pack.boundsMax.y = std::max(pack.boundsMax.y, instance.position.y);
        pack.boundsMax.z = std::max(pack.boundsMax.z, instance.position.z);
    }

    return pack;
}

bool cudaRenderPackingAvailable() noexcept {
    return true;
}

std::vector<GpuRenderInstance> buildGpuRenderInstances(const std::vector<scene::ObjectState>& objects) {
    const auto pack = buildRenderInstancePack(objects);
    std::vector<GpuRenderInstance> gpuInstances;
    gpuInstances.reserve(pack.instances.size());
    for (const auto& instance : pack.instances) {
        gpuInstances.push_back(buildGpuRenderInstance(instance, pack.boundsMin, pack.boundsMax));
    }

    return gpuInstances;
}

}  // namespace physicsmade::cuda

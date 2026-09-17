#include "physicsmade/cuda/scene_buffers.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {

struct DeviceSceneObject {
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

__global__ void packSceneObjectsKernel(
    const DeviceSceneObject* objects,
    const physicsmade::cuda::RenderInstance* renderInstances,
    int count,
    physicsmade::cuda::RenderInstance* instances,
    physicsmade::cuda::GpuRenderInstance* viewerInstances,
    physicsmade::math::Vector3 boundsMin,
    physicsmade::math::Vector3 boundsMax,
    physicsmade::math::Vector3 viewerOrigin) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= count) {
        return;
    }

    const DeviceSceneObject object = objects[index];
    if (renderInstances == nullptr) {
        const physicsmade::math::Vector3 velocity{object.velocityX, object.velocityY, object.velocityZ};
        const physicsmade::math::Vector3 angularVelocity{object.angularVelocityX, object.angularVelocityY, object.angularVelocityZ};
        const physicsmade::cuda::RenderInstance renderInstance{
            {object.positionX, object.positionY, object.positionZ},
            {object.orientationW, object.orientationX, object.orientationY, object.orientationZ},
            object.radius,
            velocity.norm(),
            angularVelocity.norm(),
            object.mass,
            object.charge,
            object.temperatureKelvin,
            object.emissiveIntensity,
        };
        instances[index] = renderInstance;
        viewerInstances[index] = physicsmade::cuda::buildGpuRenderInstance(renderInstance, boundsMin, boundsMax, viewerOrigin);
        return;
    }

    viewerInstances[index] = physicsmade::cuda::buildGpuRenderInstance(renderInstances[index], boundsMin, boundsMax, viewerOrigin);
}

void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

class CudaSceneBufferManager final : public physicsmade::cuda::SceneBufferManager {
  public:
    ~CudaSceneBufferManager() override {
        cudaFree(deviceObjects_);
        cudaFree(deviceInstances_);
        cudaFree(deviceViewerInstances_);
    }

    void reserve(std::size_t capacity) override {
        if (capacity <= capacity_) {
            return;
        }

        cudaFree(deviceObjects_);
        cudaFree(deviceInstances_);
        cudaFree(deviceViewerInstances_);
        checkCuda(cudaMalloc(&deviceObjects_, capacity * sizeof(DeviceSceneObject)), "Failed to allocate persistent scene objects");
        checkCuda(cudaMalloc(&deviceInstances_, capacity * sizeof(physicsmade::cuda::RenderInstance)), "Failed to allocate persistent render instances");
        checkCuda(cudaMalloc(&deviceViewerInstances_, capacity * sizeof(physicsmade::cuda::GpuRenderInstance)), "Failed to allocate persistent viewer instances");
        capacity_ = capacity;
    }

    void setViewerOrigin(const physicsmade::math::Vector3& origin) override {
        viewerOrigin_ = origin;
        repackViewerInstances();
    }

    void update(const std::vector<physicsmade::scene::ObjectState>& objects) override {
        reserve(objects.size());
        objectCount_ = objects.size();

        std::vector<DeviceSceneObject> hostObjects;
        hostObjects.reserve(objects.size());
        if (!objects.empty()) {
            boundsMin_ = objects.front().position;
            boundsMax_ = objects.front().position;
        }

        for (const auto& object : objects) {
            hostObjects.push_back({
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

            boundsMin_.x = std::min(boundsMin_.x, object.position.x);
            boundsMin_.y = std::min(boundsMin_.y, object.position.y);
            boundsMin_.z = std::min(boundsMin_.z, object.position.z);
            boundsMax_.x = std::max(boundsMax_.x, object.position.x);
            boundsMax_.y = std::max(boundsMax_.y, object.position.y);
            boundsMax_.z = std::max(boundsMax_.z, object.position.z);
        }

        if (hostObjects.empty()) {
            return;
        }

        checkCuda(
            cudaMemcpy(deviceObjects_, hostObjects.data(), hostObjects.size() * sizeof(DeviceSceneObject), cudaMemcpyHostToDevice),
            "Failed to upload persistent scene objects");

        constexpr int kThreadsPerBlock = 256;
        const int blocks = (static_cast<int>(hostObjects.size()) + kThreadsPerBlock - 1) / kThreadsPerBlock;
        packSceneObjectsKernel<<<blocks, kThreadsPerBlock>>>(
            deviceObjects_,
            nullptr,
            static_cast<int>(hostObjects.size()),
            deviceInstances_,
            deviceViewerInstances_,
            boundsMin_,
            boundsMax_,
            viewerOrigin_);

        checkCuda(cudaGetLastError(), "Failed to launch persistent scene packing kernel");
        checkCuda(cudaDeviceSynchronize(), "Failed to synchronize persistent scene packing kernel");
    }

    physicsmade::cuda::SceneInteropView interopView() const override {
        physicsmade::cuda::SceneInteropView view{};
        view.backend = "cuda";
        view.deviceResident = true;
        view.gravityBodyBuffer = {reinterpret_cast<std::uintptr_t>(deviceObjects_), sizeof(DeviceSceneObject), objectCount_};
        view.renderInstanceBuffer = {reinterpret_cast<std::uintptr_t>(deviceInstances_), sizeof(physicsmade::cuda::RenderInstance), objectCount_};
        view.viewerInstanceBuffer = {reinterpret_cast<std::uintptr_t>(deviceViewerInstances_), sizeof(physicsmade::cuda::GpuRenderInstance), objectCount_};
        view.boundsMin = boundsMin_;
        view.boundsMax = boundsMax_;
        view.viewerOrigin = viewerOrigin_;
        return view;
    }

    std::vector<physicsmade::cuda::RenderInstance> downloadRenderInstances() const override {
        std::vector<physicsmade::cuda::RenderInstance> instances(objectCount_);
        if (instances.empty()) {
            return instances;
        }

        checkCuda(
            cudaMemcpy(instances.data(), deviceInstances_, instances.size() * sizeof(physicsmade::cuda::RenderInstance), cudaMemcpyDeviceToHost),
            "Failed to download persistent render instances");
        return instances;
    }

    std::vector<physicsmade::cuda::GpuRenderInstance> downloadGpuRenderInstances() const override {
        std::vector<physicsmade::cuda::GpuRenderInstance> instances(objectCount_);
        if (instances.empty()) {
            return instances;
        }

        checkCuda(
            cudaMemcpy(instances.data(), deviceViewerInstances_, instances.size() * sizeof(physicsmade::cuda::GpuRenderInstance), cudaMemcpyDeviceToHost),
            "Failed to download persistent viewer instances");
        return instances;
    }

  private:
    void repackViewerInstances() {
        if (objectCount_ == 0 || deviceInstances_ == nullptr || deviceViewerInstances_ == nullptr) {
            return;
        }

        constexpr int kThreadsPerBlock = 256;
        const int blocks = (static_cast<int>(objectCount_) + kThreadsPerBlock - 1) / kThreadsPerBlock;
        packSceneObjectsKernel<<<blocks, kThreadsPerBlock>>>(
            deviceObjects_,
            deviceInstances_,
            static_cast<int>(objectCount_),
            deviceInstances_,
            deviceViewerInstances_,
            boundsMin_,
            boundsMax_,
            viewerOrigin_);
        checkCuda(cudaGetLastError(), "Failed to launch viewer instance repacking kernel");
        checkCuda(cudaDeviceSynchronize(), "Failed to synchronize viewer instance repacking kernel");
    }

    DeviceSceneObject* deviceObjects_{nullptr};
    physicsmade::cuda::RenderInstance* deviceInstances_{nullptr};
    physicsmade::cuda::GpuRenderInstance* deviceViewerInstances_{nullptr};
    std::size_t capacity_{0};
    std::size_t objectCount_{0};
    physicsmade::math::Vector3 boundsMin_{};
    physicsmade::math::Vector3 boundsMax_{};
    physicsmade::math::Vector3 viewerOrigin_{};
};

}  // namespace

namespace physicsmade::cuda {

std::unique_ptr<SceneBufferManager> createSceneBufferManager() {
    return std::make_unique<CudaSceneBufferManager>();
}

bool cudaSceneBuffersAvailable() noexcept {
    return true;
}

}  // namespace physicsmade::cuda

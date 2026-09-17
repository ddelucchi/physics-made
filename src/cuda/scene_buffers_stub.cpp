#include "physicsmade/cuda/scene_buffers.hpp"

namespace {

class CpuSceneBufferManager final : public physicsmade::cuda::SceneBufferManager {
  public:
    void reserve(std::size_t capacity) override {
        instances_.reserve(capacity);
        gpuInstances_.reserve(capacity);
    }

    void setViewerOrigin(const physicsmade::math::Vector3& origin) override {
        viewerOrigin_ = origin;
        refreshGpuInstances();
    }

    void update(const std::vector<physicsmade::scene::ObjectState>& objects) override {
        const auto pack = physicsmade::cuda::buildRenderInstancePack(objects);
        instances_ = pack.instances;
        boundsMin_ = pack.boundsMin;
        boundsMax_ = pack.boundsMax;
        refreshGpuInstances();
        view_.backend = "cpu";
        view_.deviceResident = false;
        view_.gravityBodyBuffer = {0, 0, objects.size()};
        view_.renderInstanceBuffer = {0, sizeof(physicsmade::cuda::RenderInstance), instances_.size()};
        view_.viewerInstanceBuffer = {0, sizeof(physicsmade::cuda::GpuRenderInstance), gpuInstances_.size()};
        view_.boundsMin = boundsMin_;
        view_.boundsMax = boundsMax_;
        view_.viewerOrigin = viewerOrigin_;
    }

    physicsmade::cuda::SceneInteropView interopView() const override {
        return view_;
    }

    std::vector<physicsmade::cuda::RenderInstance> downloadRenderInstances() const override {
        return instances_;
    }

    std::vector<physicsmade::cuda::GpuRenderInstance> downloadGpuRenderInstances() const override {
        return gpuInstances_;
    }

  private:
    void refreshGpuInstances() {
        gpuInstances_.clear();
        gpuInstances_.reserve(instances_.size());
        for (const auto& instance : instances_) {
            gpuInstances_.push_back(physicsmade::cuda::buildGpuRenderInstance(instance, boundsMin_, boundsMax_, viewerOrigin_));
        }
        view_.viewerInstanceBuffer = {0, sizeof(physicsmade::cuda::GpuRenderInstance), gpuInstances_.size()};
        view_.viewerOrigin = viewerOrigin_;
    }

    std::vector<physicsmade::cuda::RenderInstance> instances_{};
    std::vector<physicsmade::cuda::GpuRenderInstance> gpuInstances_{};
    physicsmade::cuda::SceneInteropView view_{};
    physicsmade::math::Vector3 boundsMin_{};
    physicsmade::math::Vector3 boundsMax_{};
    physicsmade::math::Vector3 viewerOrigin_{};
};

}  // namespace

namespace physicsmade::cuda {

std::unique_ptr<SceneBufferManager> createSceneBufferManager() {
    return std::make_unique<CpuSceneBufferManager>();
}

bool cudaSceneBuffersAvailable() noexcept {
    return false;
}

}  // namespace physicsmade::cuda

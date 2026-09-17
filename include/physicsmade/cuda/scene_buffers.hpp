#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "physicsmade/cuda/render_instances.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::cuda {

struct DeviceBufferView {
    std::uintptr_t deviceAddress{0};
    std::size_t strideBytes{0};
    std::size_t count{0};
};

struct SceneInteropView {
    std::string backend{"cpu"};
    bool deviceResident{false};
    DeviceBufferView gravityBodyBuffer{};
    DeviceBufferView renderInstanceBuffer{};
    DeviceBufferView viewerInstanceBuffer{};
    math::Vector3 boundsMin{};
    math::Vector3 boundsMax{};
    math::Vector3 viewerOrigin{};
};

class SceneBufferManager {
  public:
    virtual ~SceneBufferManager() = default;

    virtual void reserve(std::size_t capacity) = 0;
    virtual void setViewerOrigin(const math::Vector3& origin) = 0;
    virtual void update(const std::vector<scene::ObjectState>& objects) = 0;
    virtual SceneInteropView interopView() const = 0;
    virtual std::vector<RenderInstance> downloadRenderInstances() const = 0;
    virtual std::vector<GpuRenderInstance> downloadGpuRenderInstances() const = 0;
};

std::unique_ptr<SceneBufferManager> createSceneBufferManager();

bool cudaSceneBuffersAvailable() noexcept;

}  // namespace physicsmade::cuda

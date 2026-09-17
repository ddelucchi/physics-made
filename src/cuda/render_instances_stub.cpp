#include "physicsmade/cuda/render_instances.hpp"

#include <algorithm>

namespace physicsmade::cuda {

RenderInstancePack buildRenderInstancePack(const std::vector<scene::ObjectState>& objects) {
    RenderInstancePack pack{};
    pack.instances.reserve(objects.size());

    if (objects.empty()) {
        return pack;
    }

    pack.boundsMin = objects.front().position;
    pack.boundsMax = objects.front().position;

    for (const auto& object : objects) {
        pack.instances.push_back({
            object.position,
            object.orientation,
            object.radius,
            object.velocity.norm(),
            object.angularVelocity.norm(),
            object.mass,
            object.charge,
            object.temperatureKelvin,
            object.emissiveIntensity,
        });

        pack.boundsMin.x = std::min(pack.boundsMin.x, object.position.x);
        pack.boundsMin.y = std::min(pack.boundsMin.y, object.position.y);
        pack.boundsMin.z = std::min(pack.boundsMin.z, object.position.z);
        pack.boundsMax.x = std::max(pack.boundsMax.x, object.position.x);
        pack.boundsMax.y = std::max(pack.boundsMax.y, object.position.y);
        pack.boundsMax.z = std::max(pack.boundsMax.z, object.position.z);
    }

    return pack;
}

bool cudaRenderPackingAvailable() noexcept {
    return false;
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

#pragma once

#include <cmath>
#include <vector>

#include "physicsmade/common/macros.hpp"
#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::cuda {

struct RenderInstance {
    math::Vector3 position{};
    math::Quaternion orientation{};
    double radius{1.0};
    double speed{0.0};
    double angularSpeed{0.0};
    double mass{0.0};
    double charge{0.0};
    double temperatureKelvin{300.0};
    double emissiveIntensity{1.0};
};

struct GpuRenderInstance {
    float positionRadius[4]{};
    float orientation[4]{};
    float color[4]{};
    float dynamics[4]{};
};

struct RenderInstancePack {
    std::vector<RenderInstance> instances{};
    math::Vector3 boundsMin{};
    math::Vector3 boundsMax{};
};

PHYSICSMADE_HD constexpr double renderScalarAbs(double value) noexcept {
    return value < 0.0 ? -value : value;
}

PHYSICSMADE_HD constexpr double renderScalarMax(double left, double right) noexcept {
    return left > right ? left : right;
}

PHYSICSMADE_HD constexpr double renderScalarMin(double left, double right) noexcept {
    return left < right ? left : right;
}

PHYSICSMADE_HD constexpr double renderScalarClamp(double value, double minValue, double maxValue) noexcept {
    return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

PHYSICSMADE_HD inline double renderSceneExtent(
    const math::Vector3& boundsMin,
    const math::Vector3& boundsMax) noexcept {
    const auto span = boundsMax - boundsMin;
    return renderScalarMax(
        renderScalarMax(renderScalarAbs(span.x), renderScalarAbs(span.y)),
        renderScalarMax(renderScalarAbs(span.z), 1.0));
}

PHYSICSMADE_HD inline double renderVisualRadius(
    double radius,
    const math::Vector3& boundsMin,
    const math::Vector3& boundsMax) noexcept {
    const double extent = renderSceneExtent(boundsMin, boundsMax);
    const double minimumRadius = renderScalarMax(0.05, 0.01 * extent);
    const double maximumRadius = renderScalarMax(minimumRadius * 5.0, 0.18 * extent);
    return renderScalarClamp(radius, minimumRadius, maximumRadius);
}

PHYSICSMADE_HD inline GpuRenderInstance buildGpuRenderInstance(
    const RenderInstance& instance,
    const math::Vector3& boundsMin,
    const math::Vector3& boundsMax,
    const math::Vector3& viewerOrigin = {}) noexcept {
    const auto normalizedOrientation = instance.orientation.normalized();
    const double visualRadius = renderVisualRadius(instance.radius, boundsMin, boundsMax);
    const double chargeMix = renderScalarClamp(renderScalarAbs(instance.charge) * 2.5e5, 0.0, 1.0);
    const double thermalMix = renderScalarClamp((instance.temperatureKelvin - 250.0) / 6000.0, 0.0, 1.0);
    const double emissiveBoost = renderScalarClamp(instance.emissiveIntensity / 12.0, 0.0, 1.0);

    double red = 0.2 + (0.6 * thermalMix);
    double green = 0.35 + (0.35 * (1.0 - chargeMix));
    double blue = 0.45 + (0.45 * chargeMix);
    if (instance.charge < 0.0) {
        red = 0.9 - (0.4 * thermalMix);
        green = 0.35;
        blue = 0.25 + (0.45 * chargeMix);
    }

    const double emissiveColorBoost = 0.35 * emissiveBoost;
    red = renderScalarClamp(red + emissiveColorBoost, 0.0, 1.0);
    green = renderScalarClamp(green + emissiveColorBoost, 0.0, 1.0);
    blue = renderScalarClamp(blue + emissiveColorBoost, 0.0, 1.0);
    const double axisLength = visualRadius * (1.4 + renderScalarClamp(0.15 * instance.angularSpeed, 0.0, 1.0));

    const auto relativePosition = instance.position - viewerOrigin;
    GpuRenderInstance gpuInstance{};
    gpuInstance.positionRadius[0] = static_cast<float>(relativePosition.x);
    gpuInstance.positionRadius[1] = static_cast<float>(relativePosition.y);
    gpuInstance.positionRadius[2] = static_cast<float>(relativePosition.z);
    gpuInstance.positionRadius[3] = static_cast<float>(visualRadius);
    gpuInstance.orientation[0] = static_cast<float>(normalizedOrientation.x);
    gpuInstance.orientation[1] = static_cast<float>(normalizedOrientation.y);
    gpuInstance.orientation[2] = static_cast<float>(normalizedOrientation.z);
    gpuInstance.orientation[3] = static_cast<float>(normalizedOrientation.w);
    gpuInstance.color[0] = static_cast<float>(red);
    gpuInstance.color[1] = static_cast<float>(green);
    gpuInstance.color[2] = static_cast<float>(blue);
    gpuInstance.color[3] = 1.0f;
    gpuInstance.dynamics[0] = static_cast<float>(axisLength);
    gpuInstance.dynamics[1] = static_cast<float>(instance.speed);
    gpuInstance.dynamics[2] = static_cast<float>(instance.angularSpeed);
    gpuInstance.dynamics[3] = static_cast<float>(instance.emissiveIntensity);
    return gpuInstance;
}

RenderInstancePack buildRenderInstancePack(const std::vector<scene::ObjectState>& objects);
std::vector<GpuRenderInstance> buildGpuRenderInstances(const std::vector<scene::ObjectState>& objects);

bool cudaRenderPackingAvailable() noexcept;

}  // namespace physicsmade::cuda

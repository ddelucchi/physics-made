#include "physicsmade/camera/orbit_camera.hpp"

#include <algorithm>
#include <cmath>

namespace physicsmade::camera {

OrbitCamera::OrbitCamera(double radius, double azimuthRadians, double elevationRadians)
    : radius_(radius), azimuthRadians_(azimuthRadians), elevationRadians_(elevationRadians) {
    setRadius(radius);
    orbit(0.0, 0.0);
}

void OrbitCamera::orbit(double deltaAzimuthRadians, double deltaElevationRadians) {
    azimuthRadians_ += deltaAzimuthRadians;
    elevationRadians_ += deltaElevationRadians;

    const double elevationLimit = (common::kPi * 0.5) - 1.0e-3;
    elevationRadians_ = std::clamp(elevationRadians_, -elevationLimit, elevationLimit);
}

void OrbitCamera::setRadius(double radius) {
    radius_ = std::max(radius, 1.0e-6);
}

void OrbitCamera::setTargetOffset(const math::Vector3& targetOffset) noexcept {
    targetOffset_ = targetOffset;
}

CameraPose OrbitCamera::pose(const math::Vector3& targetWorldPosition) const noexcept {
    const math::Vector3 focus = targetWorldPosition + targetOffset_;
    const double cosElevation = std::cos(elevationRadians_);

    const math::Vector3 offset{
        radius_ * cosElevation * std::cos(azimuthRadians_),
        radius_ * std::sin(elevationRadians_),
        radius_ * cosElevation * std::sin(azimuthRadians_),
    };

    const math::Vector3 position = focus + offset;
    const math::Vector3 forward = (focus - position).normalized();

    const math::Vector3 worldUp{0.0, 1.0, 0.0};
    math::Vector3 right = forward.cross(worldUp).normalized();
    if (right.normSquared() <= common::kEpsilon) {
        right = {1.0, 0.0, 0.0};
    }

    const math::Vector3 up = right.cross(forward).normalized();
    return {position, forward, right, up};
}

}  // namespace physicsmade::camera

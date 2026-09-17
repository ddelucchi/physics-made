#pragma once

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::camera {

struct CameraPose {
    math::Vector3 position{};
    math::Vector3 forward{};
    math::Vector3 right{};
    math::Vector3 up{};
};

class OrbitCamera {
  public:
    OrbitCamera(double radius = 10.0, double azimuthRadians = 0.0, double elevationRadians = 0.4);

    void orbit(double deltaAzimuthRadians, double deltaElevationRadians);
    void setRadius(double radius);
    void setTargetOffset(const math::Vector3& targetOffset) noexcept;

    double radius() const noexcept {
        return radius_;
    }

    double azimuth() const noexcept {
        return azimuthRadians_;
    }

    double elevation() const noexcept {
        return elevationRadians_;
    }

    CameraPose pose(const math::Vector3& targetWorldPosition) const noexcept;

  private:
    double radius_{10.0};
    double azimuthRadians_{0.0};
    double elevationRadians_{0.4};
    math::Vector3 targetOffset_{};
};

}  // namespace physicsmade::camera

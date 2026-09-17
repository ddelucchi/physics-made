#pragma once

#include <string>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/four_vector.hpp"
#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::scene {

struct ObjectState {
    std::string name;
    double mass{1.0};
    double charge{0.0};
    double radius{1.0};
    bool pinned{false};
    math::Vector3 position{};
    math::Vector3 velocity{};
    math::Quaternion orientation{};
    math::Vector3 angularVelocity{};
    double coordinateTimeSeconds{0.0};
    double properTimeSeconds{0.0};
    math::Vector3 bodyInertiaDiagonal{};
    double dragCoefficient{0.47};
    double temperatureKelvin{300.0};
    double emissiveIntensity{1.0};
    double restitution{0.4};
    double frictionCoefficient{0.6};

    math::FourVector worldlinePoint() const noexcept {
        return worldlinePoint(coordinateTimeSeconds);
    }

    math::FourVector worldlinePoint(double timeSeconds) const noexcept {
        return {common::kSpeedOfLight * timeSeconds, position.x, position.y, position.z};
    }
};

}  // namespace physicsmade::scene

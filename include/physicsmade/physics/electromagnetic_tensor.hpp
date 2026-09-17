#pragma once

#include <array>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::physics {

using ElectromagneticTensor4 = std::array<std::array<double, 4>, 4>;

struct ElectromagneticFieldTensor {
    ElectromagneticTensor4 components{};

    static ElectromagneticFieldTensor fromElectricMagnetic(
        const math::Vector3& electricField,
        const math::Vector3& magneticField,
        double speedOfLight = common::kSpeedOfLight) noexcept {
        const double c = speedOfLight;
        ElectromagneticFieldTensor tensor{};

        tensor.components[0][1] = electricField.x / c;
        tensor.components[0][2] = electricField.y / c;
        tensor.components[0][3] = electricField.z / c;
        tensor.components[1][0] = -tensor.components[0][1];
        tensor.components[2][0] = -tensor.components[0][2];
        tensor.components[3][0] = -tensor.components[0][3];

        tensor.components[1][2] = -magneticField.z;
        tensor.components[2][1] = magneticField.z;
        tensor.components[1][3] = magneticField.y;
        tensor.components[3][1] = -magneticField.y;
        tensor.components[2][3] = -magneticField.x;
        tensor.components[3][2] = magneticField.x;
        return tensor;
    }

    math::Vector3 electricField(double speedOfLight = common::kSpeedOfLight) const noexcept {
        return {
            components[0][1] * speedOfLight,
            components[0][2] * speedOfLight,
            components[0][3] * speedOfLight,
        };
    }

    math::Vector3 magneticField() const noexcept {
        return {
            -components[2][3],
            components[1][3],
            -components[1][2],
        };
    }

    math::Vector3 lorentzForce(
        double charge,
        const math::Vector3& velocity,
        double speedOfLight = common::kSpeedOfLight) const noexcept {
        const auto electric = electricField(speedOfLight);
        const auto magnetic = magneticField();
        return charge * (electric + velocity.cross(magnetic));
    }
};

}  // namespace physicsmade::physics

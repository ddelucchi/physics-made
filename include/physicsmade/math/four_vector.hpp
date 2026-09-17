#pragma once

#include "physicsmade/math/vector3.hpp"

namespace physicsmade::math {

struct FourVector {
    double t{0.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    constexpr FourVector() noexcept = default;
    constexpr FourVector(double tIn, double xIn, double yIn, double zIn) noexcept : t(tIn), x(xIn), y(yIn), z(zIn) {}

    constexpr FourVector operator+(const FourVector& other) const noexcept {
        return {t + other.t, x + other.x, y + other.y, z + other.z};
    }

    constexpr FourVector operator-(const FourVector& other) const noexcept {
        return {t - other.t, x - other.x, y - other.y, z - other.z};
    }

    constexpr FourVector operator*(double scalar) const noexcept {
        return {t * scalar, x * scalar, y * scalar, z * scalar};
    }

    constexpr Vector3 spatial() const noexcept {
        return {x, y, z};
    }
};

}  // namespace physicsmade::math

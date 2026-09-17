#pragma once

#include <cmath>

#include "physicsmade/common/config.hpp"
#include "physicsmade/common/macros.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::math {

struct Quaternion {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};

    PHYSICSMADE_HD constexpr Quaternion() noexcept : w(1.0), x(0.0), y(0.0), z(0.0) {}
    PHYSICSMADE_HD constexpr Quaternion(double wIn, double xIn, double yIn, double zIn) noexcept
        : w(wIn), x(xIn), y(yIn), z(zIn) {}

    PHYSICSMADE_HD static constexpr Quaternion identity() noexcept {
        return {};
    }

    PHYSICSMADE_HD static Quaternion fromAxisAngle(const Vector3& axis, double angle) noexcept {
        const Vector3 unitAxis = axis.normalized();
        if (unitAxis.normSquared() <= common::kEpsilon || std::abs(angle) <= common::kEpsilon) {
            return identity();
        }

        const double halfAngle = 0.5 * angle;
        const double sine = ::sin(halfAngle);
        return {
            ::cos(halfAngle),
            unitAxis.x * sine,
            unitAxis.y * sine,
            unitAxis.z * sine,
        };
    }

    PHYSICSMADE_HD constexpr Quaternion operator*(const Quaternion& other) const noexcept {
        return {
            (w * other.w) - (x * other.x) - (y * other.y) - (z * other.z),
            (w * other.x) + (x * other.w) + (y * other.z) - (z * other.y),
            (w * other.y) - (x * other.z) + (y * other.w) + (z * other.x),
            (w * other.z) + (x * other.y) - (y * other.x) + (z * other.w),
        };
    }

    PHYSICSMADE_HD constexpr Quaternion operator*(double scalar) const noexcept {
        return {w * scalar, x * scalar, y * scalar, z * scalar};
    }

    PHYSICSMADE_HD constexpr Quaternion operator+(const Quaternion& other) const noexcept {
        return {w + other.w, x + other.x, y + other.y, z + other.z};
    }

    PHYSICSMADE_HD constexpr Quaternion conjugate() const noexcept {
        return {w, -x, -y, -z};
    }

    PHYSICSMADE_HD constexpr double normSquared() const noexcept {
        return (w * w) + (x * x) + (y * y) + (z * z);
    }

    PHYSICSMADE_HD double norm() const noexcept {
        return ::sqrt(normSquared());
    }

    PHYSICSMADE_HD Quaternion normalized(double epsilon = common::kEpsilon) const noexcept {
        const double magnitude = norm();
        if (magnitude <= epsilon) {
            return identity();
        }
        return (*this) * (1.0 / magnitude);
    }

    PHYSICSMADE_HD Vector3 rotate(const Vector3& vector) const noexcept {
        const Quaternion rotated = (*this) * Quaternion{0.0, vector.x, vector.y, vector.z} * conjugate();
        return {rotated.x, rotated.y, rotated.z};
    }
};

PHYSICSMADE_HD constexpr Quaternion operator*(double scalar, const Quaternion& quaternion) noexcept {
    return quaternion * scalar;
}

}  // namespace physicsmade::math
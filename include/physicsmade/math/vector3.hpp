#pragma once

#include <cmath>

#include "physicsmade/common/config.hpp"
#include "physicsmade/common/macros.hpp"

namespace physicsmade::math {

struct Vector3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    PHYSICSMADE_HD constexpr Vector3() noexcept : x(0.0), y(0.0), z(0.0) {}
    PHYSICSMADE_HD constexpr Vector3(double xIn, double yIn, double zIn) noexcept : x(xIn), y(yIn), z(zIn) {}

    PHYSICSMADE_HD constexpr Vector3 operator+(const Vector3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }

    PHYSICSMADE_HD constexpr Vector3 operator-(const Vector3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }

    PHYSICSMADE_HD constexpr Vector3 operator-() const noexcept {
        return {-x, -y, -z};
    }

    PHYSICSMADE_HD constexpr Vector3 operator*(double scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }

    PHYSICSMADE_HD constexpr Vector3 operator/(double scalar) const noexcept {
        return {x / scalar, y / scalar, z / scalar};
    }

    PHYSICSMADE_HD constexpr Vector3& operator+=(const Vector3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    PHYSICSMADE_HD constexpr Vector3& operator-=(const Vector3& other) noexcept {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    PHYSICSMADE_HD constexpr Vector3& operator*=(double scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    PHYSICSMADE_HD constexpr double dot(const Vector3& other) const noexcept {
        return (x * other.x) + (y * other.y) + (z * other.z);
    }

    PHYSICSMADE_HD constexpr Vector3 cross(const Vector3& other) const noexcept {
        return {
            (y * other.z) - (z * other.y),
            (z * other.x) - (x * other.z),
            (x * other.y) - (y * other.x),
        };
    }

    PHYSICSMADE_HD constexpr double normSquared() const noexcept {
        return dot(*this);
    }

    PHYSICSMADE_HD double norm() const noexcept {
        return ::sqrt(normSquared());
    }

    PHYSICSMADE_HD Vector3 normalized(double epsilon = common::kEpsilon) const noexcept {
        const double length = norm();
        if (length <= epsilon) {
            return {};
        }
        return *this / length;
    }
};

PHYSICSMADE_HD constexpr Vector3 operator*(double scalar, const Vector3& vector) noexcept {
    return vector * scalar;
}

}  // namespace physicsmade::math

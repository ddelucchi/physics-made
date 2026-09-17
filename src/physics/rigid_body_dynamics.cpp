#include "physicsmade/physics/rigid_body_dynamics.hpp"

#include <algorithm>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/quaternion.hpp"

namespace {

double solidSphereInertia(const physicsmade::scene::ObjectState& state) noexcept {
    return 0.4 * state.mass * state.radius * state.radius;
}

physicsmade::math::Vector3 elementwiseMultiply(
    const physicsmade::math::Vector3& left,
    const physicsmade::math::Vector3& right) noexcept {
    return {left.x * right.x, left.y * right.y, left.z * right.z};
}

}  // namespace

namespace physicsmade::physics {

math::Vector3 bodyInertiaDiagonal(const scene::ObjectState& state) noexcept {
    const double sphereInertia = solidSphereInertia(state);
    return {
        state.bodyInertiaDiagonal.x > common::kEpsilon ? state.bodyInertiaDiagonal.x : sphereInertia,
        state.bodyInertiaDiagonal.y > common::kEpsilon ? state.bodyInertiaDiagonal.y : sphereInertia,
        state.bodyInertiaDiagonal.z > common::kEpsilon ? state.bodyInertiaDiagonal.z : sphereInertia,
    };
}

math::Vector3 inverseBodyInertiaDiagonal(const scene::ObjectState& state) noexcept {
    if (state.pinned || state.mass <= common::kEpsilon) {
        return {};
    }

    const auto inertia = bodyInertiaDiagonal(state);
    return {
        inertia.x > common::kEpsilon ? (1.0 / inertia.x) : 0.0,
        inertia.y > common::kEpsilon ? (1.0 / inertia.y) : 0.0,
        inertia.z > common::kEpsilon ? (1.0 / inertia.z) : 0.0,
    };
}

math::Vector3 applyInertiaWorld(const scene::ObjectState& state, const math::Vector3& worldVector) noexcept {
    const auto localVector = state.orientation.conjugate().rotate(worldVector);
    const auto localResult = elementwiseMultiply(bodyInertiaDiagonal(state), localVector);
    return state.orientation.rotate(localResult);
}

math::Vector3 applyInverseInertiaWorld(const scene::ObjectState& state, const math::Vector3& worldVector) noexcept {
    const auto localVector = state.orientation.conjugate().rotate(worldVector);
    const auto localResult = elementwiseMultiply(inverseBodyInertiaDiagonal(state), localVector);
    return state.orientation.rotate(localResult);
}

math::Vector3 worldPoint(const scene::ObjectState& state, const math::Vector3& localPoint) noexcept {
    return state.position + state.orientation.rotate(localPoint);
}

math::Vector3 worldDirection(const scene::ObjectState& state, const math::Vector3& localDirection) noexcept {
    return state.orientation.rotate(localDirection).normalized();
}

math::Vector3 velocityAtPoint(const scene::ObjectState& state, const math::Vector3& worldOffsetFromCenter) noexcept {
    return state.velocity + state.angularVelocity.cross(worldOffsetFromCenter);
}

math::Vector3 angularMomentum(const scene::ObjectState& state) noexcept {
    return applyInertiaWorld(state, state.angularVelocity);
}

double rotationalKineticEnergy(const scene::ObjectState& state) noexcept {
    return 0.5 * state.angularVelocity.dot(angularMomentum(state));
}

void applyAngularImpulse(scene::ObjectState& state, const math::Vector3& angularImpulse) noexcept {
    if (state.pinned) {
        return;
    }

    state.angularVelocity += applyInverseInertiaWorld(state, angularImpulse);
}

void applyImpulse(
    scene::ObjectState& state,
    const math::Vector3& impulse,
    const math::Vector3& worldOffsetFromCenter) noexcept {
    if (state.pinned || state.mass <= common::kEpsilon) {
        return;
    }

    state.velocity += impulse / state.mass;
    applyAngularImpulse(state, worldOffsetFromCenter.cross(impulse));
}

void integrateOrientation(scene::ObjectState& state, double dtSeconds) noexcept {
    if (state.pinned) {
        return;
    }

    const double angularSpeed = state.angularVelocity.norm();
    if (angularSpeed <= common::kEpsilon || dtSeconds <= common::kEpsilon) {
        return;
    }

    const auto delta = math::Quaternion::fromAxisAngle(state.angularVelocity / angularSpeed, angularSpeed * dtSeconds);
    state.orientation = (delta * state.orientation).normalized();
}

}  // namespace physicsmade::physics
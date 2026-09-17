#pragma once

#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::physics {

math::Vector3 bodyInertiaDiagonal(const scene::ObjectState& state) noexcept;
math::Vector3 inverseBodyInertiaDiagonal(const scene::ObjectState& state) noexcept;
math::Vector3 applyInertiaWorld(const scene::ObjectState& state, const math::Vector3& worldVector) noexcept;
math::Vector3 applyInverseInertiaWorld(const scene::ObjectState& state, const math::Vector3& worldVector) noexcept;

math::Vector3 worldPoint(const scene::ObjectState& state, const math::Vector3& localPoint) noexcept;
math::Vector3 worldDirection(const scene::ObjectState& state, const math::Vector3& localDirection) noexcept;
math::Vector3 velocityAtPoint(const scene::ObjectState& state, const math::Vector3& worldOffsetFromCenter) noexcept;

math::Vector3 angularMomentum(const scene::ObjectState& state) noexcept;
double rotationalKineticEnergy(const scene::ObjectState& state) noexcept;

void applyAngularImpulse(scene::ObjectState& state, const math::Vector3& angularImpulse) noexcept;
void applyImpulse(
    scene::ObjectState& state,
    const math::Vector3& impulse,
    const math::Vector3& worldOffsetFromCenter = {}) noexcept;
void integrateOrientation(scene::ObjectState& state, double dtSeconds) noexcept;

}  // namespace physicsmade::physics
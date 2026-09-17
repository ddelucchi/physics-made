#include "physicsmade/physics/constraints.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/physics/rigid_body_dynamics.hpp"

namespace {

double inverseMass(const physicsmade::scene::ObjectState& state) noexcept {
    if (state.pinned || state.mass <= physicsmade::common::kEpsilon) {
        return 0.0;
    }

    return 1.0 / state.mass;
}

double effectiveMassAlongAxis(
    const physicsmade::scene::ObjectState& state,
    const physicsmade::math::Vector3& worldOffset,
    const physicsmade::math::Vector3& axis) noexcept {
    const double invMass = inverseMass(state);
    const auto angularComponent = physicsmade::physics::applyInverseInertiaWorld(state, worldOffset.cross(axis)).cross(worldOffset);
    return invMass + axis.dot(angularComponent);
}

double angularEffectiveMassAlongAxis(
    const physicsmade::scene::ObjectState& state,
    const physicsmade::math::Vector3& axis) noexcept {
    if (state.pinned || axis.normSquared() <= physicsmade::common::kEpsilon) {
        return 0.0;
    }

    return axis.dot(physicsmade::physics::applyInverseInertiaWorld(state, axis));
}

void applyPairImpulse(
    physicsmade::scene::ObjectState& left,
    physicsmade::scene::ObjectState& right,
    const physicsmade::math::Vector3& leftOffset,
    const physicsmade::math::Vector3& rightOffset,
    const physicsmade::math::Vector3& impulse) noexcept {
    physicsmade::physics::applyImpulse(left, -impulse, leftOffset);
    physicsmade::physics::applyImpulse(right, impulse, rightOffset);
}

void applyOrientationCorrection(
    physicsmade::scene::ObjectState& state,
    const physicsmade::math::Vector3& worldAxis,
    double angle) noexcept {
    const double axisNorm = worldAxis.norm();
    if (state.pinned || axisNorm <= physicsmade::common::kEpsilon || std::abs(angle) <= physicsmade::common::kEpsilon) {
        return;
    }

    const auto delta = physicsmade::math::Quaternion::fromAxisAngle(worldAxis / axisNorm, angle);
    state.orientation = (delta * state.orientation).normalized();
}

physicsmade::math::Vector3 averageAxis(
    const physicsmade::math::Vector3& leftAxis,
    const physicsmade::math::Vector3& rightAxis) noexcept {
    const auto combined = leftAxis + rightAxis;
    if (combined.normSquared() > physicsmade::common::kEpsilon) {
        return combined.normalized();
    }

    if (leftAxis.normSquared() > physicsmade::common::kEpsilon) {
        return leftAxis.normalized();
    }

    if (rightAxis.normSquared() > physicsmade::common::kEpsilon) {
        return rightAxis.normalized();
    }

    return {1.0, 0.0, 0.0};
}

physicsmade::math::Vector3 projectOntoPlane(
    const physicsmade::math::Vector3& vector,
    const physicsmade::math::Vector3& planeNormal) noexcept {
    return vector - (planeNormal * vector.dot(planeNormal));
}

double signedAngleAboutAxis(
    const physicsmade::math::Vector3& leftReference,
    const physicsmade::math::Vector3& rightReference,
    const physicsmade::math::Vector3& axis) noexcept {
    const auto projectedLeft = projectOntoPlane(leftReference, axis);
    const auto projectedRight = projectOntoPlane(rightReference, axis);
    const double leftNorm = projectedLeft.norm();
    const double rightNorm = projectedRight.norm();
    if (leftNorm <= physicsmade::common::kEpsilon || rightNorm <= physicsmade::common::kEpsilon) {
        return 0.0;
    }

    const auto leftUnit = projectedLeft / leftNorm;
    const auto rightUnit = projectedRight / rightNorm;
    const double cosine = std::clamp(leftUnit.dot(rightUnit), -1.0, 1.0);
    const double sine = axis.dot(leftUnit.cross(rightUnit));
    return std::atan2(sine, cosine);
}

void solveBallJointSpec(
    std::vector<physicsmade::scene::ObjectState>& states,
    const physicsmade::physics::BallJointConstraintSpec& spec,
    double dtSeconds) {
    if (spec.leftIndex >= states.size() || spec.rightIndex >= states.size()) {
        return;
    }

    auto& left = states[spec.leftIndex];
    auto& right = states[spec.rightIndex];
    const auto leftAnchor = physicsmade::physics::worldPoint(left, spec.leftLocalAnchor);
    const auto rightAnchor = physicsmade::physics::worldPoint(right, spec.rightLocalAnchor);
    const auto leftOffset = leftAnchor - left.position;
    const auto rightOffset = rightAnchor - right.position;
    const auto anchorError = rightAnchor - leftAnchor;

    constexpr std::array<physicsmade::math::Vector3, 3> basis{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};

    for (const auto& axis : basis) {
        const auto relativeVelocity = physicsmade::physics::velocityAtPoint(right, rightOffset) - physicsmade::physics::velocityAtPoint(left, leftOffset);
        const double cDot = relativeVelocity.dot(axis);
        const double positionalError = anchorError.dot(axis);
        const double denominator =
            effectiveMassAlongAxis(left, leftOffset, axis) +
            effectiveMassAlongAxis(right, rightOffset, axis);
        if (denominator <= physicsmade::common::kEpsilon) {
            continue;
        }

        const double bias = (dtSeconds > physicsmade::common::kEpsilon)
                                ? ((spec.stiffness * positionalError) / dtSeconds)
                                : 0.0;
        const double damping = spec.damping * cDot;
        const double impulseMagnitude = -(cDot + bias + damping) / denominator;
        applyPairImpulse(left, right, leftOffset, rightOffset, axis * impulseMagnitude);
    }

    const double totalInvMass = inverseMass(left) + inverseMass(right);
    if (totalInvMass > physicsmade::common::kEpsilon) {
        const auto correction = (spec.stiffness * anchorError) / totalInvMass;
        if (!left.pinned) {
            left.position += correction * inverseMass(left);
        }
        if (!right.pinned) {
            right.position -= correction * inverseMass(right);
        }
    }
}

}  // namespace

namespace physicsmade::physics {

std::string DistanceConstraint::name() const {
    return "distance-constraint";
}

void DistanceConstraint::solve(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const spacetime::SpacetimeModel& spacetime) const {
    (void)spacetime;

    if (spec_.leftIndex >= states.size() || spec_.rightIndex >= states.size()) {
        return;
    }

    auto& left = states[spec_.leftIndex];
    auto& right = states[spec_.rightIndex];
    const math::Vector3 delta = right.position - left.position;
    const double distance = std::sqrt(delta.normSquared() + common::kEpsilon);
    if (distance <= common::kEpsilon) {
        return;
    }

    const math::Vector3 direction = delta / distance;
    const double invMassLeft = inverseMass(left);
    const double invMassRight = inverseMass(right);
    const double totalInvMass = invMassLeft + invMassRight;
    if (totalInvMass <= common::kEpsilon) {
        return;
    }

    const double error = distance - spec_.targetDistance;
    const double correctionMagnitude = spec_.stiffness * error / totalInvMass;
    const math::Vector3 correction = direction * correctionMagnitude;

    left.position += correction * invMassLeft;
    right.position -= correction * invMassRight;

    const double relativeNormalSpeed = (right.velocity - left.velocity).dot(direction);
    const double bias = (dtSeconds > common::kEpsilon) ? (0.1 * error / dtSeconds) : 0.0;
    const double impulseMagnitude = -(relativeNormalSpeed + bias) / totalInvMass;
    const math::Vector3 impulse = direction * impulseMagnitude;

    left.velocity -= impulse * invMassLeft;
    right.velocity += impulse * invMassRight;
}

std::string BallJointConstraint::name() const {
    return "ball-joint";
}

void BallJointConstraint::solve(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const spacetime::SpacetimeModel& spacetime) const {
    (void)spacetime;
    solveBallJointSpec(states, spec_, dtSeconds);
}

std::string HingeConstraint::name() const {
    return "hinge-joint";
}

void HingeConstraint::solve(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const spacetime::SpacetimeModel& spacetime) const {
    (void)spacetime;
    if (spec_.anchor.leftIndex >= states.size() || spec_.anchor.rightIndex >= states.size()) {
        return;
    }

    solveBallJointSpec(states, spec_.anchor, dtSeconds);

    auto& left = states[spec_.anchor.leftIndex];
    auto& right = states[spec_.anchor.rightIndex];
    const auto leftAxis = worldDirection(left, spec_.leftLocalAxis);
    const auto rightAxis = worldDirection(right, spec_.rightLocalAxis);
    const auto rotationalError = leftAxis.cross(rightAxis);
    const double errorMagnitude = rotationalError.norm();

    if (errorMagnitude > common::kEpsilon) {
        const auto axis = rotationalError / errorMagnitude;
        const double denominator = angularEffectiveMassAlongAxis(left, axis) + angularEffectiveMassAlongAxis(right, axis);
        if (denominator > common::kEpsilon) {
            const double cDot = (right.angularVelocity - left.angularVelocity).dot(axis);
            const double bias = (dtSeconds > common::kEpsilon)
                                    ? ((spec_.angularStiffness * errorMagnitude) / dtSeconds)
                                    : 0.0;
            const double impulseMagnitude = -(cDot + bias) / denominator;
            const auto angularImpulse = axis * impulseMagnitude;
            applyAngularImpulse(left, -angularImpulse);
            applyAngularImpulse(right, angularImpulse);
        }

        const double correctionAngle = std::min(spec_.angularStiffness * errorMagnitude, 0.25 * common::kPi);
        applyOrientationCorrection(left, axis, 0.5 * correctionAngle);
        applyOrientationCorrection(right, axis, -0.5 * correctionAngle);
    }

    const auto hingeAxis = averageAxis(
        worldDirection(left, spec_.leftLocalAxis),
        worldDirection(right, spec_.rightLocalAxis));
    const double hingeDenominator = angularEffectiveMassAlongAxis(left, hingeAxis) + angularEffectiveMassAlongAxis(right, hingeAxis);
    if (hingeDenominator <= common::kEpsilon) {
        return;
    }

    const auto leftReference = worldDirection(left, spec_.leftLocalReference);
    const auto rightReference = worldDirection(right, spec_.rightLocalReference);
    const double hingeAngle = signedAngleAboutAxis(leftReference, rightReference, hingeAxis);

    if (spec_.limitsEnabled) {
        double angleError = 0.0;
        if (hingeAngle < spec_.lowerAngleLimit) {
            angleError = hingeAngle - spec_.lowerAngleLimit;
        } else if (hingeAngle > spec_.upperAngleLimit) {
            angleError = hingeAngle - spec_.upperAngleLimit;
        }

        if (std::abs(angleError) > common::kEpsilon) {
            const double cDot = (right.angularVelocity - left.angularVelocity).dot(hingeAxis);
            const double bias = (dtSeconds > common::kEpsilon)
                                    ? ((spec_.angularStiffness * angleError) / dtSeconds)
                                    : 0.0;
            const double impulseMagnitude = -(cDot + bias) / hingeDenominator;
            const auto angularImpulse = hingeAxis * impulseMagnitude;
            applyAngularImpulse(left, -angularImpulse);
            applyAngularImpulse(right, angularImpulse);

            const double correctionAngle = std::clamp(
                spec_.angularStiffness * angleError,
                -0.25 * common::kPi,
                0.25 * common::kPi);
            applyOrientationCorrection(left, hingeAxis, 0.5 * correctionAngle);
            applyOrientationCorrection(right, hingeAxis, -0.5 * correctionAngle);
        }
    }

    if (spec_.motorEnabled && spec_.maxMotorTorque > common::kEpsilon) {
        double targetAngularSpeed = spec_.targetAngularSpeed;
        if (spec_.limitsEnabled) {
            if (hingeAngle >= (spec_.upperAngleLimit - 1.0e-4) && targetAngularSpeed > 0.0) {
                targetAngularSpeed = 0.0;
            }
            if (hingeAngle <= (spec_.lowerAngleLimit + 1.0e-4) && targetAngularSpeed < 0.0) {
                targetAngularSpeed = 0.0;
            }
        }

        const double relativeAngularSpeed = (right.angularVelocity - left.angularVelocity).dot(hingeAxis);
        const double unclampedImpulse = (targetAngularSpeed - relativeAngularSpeed) / hingeDenominator;
        const double impulseLimit = spec_.maxMotorTorque * std::max(dtSeconds, 0.0);
        const double impulseMagnitude = std::clamp(unclampedImpulse, -impulseLimit, impulseLimit);
        const auto angularImpulse = hingeAxis * impulseMagnitude;
        applyAngularImpulse(left, -angularImpulse);
        applyAngularImpulse(right, angularImpulse);
    }
}

std::string SequentialImpulseConstraintSolver::name() const {
    return "sequential-impulse";
}

void SequentialImpulseConstraintSolver::solve(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const std::vector<std::unique_ptr<Constraint>>& constraints,
    const spacetime::SpacetimeModel& spacetime) const {
    for (int iteration = 0; iteration < std::max(1, iterations_); ++iteration) {
        solveSphereContacts(states, dtSeconds);
        for (const auto& constraint : constraints) {
            constraint->solve(states, dtSeconds, spacetime);
        }
    }
}

void SequentialImpulseConstraintSolver::solveSphereContacts(std::vector<scene::ObjectState>& states, double dtSeconds) const {
    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            auto& left = states[leftIndex];
            auto& right = states[rightIndex];

            const math::Vector3 delta = right.position - left.position;
            const double distance = std::sqrt(delta.normSquared() + common::kEpsilon);
            const double targetDistance = left.radius + right.radius;
            if (distance >= targetDistance) {
                continue;
            }

            const math::Vector3 direction = (distance > common::kEpsilon) ? (delta / distance) : math::Vector3{1.0, 0.0, 0.0};
            const double invMassLeft = inverseMass(left);
            const double invMassRight = inverseMass(right);
            const double totalInvMass = invMassLeft + invMassRight;
            if (totalInvMass <= common::kEpsilon) {
                continue;
            }

            const double penetration = targetDistance - distance;
            const math::Vector3 positionCorrection = direction * ((penetration * positionalCorrection_) / totalInvMass);
            left.position -= positionCorrection * invMassLeft;
            right.position += positionCorrection * invMassRight;

            const auto leftOffset = direction * left.radius;
            const auto rightOffset = -direction * right.radius;
            auto relativeVelocity = velocityAtPoint(right, rightOffset) - velocityAtPoint(left, leftOffset);
            const double relativeNormalSpeed = relativeVelocity.dot(direction);
            if (relativeNormalSpeed >= 0.0) {
                continue;
            }

            const double normalDenominator =
                effectiveMassAlongAxis(left, leftOffset, direction) +
                effectiveMassAlongAxis(right, rightOffset, direction);
            if (normalDenominator <= common::kEpsilon) {
                continue;
            }

            const double restitution = std::min(left.restitution, right.restitution);
            const double bias = (dtSeconds > common::kEpsilon)
                                    ? ((baumgarteFactor_ * penetration) / dtSeconds)
                                    : 0.0;
            double normalImpulseMagnitude = -((1.0 + restitution) * relativeNormalSpeed + bias) / normalDenominator;
            normalImpulseMagnitude = std::max(0.0, normalImpulseMagnitude);
            const auto normalImpulse = direction * normalImpulseMagnitude;
            applyPairImpulse(left, right, leftOffset, rightOffset, normalImpulse);

            relativeVelocity = velocityAtPoint(right, rightOffset) - velocityAtPoint(left, leftOffset);
            const auto tangentialVelocity = relativeVelocity - (direction * relativeVelocity.dot(direction));
            const double tangentialSpeed = tangentialVelocity.norm();
            if (tangentialSpeed <= common::kEpsilon) {
                continue;
            }

            const auto tangent = tangentialVelocity / tangentialSpeed;
            const double tangentDenominator =
                effectiveMassAlongAxis(left, leftOffset, tangent) +
                effectiveMassAlongAxis(right, rightOffset, tangent);
            if (tangentDenominator <= common::kEpsilon) {
                continue;
            }

            const double materialFriction = 0.5 * (left.frictionCoefficient + right.frictionCoefficient);
            const double frictionLimit = 0.5 * (frictionCoefficient_ + materialFriction) * normalImpulseMagnitude;
            const double frictionImpulseMagnitude = std::clamp(-tangentialSpeed / tangentDenominator, -frictionLimit, frictionLimit);
            applyPairImpulse(left, right, leftOffset, rightOffset, tangent * frictionImpulseMagnitude);
        }
    }
}

}  // namespace physicsmade::physics

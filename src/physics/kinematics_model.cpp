#include "physicsmade/physics/kinematics_model.hpp"

#include <algorithm>
#include <cmath>

namespace {

double clampedBetaSquared(const physicsmade::scene::ObjectState& state, const physicsmade::spacetime::SpacetimeModel& spacetime) noexcept {
    const double c = spacetime.speedOfLight();
    if (c <= physicsmade::common::kEpsilon) {
        return 0.0;
    }

    const double betaSquared = state.velocity.normSquared() / (c * c);
    return std::clamp(betaSquared, 0.0, 1.0 - 1.0e-12);
}

}  // namespace

namespace physicsmade::physics {

std::string NewtonianKinematics::name() const {
    return "newtonian";
}

KinematicRegime NewtonianKinematics::regime() const noexcept {
    return KinematicRegime::NonRelativistic;
}

double NewtonianKinematics::lorentzGamma(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    (void)state;
    (void)spacetime;
    return 1.0;
}

double NewtonianKinematics::properTimeStep(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime,
    double coordinateDtSeconds) const noexcept {
    (void)state;
    (void)spacetime;
    return coordinateDtSeconds;
}

math::Vector3 NewtonianKinematics::accelerationFromForce(
    const scene::ObjectState& state,
    const math::Vector3& force,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    (void)spacetime;
    if (state.mass <= common::kEpsilon) {
        return {};
    }

    return force / state.mass;
}

void NewtonianKinematics::projectVelocity(
    scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    (void)state;
    (void)spacetime;
}

math::Vector3 NewtonianKinematics::momentum(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    (void)spacetime;
    return state.velocity * state.mass;
}

double NewtonianKinematics::kineticEnergy(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    (void)spacetime;
    return 0.5 * state.mass * state.velocity.normSquared();
}

std::string SpecialRelativisticKinematics::name() const {
    return "special-relativistic";
}

KinematicRegime SpecialRelativisticKinematics::regime() const noexcept {
    return KinematicRegime::SpecialRelativistic;
}

double SpecialRelativisticKinematics::lorentzGamma(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    return 1.0 / std::sqrt(1.0 - clampedBetaSquared(state, spacetime));
}

double SpecialRelativisticKinematics::properTimeStep(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime,
    double coordinateDtSeconds) const noexcept {
    const double gamma = lorentzGamma(state, spacetime);
    return coordinateDtSeconds / gamma;
}

math::Vector3 SpecialRelativisticKinematics::accelerationFromForce(
    const scene::ObjectState& state,
    const math::Vector3& force,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    if (state.mass <= common::kEpsilon) {
        return {};
    }

    const double c = spacetime.speedOfLight();
    const double gamma = lorentzGamma(state, spacetime);
    const double velocityForceProjection = state.velocity.dot(force);
    const math::Vector3 correctedForce = force - (state.velocity * (velocityForceProjection / (c * c)));
    return correctedForce / (gamma * state.mass);
}

void SpecialRelativisticKinematics::projectVelocity(
    scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    const double c = spacetime.speedOfLight();
    const double speed = state.velocity.norm();
    const double limit = c * (1.0 - 1.0e-10);
    if (speed > limit && speed > common::kEpsilon) {
        state.velocity *= limit / speed;
    }
}

math::Vector3 SpecialRelativisticKinematics::momentum(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    return state.velocity * (lorentzGamma(state, spacetime) * state.mass);
}

double SpecialRelativisticKinematics::kineticEnergy(
    const scene::ObjectState& state,
    const spacetime::SpacetimeModel& spacetime) const noexcept {
    const double gamma = lorentzGamma(state, spacetime);
    const double c = spacetime.speedOfLight();
    return (gamma - 1.0) * state.mass * c * c;
}

const char* toString(KinematicRegime regime) noexcept {
    switch (regime) {
        case KinematicRegime::NonRelativistic:
            return "nonrelativistic";
        case KinematicRegime::SpecialRelativistic:
            return "special-relativistic";
    }

    return "unknown";
}

}  // namespace physicsmade::physics

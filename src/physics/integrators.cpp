#include "physicsmade/physics/physics_system.hpp"

#include <vector>

namespace {

std::vector<physicsmade::math::Vector3> evaluateAccelerations(
    const std::vector<physicsmade::scene::ObjectState>& states,
    const physicsmade::physics::ForceEvaluator& evaluateForces,
    const physicsmade::physics::KinematicsModel& kinematics,
    const physicsmade::spacetime::SpacetimeModel& spacetime) {
    std::vector<physicsmade::math::Vector3> forces(states.size(), {});
    evaluateForces(states, forces);

    std::vector<physicsmade::math::Vector3> accelerations(states.size(), {});
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (states[index].pinned) {
            continue;
        }

        accelerations[index] = kinematics.accelerationFromForce(states[index], forces[index], spacetime);
    }

    return accelerations;
}

void projectAllVelocities(
    std::vector<physicsmade::scene::ObjectState>& states,
    const physicsmade::physics::KinematicsModel& kinematics,
    const physicsmade::spacetime::SpacetimeModel& spacetime) {
    for (auto& state : states) {
        if (!state.pinned) {
            kinematics.projectVelocity(state, spacetime);
        }
    }
}

}  // namespace

namespace physicsmade::physics {

std::string SemiImplicitEulerIntegrator::name() const {
    return "semi-implicit-euler";
}

void SemiImplicitEulerIntegrator::integrate(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const ForceEvaluator& evaluateForces,
    const KinematicsModel& kinematics,
    const spacetime::SpacetimeModel& spacetime) const {
    const auto accelerations = evaluateAccelerations(states, evaluateForces, kinematics, spacetime);

    for (std::size_t index = 0; index < states.size(); ++index) {
        auto& state = states[index];
        if (state.pinned) {
            continue;
        }

        state.velocity += accelerations[index] * dtSeconds;
        kinematics.projectVelocity(state, spacetime);
        state.position += state.velocity * dtSeconds;
    }
}

std::string VelocityVerletIntegrator::name() const {
    return "velocity-verlet";
}

void VelocityVerletIntegrator::integrate(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const ForceEvaluator& evaluateForces,
    const KinematicsModel& kinematics,
    const spacetime::SpacetimeModel& spacetime) const {
    const auto initialAccelerations = evaluateAccelerations(states, evaluateForces, kinematics, spacetime);
    auto predictedStates = states;

    for (std::size_t index = 0; index < predictedStates.size(); ++index) {
        auto& state = predictedStates[index];
        if (state.pinned) {
            continue;
        }

        state.position += (state.velocity * dtSeconds) + (0.5 * initialAccelerations[index] * dtSeconds * dtSeconds);
        state.velocity += initialAccelerations[index] * dtSeconds;
    }

    projectAllVelocities(predictedStates, kinematics, spacetime);
    const auto finalAccelerations = evaluateAccelerations(predictedStates, evaluateForces, kinematics, spacetime);

    for (std::size_t index = 0; index < states.size(); ++index) {
        auto& state = states[index];
        if (state.pinned) {
            continue;
        }

        state.position = predictedStates[index].position;
        state.velocity += 0.5 * (initialAccelerations[index] + finalAccelerations[index]) * dtSeconds;
        kinematics.projectVelocity(state, spacetime);
    }
}

std::string RungeKutta4Integrator::name() const {
    return "runge-kutta-4";
}

void RungeKutta4Integrator::integrate(
    std::vector<scene::ObjectState>& states,
    double dtSeconds,
    const ForceEvaluator& evaluateForces,
    const KinematicsModel& kinematics,
    const spacetime::SpacetimeModel& spacetime) const {
    const std::size_t count = states.size();
    const auto stage1Accelerations = evaluateAccelerations(states, evaluateForces, kinematics, spacetime);

    std::vector<math::Vector3> k1x(count, {});
    std::vector<math::Vector3> k1v(count, {});
    for (std::size_t index = 0; index < count; ++index) {
        if (states[index].pinned) {
            continue;
        }

        k1x[index] = states[index].velocity;
        k1v[index] = stage1Accelerations[index];
    }

    auto stage2States = states;
    for (std::size_t index = 0; index < count; ++index) {
        if (stage2States[index].pinned) {
            continue;
        }

        stage2States[index].position += k1x[index] * (0.5 * dtSeconds);
        stage2States[index].velocity += k1v[index] * (0.5 * dtSeconds);
    }

    projectAllVelocities(stage2States, kinematics, spacetime);
    const auto stage2Accelerations = evaluateAccelerations(stage2States, evaluateForces, kinematics, spacetime);
    std::vector<math::Vector3> k2x(count, {});
    std::vector<math::Vector3> k2v(count, {});
    for (std::size_t index = 0; index < count; ++index) {
        if (stage2States[index].pinned) {
            continue;
        }

        k2x[index] = stage2States[index].velocity;
        k2v[index] = stage2Accelerations[index];
    }

    auto stage3States = states;
    for (std::size_t index = 0; index < count; ++index) {
        if (stage3States[index].pinned) {
            continue;
        }

        stage3States[index].position += k2x[index] * (0.5 * dtSeconds);
        stage3States[index].velocity += k2v[index] * (0.5 * dtSeconds);
    }

    projectAllVelocities(stage3States, kinematics, spacetime);
    const auto stage3Accelerations = evaluateAccelerations(stage3States, evaluateForces, kinematics, spacetime);
    std::vector<math::Vector3> k3x(count, {});
    std::vector<math::Vector3> k3v(count, {});
    for (std::size_t index = 0; index < count; ++index) {
        if (stage3States[index].pinned) {
            continue;
        }

        k3x[index] = stage3States[index].velocity;
        k3v[index] = stage3Accelerations[index];
    }

    auto stage4States = states;
    for (std::size_t index = 0; index < count; ++index) {
        if (stage4States[index].pinned) {
            continue;
        }

        stage4States[index].position += k3x[index] * dtSeconds;
        stage4States[index].velocity += k3v[index] * dtSeconds;
    }

    projectAllVelocities(stage4States, kinematics, spacetime);
    const auto stage4Accelerations = evaluateAccelerations(stage4States, evaluateForces, kinematics, spacetime);
    std::vector<math::Vector3> k4x(count, {});
    std::vector<math::Vector3> k4v(count, {});
    for (std::size_t index = 0; index < count; ++index) {
        if (stage4States[index].pinned) {
            continue;
        }

        k4x[index] = stage4States[index].velocity;
        k4v[index] = stage4Accelerations[index];
    }

    for (std::size_t index = 0; index < count; ++index) {
        auto& state = states[index];
        if (state.pinned) {
            continue;
        }

        state.position += (dtSeconds / 6.0) * (k1x[index] + (2.0 * k2x[index]) + (2.0 * k3x[index]) + k4x[index]);
        state.velocity += (dtSeconds / 6.0) * (k1v[index] + (2.0 * k2v[index]) + (2.0 * k3v[index]) + k4v[index]);
        kinematics.projectVelocity(state, spacetime);
    }
}

}  // namespace physicsmade::physics

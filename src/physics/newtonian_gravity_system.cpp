#include "physicsmade/physics/newtonian_gravity_system.hpp"

#include <cmath>

#include "physicsmade/cuda/gravity_body_forces.hpp"

namespace physicsmade::physics {

std::string NewtonianGravitySystem::name() const {
    return "newtonian-gravity";
}

void NewtonianGravitySystem::accumulateForces(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states,
    std::vector<math::Vector3>& forces) const {
    (void)spacetime;

    if (forces.size() != states.size()) {
        forces.assign(states.size(), {});
    }

    if (preferGpu_) {
        const auto gravityForces = cuda::accumulateGravityForces(states, gravitationalConstant_);
        for (std::size_t index = 0; index < gravityForces.size(); ++index) {
            forces[index] += gravityForces[index];
        }
        return;
    }

    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            const auto& left = states[leftIndex];
            const auto& right = states[rightIndex];
            const math::Vector3 delta = right.position - left.position;
            const double distanceSquared = delta.normSquared() + common::kEpsilon;
            const double distance = std::sqrt(distanceSquared);
            const math::Vector3 direction = delta / distance;
            const double magnitude = gravitationalConstant_ * left.mass * right.mass / distanceSquared;
            const math::Vector3 force = direction * magnitude;

            forces[leftIndex] += force;
            forces[rightIndex] -= force;
        }
    }
}

double NewtonianGravitySystem::potentialEnergy(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states) const {
    (void)spacetime;

    double energy = 0.0;
    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            const auto& left = states[leftIndex];
            const auto& right = states[rightIndex];
            const double distance = std::sqrt((right.position - left.position).normSquared() + common::kEpsilon);
            energy -= gravitationalConstant_ * left.mass * right.mass / distance;
        }
    }

    return energy;
}

}  // namespace physicsmade::physics

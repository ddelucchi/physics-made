#include "physicsmade/physics/electrostatic_system.hpp"

#include <cmath>

namespace physicsmade::physics {

std::string ElectrostaticSystem::name() const {
    return "electrostatic";
}

void ElectrostaticSystem::accumulateForces(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states,
    std::vector<math::Vector3>& forces) const {
    (void)spacetime;

    if (forces.size() != states.size()) {
        forces.assign(states.size(), {});
    }

    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            const auto& left = states[leftIndex];
            const auto& right = states[rightIndex];
            if (std::abs(left.charge) <= common::kEpsilon && std::abs(right.charge) <= common::kEpsilon) {
                continue;
            }

            const math::Vector3 delta = right.position - left.position;
            const double distanceSquared = delta.normSquared() + common::kEpsilon;
            const double distance = std::sqrt(distanceSquared);
            const math::Vector3 direction = delta / distance;
            const double magnitude = coulombConstant_ * left.charge * right.charge / distanceSquared;
            const math::Vector3 force = direction * magnitude;

            forces[leftIndex] -= force;
            forces[rightIndex] += force;
        }
    }
}

double ElectrostaticSystem::potentialEnergy(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states) const {
    (void)spacetime;

    double energy = 0.0;
    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            const auto& left = states[leftIndex];
            const auto& right = states[rightIndex];
            if (std::abs(left.charge) <= common::kEpsilon && std::abs(right.charge) <= common::kEpsilon) {
                continue;
            }

            const double distance = std::sqrt((right.position - left.position).normSquared() + common::kEpsilon);
            energy += coulombConstant_ * left.charge * right.charge / distance;
        }
    }

    return energy;
}

}  // namespace physicsmade::physics

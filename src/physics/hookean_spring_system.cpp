#include "physicsmade/physics/hookean_spring_system.hpp"

#include <cmath>

namespace physicsmade::physics {

std::string HookeanSpringSystem::name() const {
    return "hookean-spring-network";
}

void HookeanSpringSystem::accumulateForces(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states,
    std::vector<math::Vector3>& forces) const {
    (void)spacetime;

    if (forces.size() != states.size()) {
        forces.assign(states.size(), {});
    }

    for (const auto& link : links_) {
        if (link.leftIndex >= states.size() || link.rightIndex >= states.size()) {
            continue;
        }

        const auto& left = states[link.leftIndex];
        const auto& right = states[link.rightIndex];
        const math::Vector3 delta = right.position - left.position;
        const double distance = std::sqrt(delta.normSquared() + common::kEpsilon);
        const math::Vector3 direction = delta / distance;
        const double stretch = distance - link.restLength;
        const double relativeVelocity = (right.velocity - left.velocity).dot(direction);
        const double magnitude = (link.stiffness * stretch) + (link.damping * relativeVelocity);
        const math::Vector3 force = direction * magnitude;

        forces[link.leftIndex] += force;
        forces[link.rightIndex] -= force;
    }
}

double HookeanSpringSystem::potentialEnergy(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states) const {
    (void)spacetime;

    double energy = 0.0;
    for (const auto& link : links_) {
        if (link.leftIndex >= states.size() || link.rightIndex >= states.size()) {
            continue;
        }

        const auto& left = states[link.leftIndex];
        const auto& right = states[link.rightIndex];
        const double distance = std::sqrt((right.position - left.position).normSquared() + common::kEpsilon);
        const double stretch = distance - link.restLength;
        energy += 0.5 * link.stiffness * stretch * stretch;
    }

    return energy;
}

}  // namespace physicsmade::physics

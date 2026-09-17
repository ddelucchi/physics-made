#include "physicsmade/physics/linear_drag_system.hpp"

namespace physicsmade::physics {

std::string LinearDragSystem::name() const {
    return "linear-drag";
}

void LinearDragSystem::accumulateForces(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states,
    std::vector<math::Vector3>& forces) const {
    (void)spacetime;

    if (forces.size() != states.size()) {
        forces.assign(states.size(), {});
    }

    for (std::size_t index = 0; index < states.size(); ++index) {
        const auto& state = states[index];
        const double speed = state.velocity.norm();
        if (speed <= common::kEpsilon) {
            continue;
        }

        const double area = common::kPi * state.radius * state.radius;
        const double magnitude = dragScale_ * fluidDensity_ * state.dragCoefficient * area * speed * speed;
        forces[index] -= state.velocity.normalized() * magnitude;
    }
}

}  // namespace physicsmade::physics

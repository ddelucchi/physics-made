#include "physicsmade/physics/lorentz_force_system.hpp"

#include <cmath>

#include "physicsmade/physics/electromagnetic_tensor.hpp"

namespace physicsmade::physics {

std::string LorentzForceSystem::name() const {
    return "lorentz-force";
}

void LorentzForceSystem::accumulateForces(
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
            const double inverseDistanceCubed = 1.0 / (distanceSquared * distance);

            const math::Vector3 electricAtLeft = -(coulombConstant_ * right.charge) * delta * inverseDistanceCubed;
            const math::Vector3 electricAtRight = (coulombConstant_ * left.charge) * delta * inverseDistanceCubed;

            const double magneticScale = coulombConstant_ / (speedOfLight_ * speedOfLight_);
            const math::Vector3 magneticAtLeft = -(magneticScale * right.charge) * right.velocity.cross(delta) * inverseDistanceCubed;
            const math::Vector3 magneticAtRight = (magneticScale * left.charge) * left.velocity.cross(delta) * inverseDistanceCubed;

            const auto leftTensor = ElectromagneticFieldTensor::fromElectricMagnetic(electricAtLeft, magneticAtLeft, speedOfLight_);
            const auto rightTensor = ElectromagneticFieldTensor::fromElectricMagnetic(electricAtRight, magneticAtRight, speedOfLight_);

            forces[leftIndex] += leftTensor.lorentzForce(left.charge, left.velocity, speedOfLight_);
            forces[rightIndex] += rightTensor.lorentzForce(right.charge, right.velocity, speedOfLight_);
        }
    }
}

double LorentzForceSystem::potentialEnergy(
    const spacetime::SpacetimeModel& spacetime,
    const std::vector<scene::ObjectState>& states) const {
    (void)spacetime;

    double energy = 0.0;
    for (std::size_t leftIndex = 0; leftIndex < states.size(); ++leftIndex) {
        for (std::size_t rightIndex = leftIndex + 1; rightIndex < states.size(); ++rightIndex) {
            const auto& left = states[leftIndex];
            const auto& right = states[rightIndex];
            const double distance = std::sqrt((right.position - left.position).normSquared() + common::kEpsilon);
            energy += coulombConstant_ * left.charge * right.charge / distance;
        }
    }

    return energy;
}

}  // namespace physicsmade::physics

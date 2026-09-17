#pragma once

#include "physicsmade/common/config.hpp"
#include "physicsmade/physics/physics_system.hpp"

namespace physicsmade::physics {

class LorentzForceSystem final : public PhysicsSystem {
  public:
    explicit LorentzForceSystem(
        double coulombConstant = common::kCoulombConstant,
        double speedOfLight = common::kSpeedOfLight)
        : coulombConstant_(coulombConstant), speedOfLight_(speedOfLight) {}

    std::string name() const override;

    void accumulateForces(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states,
        std::vector<math::Vector3>& forces) const override;

    double potentialEnergy(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states) const override;

  private:
    double coulombConstant_{common::kCoulombConstant};
    double speedOfLight_{common::kSpeedOfLight};
};

}  // namespace physicsmade::physics

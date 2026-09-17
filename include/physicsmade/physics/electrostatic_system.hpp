#pragma once

#include "physicsmade/common/config.hpp"
#include "physicsmade/physics/physics_system.hpp"

namespace physicsmade::physics {

class ElectrostaticSystem final : public PhysicsSystem {
  public:
    explicit ElectrostaticSystem(double coulombConstant = common::kCoulombConstant)
        : coulombConstant_(coulombConstant) {}

    std::string name() const override;

    void accumulateForces(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states,
        std::vector<math::Vector3>& forces) const override;

    double potentialEnergy(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states) const override;

    bool conservative() const noexcept override {
        return true;
    }

  private:
    double coulombConstant_{common::kCoulombConstant};
};

}  // namespace physicsmade::physics

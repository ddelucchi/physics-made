#pragma once

#include "physicsmade/common/config.hpp"
#include "physicsmade/physics/physics_system.hpp"

namespace physicsmade::physics {

class NewtonianGravitySystem final : public PhysicsSystem {
  public:
    explicit NewtonianGravitySystem(
        double gravitationalConstant = common::kGravitationalConstant,
        bool preferGpu = true)
        : gravitationalConstant_(gravitationalConstant), preferGpu_(preferGpu) {}

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

    double gravitationalConstant() const noexcept {
        return gravitationalConstant_;
    }

    bool preferGpu() const noexcept {
      return preferGpu_;
    }

  private:
    double gravitationalConstant_{common::kGravitationalConstant};
    bool preferGpu_{true};
};

}  // namespace physicsmade::physics


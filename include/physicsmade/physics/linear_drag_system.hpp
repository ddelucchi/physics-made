#pragma once

#include "physicsmade/common/config.hpp"
#include "physicsmade/physics/physics_system.hpp"

namespace physicsmade::physics {

class LinearDragSystem final : public PhysicsSystem {
  public:
    explicit LinearDragSystem(double fluidDensity = 1.225, double dragScale = 0.5)
        : fluidDensity_(fluidDensity), dragScale_(dragScale) {}

    std::string name() const override;

    void accumulateForces(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states,
        std::vector<math::Vector3>& forces) const override;

  private:
    double fluidDensity_{1.225};
    double dragScale_{0.5};
};

}  // namespace physicsmade::physics

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/physics/physics_system.hpp"

namespace physicsmade::physics {

struct SpringLink {
    std::size_t leftIndex{0};
    std::size_t rightIndex{0};
    double restLength{1.0};
    double stiffness{1.0};
    double damping{0.0};
};

class HookeanSpringSystem final : public PhysicsSystem {
  public:
    explicit HookeanSpringSystem(std::vector<SpringLink> links) : links_(std::move(links)) {}

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
    std::vector<SpringLink> links_;
};

}  // namespace physicsmade::physics

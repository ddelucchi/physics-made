#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/physics/kinematics_model.hpp"
#include "physicsmade/scene/object_state.hpp"
#include "physicsmade/spacetime/spacetime_model.hpp"

namespace physicsmade::physics {

using ForceEvaluator = std::function<void(const std::vector<scene::ObjectState>&, std::vector<math::Vector3>&)>;

class PhysicsSystem {
  public:
    virtual ~PhysicsSystem() = default;

    virtual std::string name() const = 0;
    virtual void accumulateForces(
        const spacetime::SpacetimeModel& spacetime,
        const std::vector<scene::ObjectState>& states,
        std::vector<math::Vector3>& forces) const = 0;

    virtual double potentialEnergy(
      const spacetime::SpacetimeModel& spacetime,
      const std::vector<scene::ObjectState>& states) const {
      (void)spacetime;
      (void)states;
      return 0.0;
    }

    virtual bool conservative() const noexcept {
      return false;
    }
};

class StateIntegrator {
  public:
    virtual ~StateIntegrator() = default;

    virtual std::string name() const = 0;
    virtual void integrate(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
      const ForceEvaluator& evaluateForces,
        const KinematicsModel& kinematics,
        const spacetime::SpacetimeModel& spacetime) const = 0;
};

class SemiImplicitEulerIntegrator final : public StateIntegrator {
  public:
    std::string name() const override;

    void integrate(
      std::vector<scene::ObjectState>& states,
      double dtSeconds,
      const ForceEvaluator& evaluateForces,
      const KinematicsModel& kinematics,
      const spacetime::SpacetimeModel& spacetime) const override;
  };

  class VelocityVerletIntegrator final : public StateIntegrator {
    public:
    std::string name() const override;

    void integrate(
      std::vector<scene::ObjectState>& states,
      double dtSeconds,
      const ForceEvaluator& evaluateForces,
      const KinematicsModel& kinematics,
      const spacetime::SpacetimeModel& spacetime) const override;
  };

  class RungeKutta4Integrator final : public StateIntegrator {
    public:
    std::string name() const override;

    void integrate(
      std::vector<scene::ObjectState>& states,
      double dtSeconds,
      const ForceEvaluator& evaluateForces,
      const KinematicsModel& kinematics,
      const spacetime::SpacetimeModel& spacetime) const override;
};

}  // namespace physicsmade::physics

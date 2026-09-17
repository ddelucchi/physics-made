#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "physicsmade/math/vector3.hpp"
#include "physicsmade/physics/constraints.hpp"
#include "physicsmade/physics/physics_system.hpp"
#include "physicsmade/scene/sim_object.hpp"
#include "physicsmade/spacetime/spacetime_model.hpp"

namespace physicsmade::simulation {

struct ObjectDiagnostics {
    double coordinateTimeSeconds{0.0};
    double properTimeSeconds{0.0};
    double speed{0.0};
    double lorentzGamma{1.0};
    math::Vector3 momentum{};
    math::Vector3 angularMomentum{};
    math::Vector3 angularVelocity{};
    double angularSpeed{0.0};
    double translationalKineticEnergy{0.0};
    double rotationalKineticEnergy{0.0};
    double kineticEnergy{0.0};
};

struct WorldDiagnostics {
  std::size_t objectCount{0};
  double totalMass{0.0};
  double totalCharge{0.0};
  double totalTranslationalKineticEnergy{0.0};
  double totalRotationalKineticEnergy{0.0};
  double totalKineticEnergy{0.0};
  double totalPotentialEnergy{0.0};
  double totalEnergy{0.0};
  double maxSpeed{0.0};
  math::Vector3 totalMomentum{};
  math::Vector3 totalAngularMomentum{};
  math::Vector3 centerOfMass{};
};

class SimulationWorld {
  public:
    explicit SimulationWorld(spacetime::SpacetimeModel spacetime = spacetime::SpacetimeModel{});

    std::size_t addObject(const scene::SimObject& object);
    std::size_t addObject(scene::ObjectState state);

    void addPhysicsSystem(std::unique_ptr<physics::PhysicsSystem> system);
    void setIntegrator(std::unique_ptr<physics::StateIntegrator> integrator);
    void setKinematicsModel(std::unique_ptr<physics::KinematicsModel> kinematics);
    void addConstraint(std::unique_ptr<physics::Constraint> constraint);
    void setConstraintSolver(std::unique_ptr<physics::ConstraintSolver> solver);

    void step(double dtSeconds);

    double simulationTimeSeconds() const noexcept {
        return simulationTimeSeconds_;
    }

    const spacetime::SpacetimeModel& spacetime() const noexcept {
        return spacetime_;
    }

    std::string integratorName() const {
      return integrator_ ? integrator_->name() : "none";
    }

    std::string constraintSolverName() const {
      return constraintSolver_ ? constraintSolver_->name() : "none";
    }

    const physics::KinematicsModel& kinematicsModel() const noexcept {
      return *kinematics_;
    }

    const std::vector<scene::ObjectState>& objects() const noexcept {
        return objects_;
    }

    std::vector<scene::ObjectState>& objects() noexcept {
      return objects_;
    }

    std::optional<std::size_t> findObjectIndex(std::string_view name) const;
    ObjectDiagnostics diagnosticsFor(std::size_t index) const;
    WorldDiagnostics worldDiagnostics() const;

  private:
    spacetime::SpacetimeModel spacetime_;
    std::vector<scene::ObjectState> objects_;
    std::vector<std::unique_ptr<physics::PhysicsSystem>> systems_;
    std::vector<std::unique_ptr<physics::Constraint>> constraints_;
    std::unique_ptr<physics::StateIntegrator> integrator_;
    std::unique_ptr<physics::KinematicsModel> kinematics_;
    std::unique_ptr<physics::ConstraintSolver> constraintSolver_;
    double simulationTimeSeconds_{0.0};
};

}  // namespace physicsmade::simulation


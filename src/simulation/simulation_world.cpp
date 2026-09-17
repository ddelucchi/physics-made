#include "physicsmade/simulation/simulation_world.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "physicsmade/physics/rigid_body_dynamics.hpp"

namespace physicsmade::simulation {

SimulationWorld::SimulationWorld(spacetime::SpacetimeModel spacetime)
        : spacetime_(std::move(spacetime)),
            integrator_(std::make_unique<physics::VelocityVerletIntegrator>()),
    kinematics_(std::make_unique<physics::NewtonianKinematics>()),
    constraintSolver_(std::make_unique<physics::SequentialImpulseConstraintSolver>()) {}

std::size_t SimulationWorld::addObject(const scene::SimObject& object) {
    return addObject(object.initialState());
}

std::size_t SimulationWorld::addObject(scene::ObjectState state) {
    objects_.push_back(std::move(state));
    return objects_.size() - 1;
}

void SimulationWorld::addPhysicsSystem(std::unique_ptr<physics::PhysicsSystem> system) {
    if (system) {
        systems_.push_back(std::move(system));
    }
}

void SimulationWorld::setIntegrator(std::unique_ptr<physics::StateIntegrator> integrator) {
    if (integrator) {
        integrator_ = std::move(integrator);
    }
}

void SimulationWorld::setKinematicsModel(std::unique_ptr<physics::KinematicsModel> kinematics) {
    if (kinematics) {
        kinematics_ = std::move(kinematics);
    }
}

void SimulationWorld::addConstraint(std::unique_ptr<physics::Constraint> constraint) {
    if (constraint) {
        constraints_.push_back(std::move(constraint));
    }
}

void SimulationWorld::setConstraintSolver(std::unique_ptr<physics::ConstraintSolver> solver) {
    if (solver) {
        constraintSolver_ = std::move(solver);
    }
}

void SimulationWorld::step(double dtSeconds) {
    if (dtSeconds <= 0.0) {
        return;
    }

    const physics::ForceEvaluator evaluator = [this](
                                                const std::vector<scene::ObjectState>& states,
                                                std::vector<math::Vector3>& forces) {
        forces.assign(states.size(), {});
        for (const auto& system : systems_) {
            system->accumulateForces(spacetime_, states, forces);
        }
    };

    if (integrator_) {
        integrator_->integrate(objects_, dtSeconds, evaluator, *kinematics_, spacetime_);
    }

    if (constraintSolver_) {
        constraintSolver_->solve(objects_, dtSeconds, constraints_, spacetime_);
    }

    for (auto& object : objects_) {
        physics::integrateOrientation(object, dtSeconds);
        object.coordinateTimeSeconds += dtSeconds;
        object.properTimeSeconds += kinematics_->properTimeStep(object, spacetime_, dtSeconds);
    }

    simulationTimeSeconds_ += dtSeconds;
}

std::optional<std::size_t> SimulationWorld::findObjectIndex(std::string_view name) const {
    for (std::size_t index = 0; index < objects_.size(); ++index) {
        if (objects_[index].name == name) {
            return index;
        }
    }

    return std::nullopt;
}

ObjectDiagnostics SimulationWorld::diagnosticsFor(std::size_t index) const {
    if (index >= objects_.size()) {
        throw std::out_of_range("object index out of range");
    }

    const auto& object = objects_[index];
    return {
        object.coordinateTimeSeconds,
        object.properTimeSeconds,
        object.velocity.norm(),
        kinematics_->lorentzGamma(object, spacetime_),
        kinematics_->momentum(object, spacetime_),
        physics::angularMomentum(object),
        object.angularVelocity,
        object.angularVelocity.norm(),
        kinematics_->kineticEnergy(object, spacetime_),
        physics::rotationalKineticEnergy(object),
        kinematics_->kineticEnergy(object, spacetime_) + physics::rotationalKineticEnergy(object),
    };
}

WorldDiagnostics SimulationWorld::worldDiagnostics() const {
    WorldDiagnostics diagnostics{};
    diagnostics.objectCount = objects_.size();

    double massWeightedCount = 0.0;
    for (std::size_t index = 0; index < objects_.size(); ++index) {
        const auto& object = objects_[index];
        const auto objectDiagnostics = diagnosticsFor(index);
        diagnostics.totalMass += object.mass;
        diagnostics.totalCharge += object.charge;
        diagnostics.totalTranslationalKineticEnergy += objectDiagnostics.translationalKineticEnergy;
        diagnostics.totalRotationalKineticEnergy += objectDiagnostics.rotationalKineticEnergy;
        diagnostics.totalKineticEnergy += objectDiagnostics.kineticEnergy;
        diagnostics.totalMomentum += objectDiagnostics.momentum;
        diagnostics.totalAngularMomentum += objectDiagnostics.angularMomentum;
        diagnostics.maxSpeed = std::max(diagnostics.maxSpeed, objectDiagnostics.speed);

        if (object.mass > common::kEpsilon) {
            diagnostics.centerOfMass += object.position * object.mass;
            massWeightedCount += object.mass;
        }
    }

    for (const auto& system : systems_) {
        diagnostics.totalPotentialEnergy += system->potentialEnergy(spacetime_, objects_);
    }

    if (massWeightedCount > common::kEpsilon) {
        diagnostics.centerOfMass = diagnostics.centerOfMass / massWeightedCount;
    }

    diagnostics.totalEnergy = diagnostics.totalKineticEnergy + diagnostics.totalPotentialEnergy;
    return diagnostics;
}

}  // namespace physicsmade::simulation


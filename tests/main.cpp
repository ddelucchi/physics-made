#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "physicsmade/camera/orbit_camera.hpp"
#include "physicsmade/common/config.hpp"
#include "physicsmade/cuda/gravity_body_forces.hpp"
#include "physicsmade/cuda/field_grid.hpp"
#include "physicsmade/cuda/render_instances.hpp"
#include "physicsmade/cuda/scene_buffers.hpp"
#include "physicsmade/field_theory/coupled_su2_fermion_lattice.hpp"
#include "physicsmade/field_theory/rectangular_scalar_field_lattice.hpp"
#include "physicsmade/field_theory/scalar_field_lattice.hpp"
#include "physicsmade/field_theory/staggered_fermion_lattice.hpp"
#include "physicsmade/field_theory/su2_gauge_field_lattice.hpp"
#include "physicsmade/field_theory/u1_gauge_field_lattice.hpp"
#include "physicsmade/io/scene_serialization.hpp"
#include "physicsmade/math/four_vector.hpp"
#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/physics/constraints.hpp"
#include "physicsmade/physics/electrostatic_system.hpp"
#include "physicsmade/physics/hookean_spring_system.hpp"
#include "physicsmade/physics/kinematics_model.hpp"
#include "physicsmade/physics/linear_drag_system.hpp"
#include "physicsmade/physics/lorentz_force_system.hpp"
#include "physicsmade/physics/newtonian_gravity_system.hpp"
#include "physicsmade/physics/rigid_body_dynamics.hpp"
#include "physicsmade/plugins/plugin_discovery.hpp"
#include "physicsmade/quantum_chemistry/restricted_hartree_fock.hpp"
#include "physicsmade/runtime/generated_scenes.hpp"
#include "physicsmade/runtime/programs.hpp"
#include "physicsmade/spacetime/geodesic_integrator.hpp"
#include "physicsmade/spacetime/kerr_metric.hpp"
#include "physicsmade/spacetime/schwarzschild_metric.hpp"
#include "physicsmade/spacetime/weak_field_metric.hpp"
#include "physicsmade/simulation/simulation_world.hpp"

namespace {

bool nearlyEqual(double left, double right, double tolerance) {
    return std::abs(left - right) <= tolerance;
}

bool relativeNearlyEqual(double left, double right, double tolerance) {
    const double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= (tolerance * scale);
}

double signedAngleAroundAxis(
    const physicsmade::math::Vector3& left,
    const physicsmade::math::Vector3& right,
    const physicsmade::math::Vector3& axis) {
    const auto leftPlane = left - (axis * left.dot(axis));
    const auto rightPlane = right - (axis * right.dot(axis));
    const double leftNorm = leftPlane.norm();
    const double rightNorm = rightPlane.norm();
    if (leftNorm <= physicsmade::common::kEpsilon || rightNorm <= physicsmade::common::kEpsilon) {
        return 0.0;
    }

    const auto leftUnit = leftPlane / leftNorm;
    const auto rightUnit = rightPlane / rightNorm;
    return std::atan2(axis.dot(leftUnit.cross(rightUnit)), std::clamp(leftUnit.dot(rightUnit), -1.0, 1.0));
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path resolvePluginsRoot() {
    auto cursor = std::filesystem::current_path();
    while (true) {
        const auto candidate = cursor / "plugins";
        if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
            return candidate;
        }

        if (cursor == cursor.root_path()) {
            throw std::runtime_error("plugins directory not found from current working directory");
        }

        cursor = cursor.parent_path();
    }
}

void testVectorMath() {
    const physicsmade::math::Vector3 left{1.0, 2.0, 3.0};
    const physicsmade::math::Vector3 right{-2.0, 5.0, 1.0};
    const auto combined = left + right;

    require(nearlyEqual(combined.x, -1.0, 1.0e-12), "vector addition x failed");
    require(nearlyEqual(combined.y, 7.0, 1.0e-12), "vector addition y failed");
    require(nearlyEqual(combined.z, 4.0, 1.0e-12), "vector addition z failed");
    require(nearlyEqual(left.cross(right).z, 9.0, 1.0e-12), "vector cross product failed");
}

void testQuaternionRotation() {
    const auto quarterTurn = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, 0.5 * physicsmade::common::kPi);
    const auto rotated = quarterTurn.rotate({1.0, 0.0, 0.0});
    require(std::abs(rotated.x) <= 1.0e-12, "quaternion rotation x incorrect");
    require(relativeNearlyEqual(rotated.y, 1.0, 1.0e-12), "quaternion rotation y incorrect");
}

void testLightlikeInterval() {
    physicsmade::spacetime::SpacetimeModel spacetime;
    const physicsmade::math::FourVector eventA{0.0, 0.0, 0.0, 0.0};
    const physicsmade::math::FourVector eventB{5.0, 3.0, 4.0, 0.0};

    require(nearlyEqual(spacetime.intervalSquared(eventA, eventB), 0.0, 1.0e-12), "lightlike interval should be zero");
}

void testGravityForces() {
    physicsmade::physics::NewtonianGravitySystem gravity(10.0);

    const std::vector<physicsmade::scene::ObjectState> states{
        {"a", 2.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"b", 3.0, 0.0, 1.0, false, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    std::vector<physicsmade::math::Vector3> forces(states.size(), {});
    gravity.accumulateForces(physicsmade::spacetime::SpacetimeModel{}, states, forces);

    require(nearlyEqual(forces[0].x, 15.0, 1.0e-9), "gravity force on left body incorrect");
    require(nearlyEqual(forces[1].x, -15.0, 1.0e-9), "gravity force on right body incorrect");
}

void testOrbitCamera() {
    physicsmade::camera::OrbitCamera camera(10.0, 0.0, 0.0);
    const auto pose = camera.pose({0.0, 0.0, 0.0});

    require(nearlyEqual(pose.position.x, 10.0, 1.0e-12), "camera radius placement incorrect");
    require(nearlyEqual(pose.forward.x, -1.0, 1.0e-12), "camera forward vector incorrect");
}

void testFieldSampling() {
    const std::vector<physicsmade::scene::ObjectState> sources{
        {"source", 10.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    const physicsmade::cuda::FieldGridSpec spec{1, 1, 1, {2.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
    const auto grid = physicsmade::cuda::evaluateGravityField(sources, spec, 10.0);
    const auto& sample = grid.samples.front();

    require(nearlyEqual(sample.acceleration.x, -25.0, 1.0e-8), "field acceleration incorrect");
    require(nearlyEqual(sample.potential, -50.0, 1.0e-8), "field potential incorrect");
}

void testGpuStyleGravityBodyForces() {
    const std::vector<physicsmade::scene::ObjectState> states{
        {"a", 2.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"b", 3.0, 0.0, 1.0, false, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    const auto forces = physicsmade::cuda::accumulateGravityForces(states, 10.0);
    require(nearlyEqual(forces[0].x, 15.0, 1.0e-9), "body force solver left force incorrect");
    require(nearlyEqual(forces[1].x, -15.0, 1.0e-9), "body force solver right force incorrect");
}

void testElectrostaticForces() {
    physicsmade::physics::ElectrostaticSystem electrostatics(10.0);
    const std::vector<physicsmade::scene::ObjectState> states{
        {"q1", 1.0, 1.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"q2", 1.0, 1.0, 1.0, false, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    std::vector<physicsmade::math::Vector3> forces(states.size(), {});
    electrostatics.accumulateForces(physicsmade::spacetime::SpacetimeModel{}, states, forces);

    require(nearlyEqual(forces[0].x, -2.5, 1.0e-9), "electrostatic left force incorrect");
    require(nearlyEqual(forces[1].x, 2.5, 1.0e-9), "electrostatic right force incorrect");
    require(nearlyEqual(electrostatics.potentialEnergy(physicsmade::spacetime::SpacetimeModel{}, states), 5.0, 1.0e-9), "electrostatic potential incorrect");
}

void testLorentzForces() {
    physicsmade::physics::LorentzForceSystem lorentz(10.0, 100.0);
    const std::vector<physicsmade::scene::ObjectState> states{
        {"q1", 1.0, 1.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"q2", 1.0, 1.0, 1.0, false, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    std::vector<physicsmade::math::Vector3> forces(states.size(), {});
    lorentz.accumulateForces(physicsmade::spacetime::SpacetimeModel{}, states, forces);

    require(nearlyEqual(forces[0].x, -2.5, 1.0e-9), "lorentz left force incorrect at zero velocity");
    require(nearlyEqual(forces[1].x, 2.5, 1.0e-9), "lorentz right force incorrect at zero velocity");
    require(nearlyEqual(lorentz.potentialEnergy(physicsmade::spacetime::SpacetimeModel{}, states), 5.0, 1.0e-9), "lorentz potential incorrect");
}

void testSpringForces() {
    physicsmade::physics::HookeanSpringSystem springs({{0, 1, 1.0, 2.0, 0.0}});
    const std::vector<physicsmade::scene::ObjectState> states{
        {"anchor", 1.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"bob", 1.0, 0.0, 1.0, false, {3.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    std::vector<physicsmade::math::Vector3> forces(states.size(), {});
    springs.accumulateForces(physicsmade::spacetime::SpacetimeModel{}, states, forces);

    require(nearlyEqual(forces[0].x, 4.0, 1.0e-9), "spring force on anchor incorrect");
    require(nearlyEqual(forces[1].x, -4.0, 1.0e-9), "spring force on bob incorrect");
}

void testDistanceConstraint() {
    physicsmade::physics::DistanceConstraint constraint({0, 1, 2.0, 1.0});
    std::vector<physicsmade::scene::ObjectState> states{
        {"left", 1.0, 0.0, 0.3, false, {-2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"right", 1.0, 0.0, 0.3, false, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };

    constraint.solve(states, 0.5, physicsmade::spacetime::SpacetimeModel{});
    const double distance = (states[1].position - states[0].position).norm();
    require(nearlyEqual(distance, 2.0, 1.0e-9), "distance constraint should correct separation");
}

void testBallJointConstraint() {
    physicsmade::physics::BallJointConstraint constraint({0, 1, {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, 0.9, 0.1});
    std::vector<physicsmade::scene::ObjectState> states{
        {"anchor", 1000.0, 0.0, 0.3, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"bob", 2.0, 0.0, 0.3, false, {0.6, -2.2, 0.0}, {0.8, 0.0, 0.0}},
    };

    const double initialError = (physicsmade::physics::worldPoint(states[1], {0.0, 1.0, 0.0}) - physicsmade::physics::worldPoint(states[0], {0.0, -1.0, 0.0})).norm();
    constraint.solve(states, 0.01, physicsmade::spacetime::SpacetimeModel{});
    const double finalError = (physicsmade::physics::worldPoint(states[1], {0.0, 1.0, 0.0}) - physicsmade::physics::worldPoint(states[0], {0.0, -1.0, 0.0})).norm();
    require(finalError < initialError, "ball joint should reduce anchor error");
}

void testHingeConstraint() {
    physicsmade::physics::HingeConstraintSpec spec{
        {0, 1, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}, 0.9, 0.1},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    spec.angularStiffness = 0.9;
    physicsmade::physics::HingeConstraint constraint(spec);
    std::vector<physicsmade::scene::ObjectState> states{
        {"stator", 1000.0, 0.0, 0.3, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"rotor", 2.0, 0.0, 0.3, false, {0.0, 2.0, 0.0}, {0.0, 0.0, 0.0}},
    };
    states[1].orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, 0.35);

    const auto initialLeftAxis = physicsmade::physics::worldDirection(states[0], {0.0, 1.0, 0.0});
    const auto initialRightAxis = physicsmade::physics::worldDirection(states[1], {0.0, 1.0, 0.0});
    const double initialMisalignment = initialLeftAxis.cross(initialRightAxis).norm();
    constraint.solve(states, 0.01, physicsmade::spacetime::SpacetimeModel{});
    const auto finalLeftAxis = physicsmade::physics::worldDirection(states[0], {0.0, 1.0, 0.0});
    const auto finalRightAxis = physicsmade::physics::worldDirection(states[1], {0.0, 1.0, 0.0});
    const double finalMisalignment = finalLeftAxis.cross(finalRightAxis).norm();
    require(finalMisalignment < initialMisalignment, "hinge should reduce axis misalignment");
}

void testHingeLimitConstraint() {
    physicsmade::physics::HingeConstraintSpec spec{
        {0, 1, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}, 0.95, 0.1},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    spec.leftLocalReference = {1.0, 0.0, 0.0};
    spec.rightLocalReference = {1.0, 0.0, 0.0};
    spec.angularStiffness = 0.95;
    spec.limitsEnabled = true;
    spec.lowerAngleLimit = -0.2;
    spec.upperAngleLimit = 0.2;

    physicsmade::physics::HingeConstraint constraint(spec);
    std::vector<physicsmade::scene::ObjectState> states{
        {"stator", 1000.0, 0.0, 0.3, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"link", 2.0, 0.0, 0.3, false, {0.0, 2.0, 0.0}, {0.0, 0.0, 0.0}},
    };
    states[1].orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 1.0, 0.0}, 0.7);

    const auto initialAngle = signedAngleAroundAxis(
        physicsmade::physics::worldDirection(states[0], spec.leftLocalReference),
        physicsmade::physics::worldDirection(states[1], spec.rightLocalReference),
        {0.0, 1.0, 0.0});
    constraint.solve(states, 0.01, physicsmade::spacetime::SpacetimeModel{});
    const auto finalAngle = signedAngleAroundAxis(
        physicsmade::physics::worldDirection(states[0], spec.leftLocalReference),
        physicsmade::physics::worldDirection(states[1], spec.rightLocalReference),
        {0.0, 1.0, 0.0});
    require(std::abs(finalAngle - spec.upperAngleLimit) < std::abs(initialAngle - spec.upperAngleLimit), "hinge limit should pull the angle toward its upper stop");
}

void testHingeMotorConstraint() {
    physicsmade::physics::HingeConstraintSpec spec{
        {0, 1, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}, 0.9, 0.1},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    spec.leftLocalReference = {1.0, 0.0, 0.0};
    spec.rightLocalReference = {1.0, 0.0, 0.0};
    spec.limitsEnabled = true;
    spec.lowerAngleLimit = -0.75;
    spec.upperAngleLimit = 0.75;
    spec.motorEnabled = true;
    spec.targetAngularSpeed = 6.0;
    spec.maxMotorTorque = 120.0;

    physicsmade::physics::HingeConstraint constraint(spec);
    std::vector<physicsmade::scene::ObjectState> states{
        {"stator", 1000.0, 0.0, 0.3, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
        {"rotor", 3.0, 0.0, 0.3, false, {0.0, 2.0, 0.0}, {0.0, 0.0, 0.0}},
    };
    states[1].bodyInertiaDiagonal = {0.6, 0.2, 0.6};

    constraint.solve(states, 0.02, physicsmade::spacetime::SpacetimeModel{});
    require(states[1].angularVelocity.y > 0.0, "hinge motor should drive positive angular velocity about the hinge axis");
}

void testContactSolver() {
    physicsmade::physics::SequentialImpulseConstraintSolver solver(0.85, 10, 0.9, 0.25);
    std::vector<physicsmade::scene::ObjectState> states{
        {"left", 1.0, 0.0, 0.5, false, {-0.3, 0.2, 0.0}, {1.0, 0.4, 0.0}},
        {"right", 1.0, 0.0, 0.5, false, {0.3, -0.2, 0.0}, {-1.0, -0.1, 0.0}},
    };
    states[0].restitution = 0.9;
    states[1].restitution = 0.9;
    states[0].frictionCoefficient = 1.0;
    states[1].frictionCoefficient = 1.0;

    std::vector<std::unique_ptr<physicsmade::physics::Constraint>> constraints;
    solver.solve(states, 0.01, constraints, physicsmade::spacetime::SpacetimeModel{});

    require((states[1].position - states[0].position).norm() >= 0.85, "contact solver should separate overlapping spheres");
    require(std::abs(states[0].angularVelocity.z) + std::abs(states[1].angularVelocity.z) > 1.0e-6, "glancing contact should generate spin");
}

void testWorldStepping() {
    physicsmade::simulation::SimulationWorld world;
    world.addPhysicsSystem(std::make_unique<physicsmade::physics::NewtonianGravitySystem>(1.0));
    world.addObject({"anchor", 1.0e6, 0.0, 1.0, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});
    const auto travelerIndex = world.addObject({"traveler", 1.0, 0.0, 1.0, false, {10.0, 0.0, 0.0}, {0.0, 0.0, 0.0}});

    world.step(1.0);
    const auto& traveler = world.objects()[travelerIndex];

    require(traveler.position.x < 10.0, "traveler should move toward anchor");
}

void testRigidBodyRotation() {
    physicsmade::simulation::SimulationWorld world;
    physicsmade::scene::ObjectState spinner{"spinner", 5.0, 0.0, 0.5, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    spinner.angularVelocity = {0.0, 0.0, physicsmade::common::kPi};
    spinner.bodyInertiaDiagonal = {1.0, 1.0, 0.5};

    const auto index = world.addObject(spinner);
    world.step(0.5);

    const auto& updated = world.objects()[index];
    const auto rotated = updated.orientation.rotate({1.0, 0.0, 0.0});
    const auto diagnostics = world.diagnosticsFor(index);
    require(std::abs(rotated.x) < 1.0e-6, "rigid-body rotation should rotate x away from the original axis");
    require(rotated.y > 0.99, "rigid-body rotation should rotate x toward y");
    require(diagnostics.angularSpeed > 3.0, "rigid-body angular speed should be tracked");
    require(diagnostics.rotationalKineticEnergy > 0.0, "rigid-body rotational energy should be positive");
}

void testSpecialRelativisticKinematics() {
    const double c = physicsmade::common::kSpeedOfLight;
    const physicsmade::scene::ObjectState probe{"probe", 2.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.8 * c, 0.0, 0.0}};
    const physicsmade::physics::SpecialRelativisticKinematics kinematics;
    const physicsmade::spacetime::SpacetimeModel spacetime;

    require(relativeNearlyEqual(kinematics.lorentzGamma(probe, spacetime), 5.0 / 3.0, 1.0e-12), "lorentz gamma incorrect");
    require(relativeNearlyEqual(kinematics.properTimeStep(probe, spacetime, 1.0), 0.6, 1.0e-12), "proper time step incorrect");

    auto clippedProbe = probe;
    clippedProbe.velocity = {1.2 * c, 0.0, 0.0};
    kinematics.projectVelocity(clippedProbe, spacetime);
    require(clippedProbe.velocity.norm() < c, "relativistic velocity projection should keep speed below c");
}

void testRelativisticWorldTiming() {
    const double c = physicsmade::common::kSpeedOfLight;
    physicsmade::simulation::SimulationWorld world;
    world.setKinematicsModel(std::make_unique<physicsmade::physics::SpecialRelativisticKinematics>());
    const auto probeIndex = world.addObject({"probe", 10.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.8 * c, 0.0, 0.0}});

    world.step(1.0);

    const auto diagnostics = world.diagnosticsFor(probeIndex);
    require(relativeNearlyEqual(diagnostics.coordinateTimeSeconds, 1.0, 1.0e-12), "coordinate time tracking incorrect");
    require(relativeNearlyEqual(diagnostics.properTimeSeconds, 0.6, 1.0e-12), "proper time tracking incorrect");
    require(relativeNearlyEqual(diagnostics.lorentzGamma, 5.0 / 3.0, 1.0e-12), "world gamma incorrect");
}

void testWorldDiagnostics() {
    physicsmade::simulation::SimulationWorld world;
    world.setIntegrator(std::make_unique<physicsmade::physics::RungeKutta4Integrator>());
    world.addPhysicsSystem(std::make_unique<physicsmade::physics::NewtonianGravitySystem>(1.0, false));
    world.addObject({"left", 2.0, 1.0, 1.0, false, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
    world.addObject({"right", 3.0, -1.0, 1.0, false, {2.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}});

    const auto diagnostics = world.worldDiagnostics();
    require(nearlyEqual(diagnostics.totalMass, 5.0, 1.0e-12), "total mass incorrect");
    require(nearlyEqual(diagnostics.totalCharge, 0.0, 1.0e-12), "total charge incorrect");
    require(nearlyEqual(diagnostics.totalTranslationalKineticEnergy, 2.5, 1.0e-12), "total translational kinetic energy incorrect");
    require(nearlyEqual(diagnostics.totalRotationalKineticEnergy, 0.0, 1.0e-12), "total rotational kinetic energy incorrect");
    require(nearlyEqual(diagnostics.totalKineticEnergy, 2.5, 1.0e-12), "total kinetic energy incorrect");
    require(nearlyEqual(diagnostics.totalPotentialEnergy, -3.0, 1.0e-9), "total potential energy incorrect");
    require(nearlyEqual(diagnostics.totalEnergy, -0.5, 1.0e-9), "total energy incorrect");
}

void testWeakFieldMetric() {
    const physicsmade::spacetime::WeakFieldSphericalMassMetric metric(1.98847e30);
    const auto tensor = metric.covariant({0.0, 1.495978707e11, 0.0, 0.0});
    require(std::string(metric.name()) == "weak-field-spherical-mass", "weak-field metric name incorrect");
    require(tensor[0][0] > -1.0, "weak-field temporal component should be weakly shifted from Minkowski");
}

void testSchwarzschildMetric() {
    const double solarMass = 1.98847e30;
    const physicsmade::spacetime::SchwarzschildMetric schwarzschildMetric(solarMass);
    const physicsmade::spacetime::WeakFieldSphericalMassMetric weakFieldMetric(solarMass);
    const auto schwarzschildTensor = schwarzschildMetric.covariant({0.0, 1.495978707e11, 0.0, 0.0});
    const auto weakFieldTensor = weakFieldMetric.covariant({0.0, 1.495978707e11, 0.0, 0.0});

    require(std::string(schwarzschildMetric.name()) == "schwarzschild-isotropic", "Schwarzschild metric name incorrect");
    require(schwarzschildMetric.schwarzschildRadius() > 0.0, "Schwarzschild radius should be positive");
    require(schwarzschildMetric.isotropicHorizonRadius() > 0.0, "isotropic horizon radius should be positive");
    require(schwarzschildTensor[0][0] < 0.0, "Schwarzschild temporal component should remain timelike");
    require(relativeNearlyEqual(schwarzschildTensor[0][0], weakFieldTensor[0][0], 1.0e-10), "Schwarzschild far-field temporal component should match weak field");
    require(relativeNearlyEqual(schwarzschildTensor[1][1], weakFieldTensor[1][1], 1.0e-10), "Schwarzschild far-field spatial component should match weak field");
}

void testKerrMetric() {
    const double solarMass = 1.98847e30;
    const physicsmade::spacetime::KerrMetric rotatingMetric(4.154e6 * solarMass, 0.78);
    const physicsmade::spacetime::KerrMetric nonRotatingMetric(4.154e6 * solarMass, 0.0);
    const auto rotatingTensor = rotatingMetric.covariant({0.0, 12.0 * rotatingMetric.gravitationalRadius(), 0.0, 0.0});
    const auto nonRotatingTensor = nonRotatingMetric.covariant({0.0, 12.0 * nonRotatingMetric.gravitationalRadius(), 0.0, 0.0});

    require(std::string(rotatingMetric.name()) == "kerr-schild-cartesian", "Kerr metric name incorrect");
    require(rotatingMetric.gravitationalRadius() > 0.0, "Kerr gravitational radius should be positive");
    require(rotatingMetric.eventHorizonRadius() > rotatingMetric.gravitationalRadius(), "Kerr event horizon should exceed the gravitational radius for sub-extremal spin");
    require(rotatingTensor[0][0] < 0.0, "Kerr temporal component should remain timelike outside the horizon");
    require(std::abs(rotatingTensor[0][2]) > 1.0e-6, "Kerr frame-dragging cross term should be nonzero away from the spin axis");
    require(std::abs(nonRotatingTensor[0][2]) <= 1.0e-12, "Zero-spin Kerr limit should remove the spin-induced cross term");
}

void testNumericGeodesicIntegrator() {
    const physicsmade::spacetime::SpacetimeModel spacetime;
    const auto initialState = physicsmade::spacetime::makeTimelikeGeodesicState(
        spacetime,
        {0.0, 0.0, 0.0, 0.0},
        {0.1 * physicsmade::common::kSpeedOfLight, 0.0, 0.0});
    const physicsmade::spacetime::NumericGeodesicIntegrator integrator(spacetime, {0.5, 1.0, 1.0e-6});
    const auto trajectory = integrator.integrate(initialState, 10);
    const auto& finalState = trajectory.back();
    const double affineTime = 10.0 * integrator.options().affineStep;
    const double expectedX = initialState.position.x + (initialState.tangent.x * affineTime);

    require(trajectory.size() == 11, "geodesic trajectory should include the initial sample");
    require(relativeNearlyEqual(finalState.position.x, expectedX, 1.0e-9), "flat-spacetime geodesic should remain linear");
    require(relativeNearlyEqual(finalState.tangent.x, initialState.tangent.x, 1.0e-9), "flat-spacetime geodesic tangent should remain constant");
    require(relativeNearlyEqual(
        physicsmade::spacetime::tangentNormSquared(spacetime, finalState),
        -physicsmade::common::kSpeedOfLight * physicsmade::common::kSpeedOfLight,
        1.0e-9),
        "flat-spacetime geodesic should preserve the timelike norm");
}

void testSchwarzschildGeodesicIntegrator() {
    const double solarMass = 1.98847e30;
    const double orbitalRadius = 1.495978707e11;
    const double orbitalSpeed = std::sqrt(physicsmade::common::kGravitationalConstant * solarMass / orbitalRadius);
    const auto metric = std::make_shared<physicsmade::spacetime::SchwarzschildMetric>(solarMass);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const auto initialState = physicsmade::spacetime::makeTimelikeGeodesicState(
        spacetime,
        {0.0, orbitalRadius, 0.0, 0.0},
        {0.0, orbitalSpeed, 0.0});
    const physicsmade::spacetime::NumericGeodesicIntegrator integrator(spacetime, {1800.0, 1.0, 1.0e-6});
    const auto trajectory = integrator.integrate(initialState, 48);
    const auto& finalState = trajectory.back();
    const double radiusDrift = std::abs(finalState.position.spatial().norm() - initialState.position.spatial().norm());
    const double normalizedDrift =
        std::abs(physicsmade::spacetime::tangentNormSquared(spacetime, finalState) +
                 (physicsmade::common::kSpeedOfLight * physicsmade::common::kSpeedOfLight)) /
        (physicsmade::common::kSpeedOfLight * physicsmade::common::kSpeedOfLight);

    require(radiusDrift < 2.0e9, "far-field Schwarzschild geodesic should remain close to a circular orbit over one day");
    require(physicsmade::spacetime::coordinateSpeed(finalState) < physicsmade::common::kSpeedOfLight, "Schwarzschild geodesic coordinate speed should remain subluminal");
    require(normalizedDrift < 1.0e-4, "Schwarzschild geodesic integration should preserve the timelike norm approximately");
}

void testKerrGeodesicIntegrator() {
    const double solarMass = 1.98847e30;
    const double centralMass = 4.154e6 * solarMass;
    const auto metric = std::make_shared<physicsmade::spacetime::KerrMetric>(centralMass, 0.78);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const double orbitalRadius = 85.0 * metric->gravitationalRadius();
    const double orbitalSpeed = std::sqrt(physicsmade::common::kGravitationalConstant * centralMass / orbitalRadius);
    const auto initialState = physicsmade::spacetime::makeTimelikeGeodesicState(
        spacetime,
        {0.0, orbitalRadius, 0.0, 0.0},
        {0.0, orbitalSpeed, 0.0});
    const physicsmade::spacetime::NumericGeodesicIntegrator integrator(spacetime, {600.0, 1.0, 1.0e-6});
    const auto trajectory = integrator.integrate(initialState, 180);
    const auto& finalState = trajectory.back();

    double minimumRadius = initialState.position.spatial().norm();
    for (const auto& state : trajectory) {
        minimumRadius = std::min(minimumRadius, state.position.spatial().norm());
    }

    const double normalizedDrift =
        std::abs(physicsmade::spacetime::tangentNormSquared(spacetime, finalState) +
                 (physicsmade::common::kSpeedOfLight * physicsmade::common::kSpeedOfLight)) /
        (physicsmade::common::kSpeedOfLight * physicsmade::common::kSpeedOfLight);

    require(minimumRadius > 10.0 * metric->eventHorizonRadius(), "Kerr geodesic should remain well outside the horizon in the preview orbit");
    require(physicsmade::spacetime::coordinateSpeed(finalState) < physicsmade::common::kSpeedOfLight, "Kerr geodesic coordinate speed should remain subluminal");
    require(std::atan2(finalState.position.y, finalState.position.x) > 0.1, "Kerr geodesic should advance azimuthally along the orbit");
    require(normalizedDrift < 5.0e-4, "Kerr geodesic integration should preserve the timelike norm approximately");
}

void testRestrictedHartreeFock() {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule(1.4);
    const auto result = physicsmade::quantum_chemistry::solveRestrictedHartreeFock(molecule);

    require(result.converged, "restricted Hartree-Fock should converge for minimal-basis H2");
    require(result.basisCount == 2, "minimal H2 basis should contain two orbitals");
    require(result.occupiedOrbitals == 1, "minimal H2 should have one occupied orbital pair");
    require(result.totalEnergy < -1.0, "minimal-basis H2 energy should be bound");
    require(result.totalEnergy > -1.2, "minimal-basis H2 energy should stay in a plausible STO-3G range");
    require(!result.orbitalEnergies.empty() && result.orbitalEnergies.front() < 0.0, "occupied orbital energy should be negative");
}

void testUnrestrictedHartreeFock() {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecularCation(2.0);
    const auto result = physicsmade::quantum_chemistry::solveUnrestrictedHartreeFock(molecule);

    require(result.converged, "unrestricted Hartree-Fock should converge for minimal-basis H2+");
    require(result.basisCount == 2, "minimal H2+ basis should contain two orbitals");
    require(result.alphaElectronCount == 1 && result.betaElectronCount == 0, "minimal H2+ should default to a one-alpha open-shell configuration");
    require(result.totalEnergy < -0.5, "minimal-basis H2+ energy should be bound");
    require(result.totalEnergy > -1.5, "minimal-basis H2+ energy should stay within a plausible STO-3G range");
    require(!result.alphaOrbitalEnergies.empty() && result.alphaOrbitalEnergies.front() < 0.0, "UHF occupied alpha orbital energy should be negative");
}

void testRestrictedMollerPlesset2() {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule(1.4);
    const auto result = physicsmade::quantum_chemistry::solveRestrictedMollerPlesset2(molecule);

    require(result.scfConverged, "MP2 should inherit a converged RHF reference for minimal-basis H2");
    require(result.correlationEnergy < 0.0, "MP2 correlation energy should lower the Hartree-Fock reference energy");
    require(result.totalEnergy < result.hartreeFockTotalEnergy, "MP2 total energy should sit below the Hartree-Fock energy");
    require(result.totalEnergy > -1.3, "minimal-basis MP2 H2 energy should remain in a plausible range");
}

void testQuantumChemistryFieldSampling() {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule(1.4);
    auto steppedState = physicsmade::quantum_chemistry::initializeRestrictedHartreeFockState(molecule);
    while (!steppedState.converged && steppedState.iterations < steppedState.options.maxIterations) {
        physicsmade::quantum_chemistry::stepRestrictedHartreeFock(steppedState);
    }

    const auto solvedResult = physicsmade::quantum_chemistry::solveRestrictedHartreeFock(molecule);
    require(steppedState.converged, "stateful RHF stepping should converge for minimal-basis H2");
    require(relativeNearlyEqual(steppedState.totalEnergy, solvedResult.totalEnergy, 1.0e-10), "stateful RHF stepping should match the one-shot RHF result");

    const double centerDensity = physicsmade::quantum_chemistry::evaluateElectronDensity(
        steppedState.basis,
        steppedState.densityMatrix,
        {0.0, 0.0, 0.0});
    const double farDensity = physicsmade::quantum_chemistry::evaluateElectronDensity(
        steppedState.basis,
        steppedState.densityMatrix,
        {4.0, 0.0, 0.0});
    require(centerDensity > farDensity, "bond-center density should exceed far-field density for H2");

    const double bondingValue = physicsmade::quantum_chemistry::evaluateMolecularOrbital(
        steppedState.basis,
        steppedState.coefficientMatrix,
        0,
        {0.0, 0.0, 0.0});
    const double antibondingValue = physicsmade::quantum_chemistry::evaluateMolecularOrbital(
        steppedState.basis,
        steppedState.coefficientMatrix,
        1,
        {0.0, 0.0, 0.0});
    require(std::abs(bondingValue) > 1.0e-3, "bonding orbital amplitude should be nonzero at the H2 bond center");
    require(std::abs(antibondingValue) < 1.0e-6, "antibonding orbital should keep a node at the H2 bond center");
}

void testRenderPack() {
    const std::vector<physicsmade::scene::ObjectState> objects{
        {"left", 1.0, 0.0, 1.0, false, {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {"right", 1.0, 0.0, 2.0, false, {2.0, 3.0, -4.0}, {0.0, 2.0, 0.0}},
    };

    auto orientedObjects = objects;
    orientedObjects[0].orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, 0.5);
    orientedObjects[0].angularVelocity = {0.0, 0.0, 2.0};

    const auto pack = physicsmade::cuda::buildRenderInstancePack(orientedObjects);
    const auto gpuInstances = physicsmade::cuda::buildGpuRenderInstances(orientedObjects);
    require(pack.instances.size() == 2, "render pack instance count incorrect");
    require(gpuInstances.size() == 2, "GPU render pack instance count incorrect");
    require(nearlyEqual(pack.boundsMin.x, -1.0, 1.0e-12), "render pack min x incorrect");
    require(nearlyEqual(pack.boundsMax.y, 3.0, 1.0e-12), "render pack max y incorrect");
    require(relativeNearlyEqual(pack.instances[0].orientation.z, orientedObjects[0].orientation.z, 1.0e-12), "render pack orientation incorrect");
    require(relativeNearlyEqual(pack.instances[0].angularSpeed, 2.0, 1.0e-12), "render pack angular speed incorrect");
    require(gpuInstances[0].positionRadius[3] > 0.0f, "GPU render pack radius should be positive");
}

void testSceneBufferManager() {
    std::vector<physicsmade::scene::ObjectState> objects{
        {"left", 1.0, 0.0, 1.0, false, {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {"right", 1.0, 0.0, 2.0, false, {2.0, 3.0, -4.0}, {0.0, 2.0, 0.0}},
    };
    objects[0].angularVelocity = {0.0, 0.0, 1.0};

    auto manager = physicsmade::cuda::createSceneBufferManager();
    manager->reserve(objects.size());
    manager->update(objects);

    const auto interop = manager->interopView();
    const auto instances = manager->downloadRenderInstances();
    const auto gpuInstances = manager->downloadGpuRenderInstances();
    require(!interop.backend.empty(), "scene buffer backend should be reported");
    require(interop.renderInstanceBuffer.count == objects.size(), "scene buffer render count incorrect");
    require(interop.viewerInstanceBuffer.count == objects.size(), "scene buffer viewer count incorrect");
    require(interop.gravityBodyBuffer.count == objects.size(), "scene buffer gravity count incorrect");
    require(instances.size() == objects.size(), "scene buffer download size incorrect");
    require(gpuInstances.size() == objects.size(), "scene buffer GPU download size incorrect");
}

void testSceneSerializationRoundTrip() {
    physicsmade::io::SceneManifest manifest{};
    manifest.sceneName = "round-trip";
    manifest.kinematicsId = "newtonian";
    manifest.integratorId = "runge-kutta-4";
    manifest.previewSteps = 5;
    manifest.previewDtSeconds = 0.25;
    manifest.physicsSystems = {"newtonian-gravity", "lorentz-force"};
    manifest.distanceConstraints.push_back({0, 1, 2.0, 1.0});
    manifest.ballJointConstraints.push_back({0, 1, {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, 0.8, 0.15});
    manifest.hingeConstraints.push_back({
        {0, 1, {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, 0.85, 0.1},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        0.75,
        true,
        -0.35,
        0.5,
        true,
        2.0,
        40.0,
    });

    physicsmade::scene::ObjectState left{"left", 2.0, 1.0e-6, 0.5, false, {-1.0, 0.0, 0.0}, {0.0, 0.2, 0.0}};
    left.orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 1.0, 0.0}, 0.25);
    left.angularVelocity = {0.0, 1.0, 2.0};
    left.bodyInertiaDiagonal = {0.3, 0.4, 0.5};
    left.frictionCoefficient = 0.8;

    physicsmade::scene::ObjectState right{"right", 2.0, -1.0e-6, 0.5, false, {1.0, 0.0, 0.0}, {0.0, -0.2, 0.0}};
    right.orientation = physicsmade::math::Quaternion::fromAxisAngle({1.0, 0.0, 0.0}, -0.15);
    right.angularVelocity = {0.5, 0.0, 0.0};
    right.bodyInertiaDiagonal = {0.25, 0.35, 0.45};
    right.frictionCoefficient = 0.6;

    manifest.objects.push_back(left);
    manifest.objects.push_back(right);

    const auto text = physicsmade::io::serializeScene(manifest);
    const auto decoded = physicsmade::io::deserializeScene(text);
    require(decoded.sceneName == manifest.sceneName, "scene round-trip name incorrect");
    require(decoded.objects.size() == 2, "scene round-trip object count incorrect");
    require(decoded.distanceConstraints.size() == 1, "scene round-trip distance constraint count incorrect");
    require(decoded.ballJointConstraints.size() == 1, "scene round-trip ball joint count incorrect");
    require(decoded.hingeConstraints.size() == 1, "scene round-trip hinge count incorrect");
    require(decoded.hingeConstraints[0].limitsEnabled, "scene round-trip hinge limits incorrect");
    require(decoded.hingeConstraints[0].motorEnabled, "scene round-trip hinge motor incorrect");
    require(relativeNearlyEqual(decoded.hingeConstraints[0].targetAngularSpeed, 2.0, 1.0e-12), "scene round-trip hinge target speed incorrect");
    require(relativeNearlyEqual(decoded.objects[0].angularVelocity.z, 2.0, 1.0e-12), "scene round-trip angular velocity incorrect");
    require(relativeNearlyEqual(decoded.objects[0].frictionCoefficient, 0.8, 1.0e-12), "scene round-trip friction incorrect");

    std::vector<std::string> loadedSystems;
    auto world = physicsmade::io::instantiateScene(decoded, &loadedSystems);
    require(world.objects().size() == 2, "instantiated scene object count incorrect");
    require(loadedSystems.size() == 2, "instantiated scene system count incorrect");
}

void testArticulatedSceneAuthoring() {
    physicsmade::io::SceneManifest manifest{};
    manifest.sceneName = "servo-authoring";
    manifest.integratorId = "velocity-verlet";
    manifest.actuatorPresets.push_back({"servo", 0.95, true, -0.6, 0.6, true, 4.0, 60.0});

    physicsmade::scene::ObjectState base{"base-template", 1000.0, 0.0, 0.35, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    base.bodyInertiaDiagonal = {1.0, 1.0, 1.0};
    physicsmade::scene::ObjectState arm{"arm-template", 4.0, 0.0, 0.25, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    arm.bodyInertiaDiagonal = {0.45, 0.45, 0.18};
    arm.emissiveIntensity = 3.0;

    manifest.linkDefinitions.push_back({"base-link", base});
    manifest.linkDefinitions.push_back({"arm-link", arm});
    manifest.linkInstances.push_back({"base", "base-link", {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, false, false, 1.0, 1.0, 1.0});
    manifest.linkInstances.push_back({"arm", "arm-link", {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, false, false, 1.0, 1.0, 1.0});
    manifest.articulatedHinges.push_back({
        "servo-joint",
        "base",
        "arm",
        {0.0, 0.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        0.95,
        0.08,
        0.9,
        "servo",
    });
    manifest.loopDiagnostics.push_back({"servo-check", {"base", "arm"}, {"servo-joint"}, 0.0, 1.0e-9});

    const auto text = physicsmade::io::serializeScene(manifest);
    const auto decoded = physicsmade::io::deserializeScene(text);
    require(decoded.actuatorPresets.size() == 1, "actuator preset round-trip incorrect");
    require(decoded.linkDefinitions.size() == 2, "link definition round-trip incorrect");
    require(decoded.linkInstances.size() == 2, "link instance round-trip incorrect");
    require(decoded.articulatedHinges.size() == 1, "articulated hinge round-trip incorrect");
    require(decoded.loopDiagnostics.size() == 1, "loop diagnostic round-trip incorrect");

    const auto diagnostics = physicsmade::io::evaluateLoopDiagnostics(decoded);
    require(diagnostics.size() == 1, "loop diagnostic evaluation count incorrect");
    require(diagnostics[0].withinTolerance, "loop diagnostic should pass authored closure check");

    auto world = physicsmade::io::instantiateScene(decoded);
    require(world.objects().size() == 2, "articulated scene should instantiate authored links");
    for (int step = 0; step < 6; ++step) {
        world.step(0.02);
    }
    require(world.objects()[1].angularVelocity.z > 0.0, "actuator preset should drive authored hinge motion");
}

void testPluginDiscovery() {
    const auto pluginsRoot = resolvePluginsRoot();
    const auto manifests = physicsmade::plugins::discoverPlugins(pluginsRoot);
    require(manifests.size() >= 4, "plugin discovery should find the sample and generated plugins");

    const auto sampleOrbit = physicsmade::plugins::findPlugin(pluginsRoot, "sample-orbit");
    require(sampleOrbit.has_value(), "sample orbit plugin should exist");
    require(std::filesystem::exists(sampleOrbit->entryPath), "sample orbit scene entry should exist");

    const auto kerrGeodesic = physicsmade::plugins::findPlugin(pluginsRoot, "kerr-geodesic");
    require(kerrGeodesic.has_value(), "generated Kerr geodesic plugin should exist");
    require(kerrGeodesic->builtinSceneId == "kerr-geodesic", "generated Kerr geodesic plugin should resolve a builtin scene id");
}

void testScalarFieldLattice() {
    physicsmade::field_theory::ScalarFieldLattice lattice({64, 0.1, 1.0, 0.2});
    lattice.seedGaussianPacket(1.0, 3.0, 0.5, 4.0);
    lattice.step(0.002, 200);
    const auto observables = lattice.observables();

    require(observables.totalEnergy > 0.0, "scalar field energy should be positive");
    require(observables.maxAmplitude > 0.0, "scalar field amplitude should be nonzero");
}

void testRectangularScalarFieldLattice() {
    physicsmade::field_theory::RectangularScalarFieldLattice lattice({10, 8, 6, 0.25, 0.8, 0.12});
    lattice.seedGaussianPacket(1.0, {0.8, 0.7, 0.5}, 0.35, {1.5, 1.0, 0.5});
    lattice.step(0.002, 120);
    const auto observables = lattice.observables();

    require(observables.totalEnergy > 0.0, "rectangular scalar field energy should be positive");
    require(observables.gradientEnergy > 0.0, "rectangular scalar field gradient energy should be positive");
    require(observables.maxAmplitude > 0.0, "rectangular scalar field amplitude should be nonzero");
}

void testGaugeFieldLattice() {
    physicsmade::field_theory::U1GaugeFieldLattice lattice({10, 10, 6, 0.25, 1.0});
    lattice.seedStandingWave(0.1, {1.0, 0.5, 0.25});
    lattice.seedMagneticFluxSheet(0.05);
    lattice.step(0.01, 80);
    const auto observables = lattice.observables();

    require(observables.totalEnergy > 0.0, "gauge lattice total energy should be positive");
    require(observables.magneticEnergy > 0.0, "gauge lattice magnetic energy should be positive");
    require(std::abs(observables.meanPlaquette) <= 1.0 + 1.0e-12, "mean plaquette should stay bounded by 1");
}

void testSu2GaugeFieldLattice() {
    physicsmade::field_theory::SU2GaugeFieldLattice lattice({8, 8, 4, 0.25, 1.0});
    lattice.seedStandingWave(0.08, {1.0, 0.4, 0.2}, {0.0, 1.0, 1.0});
    lattice.seedColorMagneticFlux(0.04, {1.0, 0.3, 0.2});
    lattice.step(0.01, 80);
    const auto observables = lattice.observables();

    require(observables.totalEnergy > 0.0, "SU2 gauge lattice total energy should be positive");
    require(observables.magneticEnergy > 0.0, "SU2 gauge lattice magnetic energy should be positive");
    require(std::abs(observables.meanPlaquetteTrace) <= 1.0 + 1.0e-12, "SU2 mean plaquette trace should stay bounded by 1");
    require(observables.meanColorField.norm() > 0.0, "SU2 mean color field should be nonzero after seeding");
}

void testStaggeredFermionLattice() {
    physicsmade::field_theory::StaggeredFermionLattice lattice({8, 8, 4, 0.25, 0.55, 1.0, 0.8});
    lattice.seedGaussianPacket(1.0, {0.8, 0.8, 0.4}, 0.45, {1.0, 0.6, 0.25}, {0.2, 1.0, 0.5});
    lattice.seedColorFlux(0.06, {1.0, 0.2, 0.4});
    const auto initialObservables = lattice.observables();
    lattice.step(0.002, 120);
    const auto observables = lattice.observables();

    require(initialObservables.totalProbability > 0.0, "fermion lattice initial probability should be positive");
    require(observables.totalProbability > 0.0, "fermion lattice probability should stay positive");
    require(relativeNearlyEqual(observables.totalProbability, initialObservables.totalProbability, 5.0e-3), "fermion lattice probability should remain approximately conserved");
    require(observables.kineticActivity > 0.0, "fermion lattice kinetic activity should be positive");
    require(observables.maxSiteDensity > 0.0, "fermion lattice density should be nonzero");
    require(observables.meanCurrent.norm() > 0.0, "fermion lattice current should be nonzero after seeding");
}

void testCoupledSu2FermionLattice() {
    physicsmade::field_theory::CoupledSu2FermionLattice lattice({8, 8, 4, 0.25, 0.55, 1.0, 0.85, 0.3});
    lattice.seedFermionGaussianPacket(1.0, {0.8, 0.8, 0.4}, 0.45, {1.0, 0.6, 0.25}, {0.2, 1.0, 0.5});
    lattice.seedGaugeStandingWave(0.08, {1.0, 0.4, 0.2}, {0.0, 1.0, 1.0});
    lattice.seedColorFlux(0.04, {1.0, 0.3, 0.2});
    lattice.step(0.002, 120);
    const auto observables = lattice.observables();

    require(observables.totalProbability > 0.0, "coupled SU2 fermion lattice probability should stay positive");
    require(observables.fermionKineticActivity > 0.0, "coupled SU2 fermion lattice should produce fermion kinetic activity");
    require(observables.gaugeMagneticEnergy > 0.0, "coupled SU2 fermion lattice should produce magnetic energy");
    require(observables.interactionActivity > 0.0, "coupled SU2 fermion lattice should produce interaction activity");
    require(observables.totalEnergy > 0.0, "coupled SU2 fermion lattice total energy should be positive");
    require(std::abs(observables.meanPlaquetteTrace) <= 1.0 + 1.0e-12, "coupled SU2 fermion lattice plaquette trace should stay bounded");
    require(observables.meanCurrent.norm() > 0.0, "coupled SU2 fermion lattice mean current should be nonzero after seeding");
    require(observables.meanColorField.norm() > 0.0, "coupled SU2 fermion lattice mean color field should be nonzero after seeding");

    const auto samples = lattice.sampleSites();
    require(samples.size() == (lattice.spec().sizeX * lattice.spec().sizeY * lattice.spec().sizeZ), "coupled lattice site sampling should cover the full grid");
    require(std::any_of(samples.begin(), samples.end(), [](const auto& sample) {
        return sample.density > 0.0;
    }), "coupled lattice site sampling should expose nonzero density somewhere on the grid");
}


double harmonicOscillatorError(
    const physicsmade::physics::StateIntegrator& integrator,
    double dtSeconds) {
    std::vector<physicsmade::scene::ObjectState> states{
        {"oscillator", 1.0, 0.0, 0.1, false, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };
    const physicsmade::physics::NewtonianKinematics kinematics;
    const physicsmade::spacetime::SpacetimeModel spacetime;
    const physicsmade::physics::ForceEvaluator restoringForce =
        [](const std::vector<physicsmade::scene::ObjectState>& current,
           std::vector<physicsmade::math::Vector3>& forces) {
            forces.assign(current.size(), {});
            for (std::size_t i = 0; i < current.size(); ++i) {
                // Unit-mass/unit-frequency harmonic oscillator: x'' = -x.
                forces[i].x = -current[i].position.x;
            }
        };

    const int steps = static_cast<int>(std::llround(1.0 / dtSeconds));
    for (int step = 0; step < steps; ++step) {
        integrator.integrate(states, dtSeconds, restoringForce, kinematics, spacetime);
    }

    const double exactX = std::cos(1.0);
    const double exactV = -std::sin(1.0);
    const double dx = states.front().position.x - exactX;
    const double dv = states.front().velocity.x - exactV;
    return std::sqrt((dx * dx) + (dv * dv));
}

void testIntegratorConvergenceOrders() {
    const physicsmade::physics::VelocityVerletIntegrator verlet;
    const physicsmade::physics::RungeKutta4Integrator rk4;

    const double verletCoarse = harmonicOscillatorError(verlet, 0.2);
    const double verletFine = harmonicOscillatorError(verlet, 0.1);
    const double rk4Coarse = harmonicOscillatorError(rk4, 0.2);
    const double rk4Fine = harmonicOscillatorError(rk4, 0.1);

    require(verletFine < verletCoarse, "Velocity Verlet must converge under timestep refinement");
    require(rk4Fine < rk4Coarse, "RK4 must converge under timestep refinement");

    const double verletRatio = verletCoarse / verletFine;
    const double rk4Ratio = rk4Coarse / rk4Fine;

    require(verletRatio > 3.2 && verletRatio < 4.8,
            "Velocity Verlet should exhibit second-order global convergence on the analytic oscillator benchmark");
    require(rk4Ratio > 12.0 && rk4Ratio < 20.0,
            "RK4 should exhibit fourth-order global convergence on the analytic oscillator benchmark");
    require(rk4Fine < 0.01 * verletFine,
            "RK4 fine-step error should be decisively below Velocity Verlet on the same benchmark");
}

void testGeneratedScenes() {
    const auto quantumChemistryScene = physicsmade::runtime::buildGeneratedScene("quantum-chemistry-live");
    require(quantumChemistryScene.has_value(), "generated quantum chemistry scene should exist");
    require(quantumChemistryScene->objects.size() > 40, "generated quantum chemistry scene should expose live density probes");

    const auto kerrScene = physicsmade::runtime::buildGeneratedScene("kerr-geodesic");
    require(kerrScene.has_value(), "generated Kerr scene should exist");
    require(kerrScene->objects.size() > 20, "generated Kerr scene should expose a breadcrumb trajectory");

    const auto coupledScene = physicsmade::runtime::buildGeneratedScene("coupled-su2-fermion-state");
    require(coupledScene.has_value(), "generated coupled lattice scene should exist");
    require(coupledScene->objects.size() > 10, "generated coupled lattice scene should expose visible lattice probes");

    auto world = physicsmade::io::instantiateScene(*coupledScene);
    require(!world.objects().empty(), "generated coupled lattice scene should instantiate into a non-empty world");

    const auto liveKerrScene = physicsmade::runtime::buildGeneratedViewerScene("kerr-geodesic");
    require(liveKerrScene.has_value(), "live Kerr viewer scene should exist");
    auto liveKerrWorld = physicsmade::io::instantiateScene(liveKerrScene->manifest);
    const auto initialTravelerPosition = liveKerrWorld.objects()[static_cast<std::size_t>(liveKerrScene->suggestedFollowIndex)].position;
    liveKerrScene->step(liveKerrWorld, liveKerrScene->manifest.previewDtSeconds);
    const auto updatedTravelerPosition = liveKerrWorld.objects()[static_cast<std::size_t>(liveKerrScene->suggestedFollowIndex)].position;
    require((updatedTravelerPosition - initialTravelerPosition).norm() > 1.0e-6, "live Kerr viewer scene should move the tracked traveler");

    const auto liveCoupledScene = physicsmade::runtime::buildGeneratedViewerScene("coupled-su2-fermion-state");
    require(liveCoupledScene.has_value(), "live coupled lattice viewer scene should exist");
    auto liveCoupledWorld = physicsmade::io::instantiateScene(liveCoupledScene->manifest);
    const auto initialCoupledObjects = liveCoupledWorld.objects();
    liveCoupledScene->step(liveCoupledWorld, liveCoupledScene->manifest.previewDtSeconds);
    bool coupledSceneChanged = false;
    for (std::size_t index = 0; index < initialCoupledObjects.size(); ++index) {
        const auto& before = initialCoupledObjects[index];
        const auto& after = liveCoupledWorld.objects()[index];
        if (!relativeNearlyEqual(before.radius, after.radius, 1.0e-9) ||
            !relativeNearlyEqual(before.emissiveIntensity, after.emissiveIntensity, 1.0e-9) ||
            (after.position - before.position).norm() > 1.0e-9) {
            coupledSceneChanged = true;
            break;
        }
    }
    require(coupledSceneChanged, "live coupled lattice viewer scene should update rendered object state");

    const auto liveQuantumChemistryScene = physicsmade::runtime::buildGeneratedViewerScene("quantum-chemistry-live");
    require(liveQuantumChemistryScene.has_value(), "live quantum chemistry viewer scene should exist");
    auto liveQuantumChemistryWorld = physicsmade::io::instantiateScene(liveQuantumChemistryScene->manifest);
    const auto initialQuantumChemistryObjects = liveQuantumChemistryWorld.objects();
    liveQuantumChemistryScene->step(liveQuantumChemistryWorld, liveQuantumChemistryScene->manifest.previewDtSeconds);
    bool quantumChemistrySceneChanged = false;
    for (std::size_t index = 0; index < initialQuantumChemistryObjects.size(); ++index) {
        const auto& before = initialQuantumChemistryObjects[index];
        const auto& after = liveQuantumChemistryWorld.objects()[index];
        if (!relativeNearlyEqual(before.radius, after.radius, 1.0e-9) ||
            !relativeNearlyEqual(before.emissiveIntensity, after.emissiveIntensity, 1.0e-9) ||
            !relativeNearlyEqual(before.charge, after.charge, 1.0e-9)) {
            quantumChemistrySceneChanged = true;
            break;
        }
    }
    require(quantumChemistrySceneChanged, "live quantum chemistry viewer scene should update sampled electronic structure");
}

void testRuntimeCatalog() {
    const auto objectPrograms = physicsmade::runtime::availableObjectPrograms();
    const auto integrators = physicsmade::runtime::availableIntegrators();
    const auto launchTargets = physicsmade::runtime::availableLaunchTargets();

    require(objectPrograms.size() >= 15, "object program catalog should expose built-in objects");
    require(integrators.size() >= 3, "integrator catalog should expose built-in integrators");
    require(launchTargets.size() >= 36, "launch catalog should expose launcher targets and discovered plugins");

    std::ostringstream output;
    require(
        physicsmade::runtime::runObjectProgram(
            "rigid-spinner",
            physicsmade::physics::KinematicRegime::NonRelativistic,
            "runge-kutta-4",
            output) == 0,
        "rigid spinner object program should run");
    require(output.str().find("angular_speed=") != std::string::npos, "object program output missing angular diagnostics");
    require(output.str().find("orientation=") != std::string::npos, "object program output missing orientation diagnostics");
    require(output.str().find("scene_buffer_backend=") != std::string::npos, "object program output missing scene buffer summary");

    std::ostringstream quantumOutput;
    require(physicsmade::runtime::runLaunchTarget("quantum-scalar-preview", quantumOutput) == 0, "quantum preview should run");
    require(quantumOutput.str().find("lattice-scalar-field") != std::string::npos, "quantum preview output missing model marker");

    std::ostringstream scalarGridOutput;
    require(physicsmade::runtime::runLaunchTarget("quantum-scalar-grid-preview", scalarGridOutput) == 0, "scalar grid preview should run");
    require(scalarGridOutput.str().find("rectangular-scalar-lattice") != std::string::npos, "scalar grid preview output missing model marker");

    std::ostringstream gaugeOutput;
    require(physicsmade::runtime::runLaunchTarget("gauge-u1-preview", gaugeOutput) == 0, "gauge preview should run");
    require(gaugeOutput.str().find("compact-u1-gauge-lattice") != std::string::npos, "gauge preview output missing model marker");

    std::ostringstream su2GaugeOutput;
    require(physicsmade::runtime::runLaunchTarget("gauge-su2-preview", su2GaugeOutput) == 0, "SU2 gauge preview should run");
    require(su2GaugeOutput.str().find("compact-su2-gauge-lattice") != std::string::npos, "SU2 gauge preview output missing model marker");

    std::ostringstream fermionOutput;
    require(physicsmade::runtime::runLaunchTarget("fermion-staggered-preview", fermionOutput) == 0, "fermion preview should run");
    require(fermionOutput.str().find("staggered-su2-fermion-lattice") != std::string::npos, "fermion preview output missing model marker");

    std::ostringstream schwarzschildOutput;
    require(physicsmade::runtime::runLaunchTarget("schwarzschild-preview", schwarzschildOutput) == 0, "Schwarzschild preview should run");
    require(schwarzschildOutput.str().find("schwarzschild-isotropic") != std::string::npos, "Schwarzschild preview output missing metric marker");

    std::ostringstream kerrOutput;
    require(physicsmade::runtime::runLaunchTarget("kerr-preview", kerrOutput) == 0, "Kerr preview should run");
    require(kerrOutput.str().find("kerr-schild-cartesian") != std::string::npos, "Kerr preview output missing metric marker");

    std::ostringstream schwarzschildGeodesicOutput;
    require(physicsmade::runtime::runLaunchTarget("schwarzschild-geodesic-preview", schwarzschildGeodesicOutput) == 0, "Schwarzschild geodesic preview should run");
    require(schwarzschildGeodesicOutput.str().find("radius_drift=") != std::string::npos, "Schwarzschild geodesic preview output missing drift diagnostics");

    std::ostringstream kerrGeodesicOutput;
    require(physicsmade::runtime::runLaunchTarget("kerr-geodesic-preview", kerrGeodesicOutput) == 0, "Kerr geodesic preview should run");
    require(kerrGeodesicOutput.str().find("normalized_tangent_norm_drift=") != std::string::npos, "Kerr geodesic preview output missing norm diagnostics");

    std::ostringstream chemistryOutput;
    require(physicsmade::runtime::runLaunchTarget("quantum-chemistry-preview", chemistryOutput) == 0, "quantum chemistry preview should run");
    require(chemistryOutput.str().find("restricted-hartree-fock-sto3g") != std::string::npos, "quantum chemistry preview output missing model marker");

    std::ostringstream chemistryUhfOutput;
    require(physicsmade::runtime::runLaunchTarget("quantum-chemistry-uhf-preview", chemistryUhfOutput) == 0, "unrestricted quantum chemistry preview should run");
    require(chemistryUhfOutput.str().find("unrestricted-hartree-fock-sto3g") != std::string::npos, "unrestricted quantum chemistry preview output missing model marker");

    std::ostringstream chemistryMp2Output;
    require(physicsmade::runtime::runLaunchTarget("quantum-chemistry-mp2-preview", chemistryMp2Output) == 0, "MP2 quantum chemistry preview should run");
    require(chemistryMp2Output.str().find("restricted-moller-plesset-2-sto3g") != std::string::npos, "MP2 quantum chemistry preview output missing model marker");

    std::ostringstream coupledFermionOutput;
    require(physicsmade::runtime::runLaunchTarget("coupled-su2-fermion-preview", coupledFermionOutput) == 0, "coupled SU2 fermion preview should run");
    require(coupledFermionOutput.str().find("coupled-su2-staggered-fermion-lattice") != std::string::npos, "coupled SU2 fermion preview output missing model marker");

    std::ostringstream linkageOutput;
    require(
        physicsmade::runtime::runObjectProgram(
            "four-bar-loop",
            physicsmade::physics::KinematicRegime::NonRelativistic,
            "velocity-verlet",
            linkageOutput) == 0,
        "four-bar loop object program should run");
    require(linkageOutput.str().find("program=four-bar-loop") != std::string::npos, "four-bar loop output missing program marker");

    std::ostringstream pluginOutput;
    require(physicsmade::runtime::runLaunchTarget("plugin:sample-orbit", pluginOutput) == 0, "sample orbit plugin should run");
    require(pluginOutput.str().find("scene_name=sample-orbit") != std::string::npos, "plugin preview output missing scene name");

    std::ostringstream fourBarPluginOutput;
    require(physicsmade::runtime::runLaunchTarget("plugin:four-bar-loop", fourBarPluginOutput) == 0, "four-bar plugin should run");
    require(fourBarPluginOutput.str().find("scene_name=four-bar-loop") != std::string::npos, "four-bar plugin preview output missing scene name");
    require(fourBarPluginOutput.str().find("loop_diagnostic_count=") != std::string::npos, "four-bar plugin preview should report loop diagnostics");

    std::ostringstream kerrPluginOutput;
    require(physicsmade::runtime::runLaunchTarget("plugin:kerr-geodesic", kerrPluginOutput) == 0, "generated Kerr plugin should run");
    require(kerrPluginOutput.str().find("scene_name=kerr-geodesic") != std::string::npos, "generated Kerr plugin preview output missing scene name");

    std::ostringstream coupledPluginOutput;
    require(physicsmade::runtime::runLaunchTarget("plugin:coupled-su2-fermion-state", coupledPluginOutput) == 0, "generated coupled lattice plugin should run");
    require(coupledPluginOutput.str().find("scene_name=coupled-su2-fermion-state") != std::string::npos, "generated coupled lattice plugin preview output missing scene name");

    std::ostringstream quantumChemistryPluginOutput;
    require(physicsmade::runtime::runLaunchTarget("plugin:quantum-chemistry-live", quantumChemistryPluginOutput) == 0, "generated quantum chemistry plugin should run");
    require(quantumChemistryPluginOutput.str().find("scene_name=quantum-chemistry-live") != std::string::npos, "generated quantum chemistry plugin preview output missing scene name");
}

}  // namespace

int main() {
    try {
        testVectorMath();
        testQuaternionRotation();
        testLightlikeInterval();
        testGravityForces();
        testOrbitCamera();
        testFieldSampling();
        testGpuStyleGravityBodyForces();
        testElectrostaticForces();
        testLorentzForces();
        testSpringForces();
        testDistanceConstraint();
        testBallJointConstraint();
        testHingeConstraint();
        testHingeLimitConstraint();
        testHingeMotorConstraint();
        testContactSolver();
        testIntegratorConvergenceOrders();
        testWorldStepping();
        testRigidBodyRotation();
        testSpecialRelativisticKinematics();
        testRelativisticWorldTiming();
        testWorldDiagnostics();
        testWeakFieldMetric();
        testSchwarzschildMetric();
        testKerrMetric();
        testNumericGeodesicIntegrator();
        testSchwarzschildGeodesicIntegrator();
        testKerrGeodesicIntegrator();
        testRestrictedHartreeFock();
        testUnrestrictedHartreeFock();
        testRestrictedMollerPlesset2();
        testQuantumChemistryFieldSampling();
        testRenderPack();
        testSceneBufferManager();
        testSceneSerializationRoundTrip();
        testArticulatedSceneAuthoring();
        testPluginDiscovery();
        testScalarFieldLattice();
        testRectangularScalarFieldLattice();
        testGaugeFieldLattice();
        testSu2GaugeFieldLattice();
        testStaggeredFermionLattice();
        testCoupledSu2FermionLattice();
        testGeneratedScenes();
        testRuntimeCatalog();
    } catch (const std::exception& exception) {
        std::cerr << "test failure: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "physicsmade_tests passed\n";
    return EXIT_SUCCESS;
}

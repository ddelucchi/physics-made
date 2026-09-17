#include "physicsmade/runtime/programs.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "physicsmade/camera/orbit_camera.hpp"
#include "physicsmade/common/config.hpp"
#include "physicsmade/cuda/field_grid.hpp"
#include "physicsmade/cuda/gravity_body_forces.hpp"
#include "physicsmade/cuda/scene_buffers.hpp"
#include "physicsmade/field_theory/coupled_su2_fermion_lattice.hpp"
#include "physicsmade/field_theory/rectangular_scalar_field_lattice.hpp"
#include "physicsmade/field_theory/scalar_field_lattice.hpp"
#include "physicsmade/field_theory/staggered_fermion_lattice.hpp"
#include "physicsmade/field_theory/su2_gauge_field_lattice.hpp"
#include "physicsmade/field_theory/u1_gauge_field_lattice.hpp"
#include "physicsmade/io/scene_serialization.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/physics/constraints.hpp"
#include "physicsmade/physics/electrostatic_system.hpp"
#include "physicsmade/physics/hookean_spring_system.hpp"
#include "physicsmade/physics/kinematics_model.hpp"
#include "physicsmade/physics/linear_drag_system.hpp"
#include "physicsmade/physics/lorentz_force_system.hpp"
#include "physicsmade/physics/newtonian_gravity_system.hpp"
#include "physicsmade/plugins/plugin_discovery.hpp"
#include "physicsmade/quantum_chemistry/restricted_hartree_fock.hpp"
#include "physicsmade/runtime/generated_scenes.hpp"
#include "physicsmade/scene/sim_object.hpp"
#include "physicsmade/spacetime/geodesic_integrator.hpp"
#include "physicsmade/spacetime/kerr_metric.hpp"
#include "physicsmade/spacetime/schwarzschild_metric.hpp"
#include "physicsmade/spacetime/weak_field_metric.hpp"
#include "physicsmade/simulation/simulation_world.hpp"

namespace {

class ConfiguredObject final : public physicsmade::scene::SimObject {
  public:
    explicit ConfiguredObject(physicsmade::scene::ObjectState state, std::string kind)
        : state_(std::move(state)), kind_(std::move(kind)) {}

    std::string kind() const override {
        return kind_;
    }

    physicsmade::scene::ObjectState initialState() const override {
        return state_;
    }

  private:
    physicsmade::scene::ObjectState state_;
    std::string kind_;
};

void printVector(std::ostream& output, const physicsmade::math::Vector3& vector) {
    output << std::scientific << std::setprecision(6)
           << "(" << vector.x << ", " << vector.y << ", " << vector.z << ")";
}

void printQuaternion(std::ostream& output, const physicsmade::math::Quaternion& quaternion) {
    output << std::scientific << std::setprecision(6)
           << "(" << quaternion.w << ", " << quaternion.x << ", " << quaternion.y << ", " << quaternion.z << ")";
}

void printObjectSummary(
    std::ostream& output,
    const physicsmade::simulation::SimulationWorld& world,
    std::size_t index) {
    const auto& object = world.objects()[index];
    const auto diagnostics = world.diagnosticsFor(index);

    output << "object=" << object.name << "\n";
    output << "position=";
    printVector(output, object.position);
    output << "\n";
    output << "velocity=";
    printVector(output, object.velocity);
    output << "\n";
    output << "momentum=";
    printVector(output, diagnostics.momentum);
    output << "\n";
    output << "orientation=";
    printQuaternion(output, object.orientation);
    output << "\n";
    output << "angular_velocity=";
    printVector(output, diagnostics.angularVelocity);
    output << "\n";
    output << "angular_momentum=";
    printVector(output, diagnostics.angularMomentum);
    output << "\n";
    output << "coordinate_time_seconds=" << diagnostics.coordinateTimeSeconds << "\n";
    output << "proper_time_seconds=" << diagnostics.properTimeSeconds << "\n";
    output << "speed=" << diagnostics.speed << "\n";
    output << "angular_speed=" << diagnostics.angularSpeed << "\n";
    output << "lorentz_gamma=" << diagnostics.lorentzGamma << "\n";
    output << "translational_kinetic_energy=" << diagnostics.translationalKineticEnergy << "\n";
    output << "rotational_kinetic_energy=" << diagnostics.rotationalKineticEnergy << "\n";
    output << "kinetic_energy=" << diagnostics.kineticEnergy << "\n";
}

void printWorldSummary(
    std::ostream& output,
    const physicsmade::simulation::SimulationWorld& world) {
    const auto diagnostics = world.worldDiagnostics();
    output << "object_count=" << diagnostics.objectCount << "\n";
    output << "total_mass=" << diagnostics.totalMass << "\n";
    output << "total_charge=" << diagnostics.totalCharge << "\n";
    output << "total_translational_kinetic_energy=" << diagnostics.totalTranslationalKineticEnergy << "\n";
    output << "total_rotational_kinetic_energy=" << diagnostics.totalRotationalKineticEnergy << "\n";
    output << "total_kinetic_energy=" << diagnostics.totalKineticEnergy << "\n";
    output << "total_potential_energy=" << diagnostics.totalPotentialEnergy << "\n";
    output << "total_energy=" << diagnostics.totalEnergy << "\n";
    output << "total_momentum=";
    printVector(output, diagnostics.totalMomentum);
    output << "\n";
    output << "total_angular_momentum=";
    printVector(output, diagnostics.totalAngularMomentum);
    output << "\n";
    output << "center_of_mass=";
    printVector(output, diagnostics.centerOfMass);
    output << "\n";
    output << "max_speed=" << diagnostics.maxSpeed << "\n";
}

void printInteropSummary(
    std::ostream& output,
    const std::vector<physicsmade::scene::ObjectState>& objects) {
    auto buffers = physicsmade::cuda::createSceneBufferManager();
    buffers->reserve(objects.size());
    buffers->update(objects);

    const auto view = buffers->interopView();
    output << "scene_buffer_backend=" << view.backend << "\n";
    output << "scene_buffers_device_resident=" << (view.deviceResident ? "true" : "false") << "\n";
    output << "gravity_body_buffer_address=" << view.gravityBodyBuffer.deviceAddress << "\n";
    output << "gravity_body_buffer_count=" << view.gravityBodyBuffer.count << "\n";
    output << "render_instance_buffer_address=" << view.renderInstanceBuffer.deviceAddress << "\n";
    output << "render_instance_count=" << view.renderInstanceBuffer.count << "\n";
    output << "viewer_instance_buffer_address=" << view.viewerInstanceBuffer.deviceAddress << "\n";
    output << "viewer_instance_count=" << view.viewerInstanceBuffer.count << "\n";
    output << "render_bounds_min=";
    printVector(output, view.boundsMin);
    output << "\n";
    output << "render_bounds_max=";
    printVector(output, view.boundsMax);
    output << "\n";
}

std::unique_ptr<physicsmade::physics::KinematicsModel> makeKinematics(physicsmade::physics::KinematicRegime regime) {
    switch (regime) {
        case physicsmade::physics::KinematicRegime::NonRelativistic:
            return std::make_unique<physicsmade::physics::NewtonianKinematics>();
        case physicsmade::physics::KinematicRegime::SpecialRelativistic:
            return std::make_unique<physicsmade::physics::SpecialRelativisticKinematics>();
    }

    return std::make_unique<physicsmade::physics::NewtonianKinematics>();
}

std::unique_ptr<physicsmade::physics::StateIntegrator> makeIntegrator(std::string_view id) {
    if (id == "semi-implicit-euler") {
        return std::make_unique<physicsmade::physics::SemiImplicitEulerIntegrator>();
    }

    if (id == "velocity-verlet") {
        return std::make_unique<physicsmade::physics::VelocityVerletIntegrator>();
    }

    if (id == "runge-kutta-4") {
        return std::make_unique<physicsmade::physics::RungeKutta4Integrator>();
    }

    return nullptr;
}

bool isKnownIntegrator(std::string_view id) {
    return id == "semi-implicit-euler" || id == "velocity-verlet" || id == "runge-kutta-4";
}

void configureWorld(
    physicsmade::simulation::SimulationWorld& world,
    physicsmade::physics::KinematicRegime regime,
    std::string_view integratorId) {
    world.setKinematicsModel(makeKinematics(regime));
    if (auto integrator = makeIntegrator(integratorId)) {
        world.setIntegrator(std::move(integrator));
    }
}

struct ObjectProgramSetup {
    physicsmade::simulation::SimulationWorld world{};
    std::size_t primaryIndex{0};
    int steps{0};
    double dtSeconds{0.0};
};

ObjectProgramSetup buildObjectProgram(
    std::string_view id,
    physicsmade::physics::KinematicRegime regime,
    std::string_view integratorId) {
    ObjectProgramSetup setup{};
    configureWorld(setup.world, regime, integratorId);

    if (id == "inertial-probe") {
        ConfiguredObject probe({"inertial-probe", 1'200.0, 0.0, 2.0, false, {0.0, 0.0, 0.0}, {220.0, 35.0, 0.0}}, "probe");
        setup.primaryIndex = setup.world.addObject(probe);
        setup.steps = 120;
        setup.dtSeconds = 0.5;
        return setup;
    }

    if (id == "planet-seed") {
        setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::NewtonianGravitySystem>());

        ConfiguredObject star(
            {"reference-star", 1.98847e30, 0.0, 6.9634e8, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
            "star");
        ConfiguredObject planet(
            {"planet-seed", 5.9722e24, 0.0, 6.371e6, false, {1.495978707e11, 0.0, 0.0}, {0.0, 29'780.0, 0.0}},
            "planet");

        setup.world.addObject(star);
        setup.primaryIndex = setup.world.addObject(planet);
        setup.steps = 96;
        setup.dtSeconds = 900.0;
        return setup;
    }

    if (id == "relativistic-probe") {
        const double c = setup.world.spacetime().speedOfLight();
        ConfiguredObject probe(
            {"relativistic-probe", 750.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.8 * c, 0.0, 0.0}},
            "probe");
        setup.primaryIndex = setup.world.addObject(probe);
        setup.steps = 10;
        setup.dtSeconds = 1.0;
        return setup;
    }

    if (id == "charged-dipole") {
        setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::LorentzForceSystem>());

        ConfiguredObject leftCharge(
            {"left-charge", 1.0, 2.0e-6, 0.2, false, {-1.0, 0.0, 0.0}, {0.0, 0.15, 0.0}},
            "charged-particle");
        ConfiguredObject rightCharge(
            {"right-charge", 1.0, -2.0e-6, 0.2, false, {1.0, 0.0, 0.0}, {0.0, -0.15, 0.0}},
            "charged-particle");

        setup.primaryIndex = setup.world.addObject(leftCharge);
        setup.world.addObject(rightCharge);
        setup.steps = 300;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "magnetic-pair") {
        setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::LorentzForceSystem>());

        ConfiguredObject upper(
            {"upper-charge", 1.0, 3.0e-6, 0.25, false, {0.0, 1.0, 0.0}, {0.7, 0.0, 0.0}},
            "charged-particle");
        ConfiguredObject lower(
            {"lower-charge", 1.0, 3.0e-6, 0.25, false, {0.0, -1.0, 0.0}, {0.7, 0.0, 0.0}},
            "charged-particle");

        setup.primaryIndex = setup.world.addObject(upper);
        setup.world.addObject(lower);
        setup.steps = 400;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "rigid-spinner") {
        physicsmade::scene::ObjectState spinner{"rigid-spinner", 6.0, 0.0, 0.45, false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        spinner.bodyInertiaDiagonal = {0.35, 0.85, 1.15};
        spinner.angularVelocity = {0.0, 5.0, 9.0};
        spinner.emissiveIntensity = 3.0;
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(spinner, "rigid-body"));
        setup.steps = 240;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "ball-joint-swing") {
        physicsmade::scene::ObjectState anchor{"joint-anchor", 500.0, 0.0, 0.4, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState bob{"joint-bob", 4.0, 0.0, 0.35, false, {0.5, -2.0, 0.0}, {1.0, 0.0, 0.4}};
        bob.bodyInertiaDiagonal = {0.22, 0.22, 0.22};

        setup.world.addObject(ConfiguredObject(anchor, "anchor"));
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(bob, "rigid-bob"));
        setup.world.addConstraint(std::make_unique<physicsmade::physics::BallJointConstraint>(
            physicsmade::physics::BallJointConstraintSpec{0, 1, {0.0, -1.0, 0.0}, {0.0, 1.0, 0.0}, 0.85, 0.12}));
        setup.steps = 500;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "hinge-rotor") {
        physicsmade::scene::ObjectState stator{"hinge-stator", 1000.0, 0.0, 0.4, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState rotor{"hinge-rotor", 5.0, 0.0, 0.4, false, {0.0, 2.0, 0.0}, {0.0, 0.0, 0.0}};
        rotor.angularVelocity = {0.0, 12.0, 0.0};
        rotor.bodyInertiaDiagonal = {0.6, 0.25, 0.6};

        setup.world.addObject(ConfiguredObject(stator, "stator"));
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(rotor, "rotor"));
        physicsmade::physics::HingeConstraintSpec hinge{
            physicsmade::physics::BallJointConstraintSpec{0, 1, {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0}, 0.9, 0.1},
            {0.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
        };
        hinge.angularStiffness = 0.85;
        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(hinge));
        setup.steps = 400;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "motorized-hinge-limit") {
        physicsmade::scene::ObjectState base{"servo-base", 1000.0, 0.0, 0.35, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState arm{"servo-arm", 4.0, 0.0, 0.3, false, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        arm.bodyInertiaDiagonal = {0.45, 0.45, 0.18};
        arm.emissiveIntensity = 4.0;

        setup.world.addObject(ConfiguredObject(base, "servo-base"));
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(arm, "servo-arm"));

        physicsmade::physics::HingeConstraintSpec hinge{
            physicsmade::physics::BallJointConstraintSpec{0, 1, {0.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}, 0.95, 0.1},
            {0.0, 0.0, 1.0},
            {0.0, 0.0, 1.0},
        };
        hinge.leftLocalReference = {1.0, 0.0, 0.0};
        hinge.rightLocalReference = {1.0, 0.0, 0.0};
        hinge.angularStiffness = 0.95;
        hinge.limitsEnabled = true;
        hinge.lowerAngleLimit = -0.7;
        hinge.upperAngleLimit = 0.7;
        hinge.motorEnabled = true;
        hinge.targetAngularSpeed = 3.5;
        hinge.maxMotorTorque = 80.0;
        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(hinge));

        setup.steps = 540;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "four-bar-loop") {
        physicsmade::scene::ObjectState ground{"four-bar-ground", 3000.0, 0.0, 0.35, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState crank{"four-bar-crank", 3.5, 0.0, 0.28, false, {-1.6, 0.692820323, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState coupler{"four-bar-coupler", 4.5, 0.0, 0.28, false, {-0.004, 1.3813, 0.0}, {0.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState rocker{"four-bar-rocker", 3.5, 0.0, 0.28, false, {1.596, 0.6885, 0.0}, {0.0, 0.0, 0.0}};

        ground.bodyInertiaDiagonal = {1.0, 1.0, 1.0};
        crank.bodyInertiaDiagonal = {0.55, 0.55, 0.2};
        coupler.bodyInertiaDiagonal = {0.8, 0.8, 0.28};
        rocker.bodyInertiaDiagonal = {0.55, 0.55, 0.2};

        crank.orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, 1.0471975512);
        coupler.orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, -0.0038);
        rocker.orientation = physicsmade::math::Quaternion::fromAxisAngle({0.0, 0.0, 1.0}, -1.0402);
        crank.emissiveIntensity = 4.0;
        coupler.emissiveIntensity = 2.5;
        rocker.emissiveIntensity = 3.5;

        setup.world.addObject(ConfiguredObject(ground, "ground-link"));
        setup.world.addObject(ConfiguredObject(crank, "crank-link"));
        setup.world.addObject(ConfiguredObject(coupler, "coupler-link"));
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(rocker, "rocker-link"));

        auto makeFourBarHinge = [](std::size_t leftIndex,
                                   std::size_t rightIndex,
                                   const physicsmade::math::Vector3& leftAnchor,
                                   const physicsmade::math::Vector3& rightAnchor) {
            physicsmade::physics::HingeConstraintSpec hinge{
                physicsmade::physics::BallJointConstraintSpec{leftIndex, rightIndex, leftAnchor, rightAnchor, 0.95, 0.08},
                {0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0},
            };
            hinge.leftLocalReference = {1.0, 0.0, 0.0};
            hinge.rightLocalReference = {1.0, 0.0, 0.0};
            hinge.angularStiffness = 0.92;
            return hinge;
        };

        auto driveHinge = makeFourBarHinge(0, 1, {-2.0, 0.0, 0.0}, {-0.8, 0.0, 0.0});
        driveHinge.motorEnabled = true;
        driveHinge.targetAngularSpeed = 2.8;
        driveHinge.maxMotorTorque = 120.0;

        auto couplerLeft = makeFourBarHinge(1, 2, {0.8, 0.0, 0.0}, {-1.2, 0.0, 0.0});
        auto couplerRight = makeFourBarHinge(2, 3, {1.2, 0.0, 0.0}, {-0.8, 0.0, 0.0});
        auto supportHinge = makeFourBarHinge(3, 0, {0.8, 0.0, 0.0}, {2.0, 0.0, 0.0});
        supportHinge.limitsEnabled = true;
        supportHinge.lowerAngleLimit = -1.2;
        supportHinge.upperAngleLimit = 1.2;

        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(driveHinge));
        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(couplerLeft));
        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(couplerRight));
        setup.world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(supportHinge));

        setup.steps = 720;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "collision-pair") {
        physicsmade::scene::ObjectState left{"left-collider", 4.0, 0.0, 0.5, false, {-2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
        physicsmade::scene::ObjectState right{"right-collider", 4.0, 0.0, 0.5, false, {2.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
        left.restitution = 0.9;
        right.restitution = 0.9;

        setup.primaryIndex = setup.world.addObject(ConfiguredObject(left, "rigid-sphere"));
        setup.world.addObject(ConfiguredObject(right, "rigid-sphere"));
        setup.steps = 500;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "contact-spinner") {
        physicsmade::scene::ObjectState left{"contact-left", 3.0, 0.0, 0.45, false, {-1.4, 0.25, 0.0}, {1.4, -0.4, 0.0}};
        physicsmade::scene::ObjectState right{"contact-right", 3.0, 0.0, 0.45, false, {1.4, -0.25, 0.0}, {-1.0, 0.1, 0.0}};
        left.restitution = 0.8;
        right.restitution = 0.8;
        left.frictionCoefficient = 0.9;
        right.frictionCoefficient = 0.9;

        setup.primaryIndex = setup.world.addObject(ConfiguredObject(left, "rigid-sphere"));
        setup.world.addObject(ConfiguredObject(right, "rigid-sphere"));
        setup.steps = 320;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "constraint-dumbbell") {
        physicsmade::scene::ObjectState left{"dumbbell-left", 2.0, 0.0, 0.3, false, {-1.5, 0.0, 0.0}, {0.0, 0.6, 0.0}};
        physicsmade::scene::ObjectState right{"dumbbell-right", 2.0, 0.0, 0.3, false, {1.5, 0.0, 0.0}, {0.0, -0.6, 0.0}};
        setup.primaryIndex = setup.world.addObject(ConfiguredObject(left, "rigid-node"));
        setup.world.addObject(ConfiguredObject(right, "rigid-node"));
        setup.world.addConstraint(std::make_unique<physicsmade::physics::DistanceConstraint>(
            physicsmade::physics::DistanceConstraintSpec{0, 1, 3.0, 1.0}));
        setup.steps = 500;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "spring-pair") {
        setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::HookeanSpringSystem>(
            std::vector<physicsmade::physics::SpringLink>{{0, 1, 2.5, 14.0, 0.2}}));

        ConfiguredObject anchor({"anchor", 1000.0, 0.0, 0.5, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, "anchor");
        ConfiguredObject bob({"spring-bob", 4.0, 0.0, 0.35, false, {4.0, 0.0, 0.0}, {0.0, 0.6, 0.0}}, "spring-bob");

        setup.world.addObject(anchor);
        setup.primaryIndex = setup.world.addObject(bob);
        setup.steps = 800;
        setup.dtSeconds = 0.01;
        return setup;
    }

    if (id == "drag-probe") {
        setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::LinearDragSystem>(1.225, 0.5));

        physicsmade::scene::ObjectState probeState{"drag-probe", 80.0, 0.0, 0.3, false, {0.0, 0.0, 0.0}, {220.0, 0.0, 0.0}};
        probeState.dragCoefficient = 0.82;
        probeState.temperatureKelvin = 1200.0;
        probeState.emissiveIntensity = 5.0;

        ConfiguredObject probe(probeState, "reentry-probe");
        setup.primaryIndex = setup.world.addObject(probe);
        setup.steps = 300;
        setup.dtSeconds = 0.05;
        return setup;
    }

    return {};
}

physicsmade::physics::KinematicRegime defaultRegimeForObject(std::string_view id) {
    if (id == "relativistic-probe") {
        return physicsmade::physics::KinematicRegime::SpecialRelativistic;
    }

    return physicsmade::physics::KinematicRegime::NonRelativistic;
}

bool isKnownObjectProgram(std::string_view id) {
    return id == "inertial-probe" ||
           id == "planet-seed" ||
           id == "relativistic-probe" ||
           id == "charged-dipole" ||
           id == "magnetic-pair" ||
           id == "rigid-spinner" ||
           id == "ball-joint-swing" ||
           id == "hinge-rotor" ||
           id == "motorized-hinge-limit" ||
           id == "four-bar-loop" ||
           id == "collision-pair" ||
           id == "contact-spinner" ||
           id == "constraint-dumbbell" ||
           id == "spring-pair" ||
           id == "drag-probe";
}

std::filesystem::path resolvePluginsRoot() {
    auto cursor = std::filesystem::current_path();
    while (true) {
        const auto candidate = cursor / "plugins";
        if (std::filesystem::exists(candidate) && std::filesystem::is_directory(candidate)) {
            return candidate;
        }

        if (cursor == cursor.root_path()) {
            return std::filesystem::current_path() / "plugins";
        }

        cursor = cursor.parent_path();
    }
}

struct SandboxSetup {
    physicsmade::simulation::SimulationWorld world{};
    std::size_t planetIndex{0};
};

SandboxSetup buildSandboxScene() {
    SandboxSetup setup{};
    configureWorld(setup.world, physicsmade::physics::KinematicRegime::NonRelativistic, "velocity-verlet");
    setup.world.addPhysicsSystem(std::make_unique<physicsmade::physics::NewtonianGravitySystem>());

    physicsmade::scene::ObjectState starState{"star", 1.98847e30, 0.0, 6.9634e8, true, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    starState.temperatureKelvin = 5778.0;
    starState.emissiveIntensity = 60.0;
    ConfiguredObject star(starState, "star");

    physicsmade::scene::ObjectState planetState{"planet", 5.9722e24, 0.0, 6.371e6, false, {1.495978707e11, 0.0, 0.0}, {0.0, 29'780.0, 0.0}};
    planetState.temperatureKelvin = 288.0;
    planetState.emissiveIntensity = 1.0;
    ConfiguredObject planet(planetState, "planet");

    physicsmade::scene::ObjectState probeState{"probe", 1'200.0, 0.0, 2.0, false, {1.495978707e11 + 4.2e7, 0.0, 0.0}, {0.0, 29'780.0 + 3'200.0, 0.0}};
    probeState.temperatureKelvin = 900.0;
    probeState.emissiveIntensity = 8.0;
    ConfiguredObject probe(probeState, "probe");

    setup.world.addObject(star);
    setup.planetIndex = setup.world.addObject(planet);
    setup.world.addObject(probe);
    return setup;
}

int runWeakFieldPreview(std::ostream& output) {
    const auto metric = std::make_shared<physicsmade::spacetime::WeakFieldSphericalMassMetric>(1.98847e30);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const physicsmade::math::FourVector eventA{0.0, 1.495978707e11, 0.0, 0.0};
    const physicsmade::math::FourVector eventB{physicsmade::common::kSpeedOfLight * 60.0, 1.495978707e11, 1.8e6, 0.0};

    output << "Physics Made weak-field preview\n";
    output << "metric=" << metric->name() << "\n";
    output << "interval_squared=" << spacetime.intervalSquared(eventA, eventB) << "\n";
    return 0;
}

int runRenderPreview(std::ostream& output) {
    auto sandbox = buildSandboxScene();
    for (int step = 0; step < 48; ++step) {
        sandbox.world.step(1'800.0);
    }

    output << "Physics Made render preview\n";
    printInteropSummary(output, sandbox.world.objects());
    return 0;
}

}  // namespace

namespace physicsmade::runtime {

std::vector<ObjectProgramDescriptor> availableObjectPrograms() {
    return {
        {"inertial-probe", "Run a standalone inertial object with no external forces.", physics::KinematicRegime::NonRelativistic},
        {"planet-seed", "Run a single planet object inside a minimal star-plus-gravity scene.", physics::KinematicRegime::NonRelativistic},
        {"relativistic-probe", "Run a single high-velocity object under special-relativistic kinematics.", physics::KinematicRegime::SpecialRelativistic},
        {"charged-dipole", "Run a pair of charged particles with electric and magnetic Lorentz terms.", physics::KinematicRegime::NonRelativistic},
        {"magnetic-pair", "Run parallel moving charges to expose magnetic interaction terms.", physics::KinematicRegime::NonRelativistic},
        {"rigid-spinner", "Run a free rigid body with anisotropic inertia and observable spin evolution.", physics::KinematicRegime::NonRelativistic},
        {"ball-joint-swing", "Run a rigid body constrained by a ball joint with off-center anchors.", physics::KinematicRegime::NonRelativistic},
        {"hinge-rotor", "Run a rotor constrained to a single hinge axis.", physics::KinematicRegime::NonRelativistic},
        {"motorized-hinge-limit", "Run a servo-like hinge with active drive and angular hard stops.", physics::KinematicRegime::NonRelativistic},
        {"four-bar-loop", "Run a closed-loop four-bar linkage driven by a motorized hinge.", physics::KinematicRegime::NonRelativistic},
        {"collision-pair", "Run two spheres through contact resolution without recompiling physics code.", physics::KinematicRegime::NonRelativistic},
        {"contact-spinner", "Run a glancing rigid-sphere impact that generates spin through frictional contact.", physics::KinematicRegime::NonRelativistic},
        {"constraint-dumbbell", "Run two rigid nodes locked by a distance constraint.", physics::KinematicRegime::NonRelativistic},
        {"spring-pair", "Run a spring-coupled anchor and bob system.", physics::KinematicRegime::NonRelativistic},
        {"drag-probe", "Run a drag-damped probe to inspect dissipative motion.", physics::KinematicRegime::NonRelativistic},
    };
}

std::vector<IntegratorDescriptor> availableIntegrators() {
    return {
        {"semi-implicit-euler", "Simple first-order integrator."},
        {"velocity-verlet", "Second-order symplectic-friendly position/velocity integrator."},
        {"runge-kutta-4", "Classical fourth-order Runge-Kutta integrator."},
    };
}

std::vector<LaunchTargetDescriptor> availableLaunchTargets() {
    std::vector<LaunchTargetDescriptor> targets{
        {"sandbox", "Run the reference multi-object spacetime sandbox."},
        {"relativity-compare", "Compare Newtonian and special-relativistic propagation for the same probe."},
        {"render-preview", "Pack the live scene into persistent renderer-facing buffers and report interop handles."},
        {"weak-field-preview", "Inspect a weak-field relativistic metric around a central mass."},
        {"kerr-preview", "Inspect a rotating Kerr metric in Cartesian Kerr-Schild coordinates."},
        {"kerr-geodesic-preview", "Integrate a timelike geodesic through the rotating Kerr geometry."},
        {"schwarzschild-preview", "Inspect an exact isotropic Schwarzschild metric around a central mass."},
        {"schwarzschild-geodesic-preview", "Integrate a timelike geodesic through the exact isotropic Schwarzschild geometry."},
        {"quantum-chemistry-preview", "Run a minimal-basis restricted Hartree-Fock molecular calculation."},
        {"quantum-chemistry-uhf-preview", "Run an open-shell minimal-basis unrestricted Hartree-Fock molecular calculation."},
        {"quantum-chemistry-mp2-preview", "Run a post-Hartree-Fock MP2 correction on top of the restricted molecular calculation."},
        {"quantum-scalar-preview", "Run a lattice scalar-field theory preview with Klein-Gordon-style dynamics."},
        {"quantum-scalar-grid-preview", "Run a higher-dimensional scalar lattice preview over a rectangular volume."},
        {"gauge-u1-preview", "Run a compact U(1) gauge-field scaffold and report plaquette observables."},
        {"gauge-su2-preview", "Run a compact SU(2) non-Abelian gauge scaffold and report color plaquette observables."},
        {"fermion-staggered-preview", "Run a staggered fermion lattice with SU(2)-style color transport and report probability/current observables."},
        {"coupled-su2-fermion-preview", "Run a coupled SU(2) gauge plus staggered-fermion lattice with explicit backreaction."},
    };

    for (const auto& program : availableObjectPrograms()) {
        targets.push_back({"object:" + program.id, "Run object program '" + program.id + "' with its default regime."});
    }

    const auto pluginsRoot = resolvePluginsRoot();
    for (const auto& manifest : physicsmade::plugins::discoverPlugins(pluginsRoot)) {
        targets.push_back({"plugin:" + manifest.id, "Run discovered " + std::string(physicsmade::plugins::toString(manifest.kind)) + " plugin '" + manifest.id + "'."});
    }

    return targets;
}

int runObjectProgram(
    std::string_view id,
    physics::KinematicRegime regime,
    std::string_view integratorId,
    std::ostream& output) {
    if (!isKnownObjectProgram(id)) {
        output << "Unknown object program: " << id << "\n";
        return 1;
    }

    if (!isKnownIntegrator(integratorId)) {
        output << "Unknown integrator: " << integratorId << "\n";
        return 1;
    }

    auto setup = buildObjectProgram(id, regime, integratorId);
    for (int step = 0; step < setup.steps; ++step) {
        setup.world.step(setup.dtSeconds);
    }

    output << "Physics Made object runner\n";
    output << "program=" << id << "\n";
    output << "regime=" << physics::toString(regime) << "\n";
    output << "integrator=" << integratorId << "\n";
    output << "constraint_solver=" << setup.world.constraintSolverName() << "\n";
    output << "kinematics_model=" << setup.world.kinematicsModel().name() << "\n";
    output << "simulation_time_seconds=" << setup.world.simulationTimeSeconds() << "\n";
    printWorldSummary(output, setup.world);
    printObjectSummary(output, setup.world, setup.primaryIndex);
    printInteropSummary(output, setup.world.objects());
    return 0;
}

int runSandbox(std::ostream& output) {
    using physicsmade::camera::OrbitCamera;
    using physicsmade::cuda::FieldGridSpec;
    using physicsmade::math::Vector3;

    auto sandbox = buildSandboxScene();

    for (int step = 0; step < 48; ++step) {
        sandbox.world.step(1'800.0);
    }

    const auto& planetState = sandbox.world.objects()[sandbox.planetIndex];

    OrbitCamera camera(1.2e8, 0.35, 0.45);
    camera.orbit(0.25, -0.05);
    const auto pose = camera.pose(planetState.position);

    const Vector3 localExtent{1.5e7, 1.5e7, 1.5e7};
    const FieldGridSpec fieldSpec{4, 4, 4, planetState.position - localExtent, {1.0e7, 1.0e7, 1.0e7}};
    const auto field = physicsmade::cuda::evaluateGravityField(sandbox.world.objects(), fieldSpec);
    const auto& centerSample = field.at(1, 1, 1);

    output << "Physics Made sandbox\n";
    output << "integrator=" << sandbox.world.integratorName() << "\n";
    output << "constraint_solver=" << sandbox.world.constraintSolverName() << "\n";
    output << "kinematics_model=" << sandbox.world.kinematicsModel().name() << "\n";
    output << "simulation_time_seconds=" << sandbox.world.simulationTimeSeconds() << "\n";
    output << "cuda_field_solver=" << (physicsmade::cuda::cudaFieldSolverAvailable() ? "enabled" : "cpu-fallback") << "\n";
    output << "cuda_body_force_solver=" << (physicsmade::cuda::cudaBodyForceSolverAvailable() ? "enabled" : "cpu-fallback") << "\n";
    output << "cuda_scene_buffers=" << (physicsmade::cuda::cudaSceneBuffersAvailable() ? "enabled" : "cpu-fallback") << "\n";
    printWorldSummary(output, sandbox.world);
    printObjectSummary(output, sandbox.world, sandbox.planetIndex);
    printInteropSummary(output, sandbox.world.objects());

    output << "camera_position=";
    printVector(output, pose.position);
    output << "\n";
    output << "camera_forward=";
    printVector(output, pose.forward);
    output << "\n";
    output << "sample_acceleration=";
    printVector(output, centerSample.acceleration);
    output << "\n";
    output << "sample_potential=" << centerSample.potential << "\n";
    return 0;
}

int runRelativityComparison(std::ostream& output) {
    using physicsmade::simulation::SimulationWorld;

    const double c = physicsmade::common::kSpeedOfLight;
    const physicsmade::scene::ObjectState probe{"comparison-probe", 25.0, 0.0, 1.0, false, {0.0, 0.0, 0.0}, {0.8 * c, 0.0, 0.0}};

    SimulationWorld newtonianWorld;
    newtonianWorld.setKinematicsModel(std::make_unique<physicsmade::physics::NewtonianKinematics>());
    newtonianWorld.setIntegrator(std::make_unique<physicsmade::physics::RungeKutta4Integrator>());
    const auto newtonianIndex = newtonianWorld.addObject(probe);

    SimulationWorld relativisticWorld;
    relativisticWorld.setKinematicsModel(std::make_unique<physicsmade::physics::SpecialRelativisticKinematics>());
    relativisticWorld.setIntegrator(std::make_unique<physicsmade::physics::RungeKutta4Integrator>());
    const auto relativisticIndex = relativisticWorld.addObject(probe);

    for (int step = 0; step < 5; ++step) {
        newtonianWorld.step(1.0);
        relativisticWorld.step(1.0);
    }

    const auto newtonianDiagnostics = newtonianWorld.diagnosticsFor(newtonianIndex);
    const auto relativisticDiagnostics = relativisticWorld.diagnosticsFor(relativisticIndex);

    output << "Physics Made relativity comparison\n";
    output << "probe_speed_fraction_of_c=" << (probe.velocity.norm() / c) << "\n";
    output << "newtonian_proper_time_seconds=" << newtonianDiagnostics.properTimeSeconds << "\n";
    output << "special_relativity_proper_time_seconds=" << relativisticDiagnostics.properTimeSeconds << "\n";
    output << "newtonian_gamma=" << newtonianDiagnostics.lorentzGamma << "\n";
    output << "special_relativity_gamma=" << relativisticDiagnostics.lorentzGamma << "\n";
    output << "newtonian_kinetic_energy=" << newtonianDiagnostics.kineticEnergy << "\n";
    output << "special_relativity_kinetic_energy=" << relativisticDiagnostics.kineticEnergy << "\n";
    return 0;
}

int runDiscoveredScenePreview(const std::filesystem::path& pluginsRoot, std::string_view pluginId, std::ostream& output) {
    const auto manifest = physicsmade::plugins::findPlugin(pluginsRoot, std::string(pluginId));
    if (!manifest) {
        output << "Unknown discovered plugin: " << pluginId << "\n";
        return 1;
    }

    physicsmade::io::SceneManifest sceneManifest{};
    if (!manifest->builtinSceneId.empty()) {
        const auto generated = physicsmade::runtime::buildGeneratedScene(manifest->builtinSceneId);
        if (!generated) {
            output << "Unknown generated plugin scene: " << manifest->builtinSceneId << "\n";
            return 1;
        }
        sceneManifest = *generated;
    } else {
        sceneManifest = physicsmade::io::readSceneFile(manifest->entryPath);
    }
    std::vector<std::string> loadedSystems;
    auto world = physicsmade::io::instantiateScene(sceneManifest, &loadedSystems);
    for (int step = 0; step < sceneManifest.previewSteps; ++step) {
        world.step(sceneManifest.previewDtSeconds);
    }

    output << "Physics Made discovered plugin preview\n";
    output << "plugin_id=" << manifest->id << "\n";
    output << "plugin_kind=" << physicsmade::plugins::toString(manifest->kind) << "\n";
    output << "scene_name=" << sceneManifest.sceneName << "\n";
    output << "integrator=" << world.integratorName() << "\n";
    output << "constraint_solver=" << world.constraintSolverName() << "\n";
    output << "kinematics_model=" << world.kinematicsModel().name() << "\n";
    output << "loaded_system_count=" << loadedSystems.size() << "\n";
    const auto loopDiagnostics = physicsmade::io::evaluateLoopDiagnostics(sceneManifest);
    output << "loop_diagnostic_count=" << loopDiagnostics.size() << "\n";
    for (const auto& diagnostic : loopDiagnostics) {
        output << "loop_diagnostic." << diagnostic.id << ".closure_distance=" << diagnostic.chainClosureDistance << "\n";
        output << "loop_diagnostic." << diagnostic.id << ".max_anchor_mismatch=" << diagnostic.maxAnchorMismatch << "\n";
        output << "loop_diagnostic." << diagnostic.id << ".within_tolerance=" << (diagnostic.withinTolerance ? "true" : "false") << "\n";
    }
    printWorldSummary(output, world);
    if (!world.objects().empty()) {
        printObjectSummary(output, world, 0);
    }
    printInteropSummary(output, world.objects());
    return 0;
}

int runKerrPreview(std::ostream& output) {
    const double solarMass = 1.98847e30;
    const auto metric = std::make_shared<physicsmade::spacetime::KerrMetric>(4.154e6 * solarMass, 0.78);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const double evaluationRadius = 12.0 * metric->gravitationalRadius();
    const physicsmade::math::FourVector eventA{0.0, evaluationRadius, 0.0, 0.0};
    const physicsmade::math::FourVector eventB{physicsmade::common::kSpeedOfLight * 60.0, evaluationRadius, 0.15 * evaluationRadius, 0.0};
    const auto tensor = metric->covariant(eventA);

    output << "Physics Made Kerr preview\n";
    output << "metric=" << metric->name() << "\n";
    output << "central_mass=" << metric->centralMass() << "\n";
    output << "dimensionless_spin=" << metric->dimensionlessSpin() << "\n";
    output << "gravitational_radius=" << metric->gravitationalRadius() << "\n";
    output << "spin_parameter=" << metric->spinParameter() << "\n";
    output << "event_horizon_radius=" << metric->eventHorizonRadius() << "\n";
    output << "g_tt=" << tensor[0][0] << "\n";
    output << "g_tx=" << tensor[0][1] << "\n";
    output << "g_ty=" << tensor[0][2] << "\n";
    output << "g_xx=" << tensor[1][1] << "\n";
    output << "kerr_interval_squared=" << spacetime.intervalSquared(eventA, eventB) << "\n";
    return 0;
}

int runKerrGeodesicPreview(std::ostream& output) {
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

    output << "Physics Made Kerr geodesic preview\n";
    output << "metric=" << metric->name() << "\n";
    output << "dimensionless_spin=" << metric->dimensionlessSpin() << "\n";
    output << "initial_radius=" << initialState.position.spatial().norm() << "\n";
    output << "minimum_radius=" << minimumRadius << "\n";
    output << "final_radius=" << finalState.position.spatial().norm() << "\n";
    output << "initial_speed=" << physicsmade::spacetime::coordinateSpeed(initialState) << "\n";
    output << "final_speed=" << physicsmade::spacetime::coordinateSpeed(finalState) << "\n";
    output << "final_azimuth_radians=" << std::atan2(finalState.position.y, finalState.position.x) << "\n";
    output << "final_affine_parameter=" << finalState.affineParameter << "\n";
    output << "initial_tangent_norm=" << physicsmade::spacetime::tangentNormSquared(spacetime, initialState) << "\n";
    output << "final_tangent_norm=" << physicsmade::spacetime::tangentNormSquared(spacetime, finalState) << "\n";
    output << "normalized_tangent_norm_drift=" << normalizedDrift << "\n";
    return 0;
}

int runSchwarzschildPreview(std::ostream& output) {
    const double solarMass = 1.98847e30;
    const auto metric = std::make_shared<physicsmade::spacetime::SchwarzschildMetric>(solarMass);
    const auto weakFieldMetric = std::make_shared<physicsmade::spacetime::WeakFieldSphericalMassMetric>(solarMass);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const physicsmade::spacetime::SpacetimeModel weakFieldSpacetime(weakFieldMetric);
    const physicsmade::math::FourVector eventA{0.0, 1.495978707e11, 0.0, 0.0};
    const physicsmade::math::FourVector eventB{physicsmade::common::kSpeedOfLight * 60.0, 1.495978707e11, 1.8e6, 0.0};
    const auto tensor = metric->covariant(eventA);
    const auto weakFieldTensor = weakFieldMetric->covariant(eventA);

    output << "Physics Made Schwarzschild preview\n";
    output << "metric=" << metric->name() << "\n";
    output << "central_mass=" << solarMass << "\n";
    output << "schwarzschild_radius=" << metric->schwarzschildRadius() << "\n";
    output << "isotropic_horizon_radius=" << metric->isotropicHorizonRadius() << "\n";
    output << "g_tt=" << tensor[0][0] << "\n";
    output << "g_xx=" << tensor[1][1] << "\n";
    output << "weak_field_g_tt=" << weakFieldTensor[0][0] << "\n";
    output << "weak_field_g_xx=" << weakFieldTensor[1][1] << "\n";
    output << "schwarzschild_interval_squared=" << spacetime.intervalSquared(eventA, eventB) << "\n";
    output << "weak_field_interval_squared=" << weakFieldSpacetime.intervalSquared(eventA, eventB) << "\n";
    return 0;
}

int runSchwarzschildGeodesicPreview(std::ostream& output) {
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

    output << "Physics Made Schwarzschild geodesic preview\n";
    output << "metric=" << metric->name() << "\n";
    output << "initial_radius=" << initialState.position.spatial().norm() << "\n";
    output << "final_radius=" << finalState.position.spatial().norm() << "\n";
    output << "radius_drift=" << (finalState.position.spatial().norm() - initialState.position.spatial().norm()) << "\n";
    output << "initial_speed=" << physicsmade::spacetime::coordinateSpeed(initialState) << "\n";
    output << "final_speed=" << physicsmade::spacetime::coordinateSpeed(finalState) << "\n";
    output << "final_affine_parameter=" << finalState.affineParameter << "\n";
    output << "initial_tangent_norm=" << physicsmade::spacetime::tangentNormSquared(spacetime, initialState) << "\n";
    output << "final_tangent_norm=" << physicsmade::spacetime::tangentNormSquared(spacetime, finalState) << "\n";
    return 0;
}

int runQuantumChemistryPreview(std::ostream& output) {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule(1.4);
    const auto result = physicsmade::quantum_chemistry::solveRestrictedHartreeFock(molecule);

    output << "Physics Made quantum chemistry preview\n";
    output << "model=restricted-hartree-fock-sto3g\n";
    output << "molecule=H2\n";
    output << "electron_count=" << molecule.electronCount << "\n";
    output << "basis_count=" << result.basisCount << "\n";
    output << "occupied_orbitals=" << result.occupiedOrbitals << "\n";
    output << "converged=" << (result.converged ? "true" : "false") << "\n";
    output << "iterations=" << result.iterations << "\n";
    output << "electronic_energy=" << result.electronicEnergy << "\n";
    output << "nuclear_repulsion_energy=" << result.nuclearRepulsionEnergy << "\n";
    output << "total_energy=" << result.totalEnergy << "\n";
    if (!result.orbitalEnergies.empty()) {
        output << "orbital_energy_0=" << result.orbitalEnergies[0] << "\n";
    }
    if (result.orbitalEnergies.size() > 1) {
        output << "orbital_energy_1=" << result.orbitalEnergies[1] << "\n";
    }
    return 0;
}

int runQuantumChemistryUhfPreview(std::ostream& output) {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecularCation(2.0);
    const auto result = physicsmade::quantum_chemistry::solveUnrestrictedHartreeFock(molecule);

    output << "Physics Made unrestricted quantum chemistry preview\n";
    output << "model=unrestricted-hartree-fock-sto3g\n";
    output << "molecule=H2+\n";
    output << "electron_count=" << molecule.electronCount << "\n";
    output << "basis_count=" << result.basisCount << "\n";
    output << "alpha_electrons=" << result.alphaElectronCount << "\n";
    output << "beta_electrons=" << result.betaElectronCount << "\n";
    output << "converged=" << (result.converged ? "true" : "false") << "\n";
    output << "iterations=" << result.iterations << "\n";
    output << "electronic_energy=" << result.electronicEnergy << "\n";
    output << "nuclear_repulsion_energy=" << result.nuclearRepulsionEnergy << "\n";
    output << "total_energy=" << result.totalEnergy << "\n";
    if (!result.alphaOrbitalEnergies.empty()) {
        output << "alpha_orbital_energy_0=" << result.alphaOrbitalEnergies[0] << "\n";
    }
    if (!result.betaOrbitalEnergies.empty()) {
        output << "beta_orbital_energy_0=" << result.betaOrbitalEnergies[0] << "\n";
    }
    return 0;
}

int runQuantumChemistryMp2Preview(std::ostream& output) {
    const auto molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule(1.4);
    const auto result = physicsmade::quantum_chemistry::solveRestrictedMollerPlesset2(molecule);

    output << "Physics Made MP2 quantum chemistry preview\n";
    output << "model=restricted-moller-plesset-2-sto3g\n";
    output << "molecule=H2\n";
    output << "electron_count=" << molecule.electronCount << "\n";
    output << "basis_count=" << result.basisCount << "\n";
    output << "occupied_orbitals=" << result.occupiedOrbitals << "\n";
    output << "scf_converged=" << (result.scfConverged ? "true" : "false") << "\n";
    output << "scf_iterations=" << result.scfIterations << "\n";
    output << "hartree_fock_total_energy=" << result.hartreeFockTotalEnergy << "\n";
    output << "correlation_energy=" << result.correlationEnergy << "\n";
    output << "total_energy=" << result.totalEnergy << "\n";
    if (!result.orbitalEnergies.empty()) {
        output << "orbital_energy_0=" << result.orbitalEnergies[0] << "\n";
    }
    if (result.orbitalEnergies.size() > 1) {
        output << "orbital_energy_1=" << result.orbitalEnergies[1] << "\n";
    }
    return 0;
}

int runQuantumScalarPreview(std::ostream& output) {
    physicsmade::field_theory::ScalarFieldLattice lattice({128, 0.05, 1.2, 0.25});
    lattice.seedGaussianPacket(1.0, 3.0, 0.4, 6.0);
    lattice.step(0.002, 600);
    const auto observables = lattice.observables();

    output << "Physics Made quantum scalar preview\n";
    output << "model=lattice-scalar-field\n";
    output << "site_count=" << lattice.spec().siteCount << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "mass=" << lattice.spec().mass << "\n";
    output << "self_coupling=" << lattice.spec().selfCoupling << "\n";
    output << "total_energy=" << observables.totalEnergy << "\n";
    output << "mean_field=" << observables.meanField << "\n";
    output << "max_amplitude=" << observables.maxAmplitude << "\n";
    output << "nearest_neighbor_correlation=" << observables.nearestNeighborCorrelation << "\n";
    return 0;
}

int runQuantumScalarGridPreview(std::ostream& output) {
    physicsmade::field_theory::RectangularScalarFieldLattice lattice({20, 16, 12, 0.2, 0.9, 0.15});
    lattice.seedGaussianPacket(1.0, {1.8, 1.4, 1.0}, 0.55, {2.0, 1.5, 1.0});
    lattice.step(0.002, 240);
    const auto observables = lattice.observables();

    output << "Physics Made quantum scalar grid preview\n";
    output << "model=rectangular-scalar-lattice\n";
    output << "size_x=" << lattice.spec().sizeX << "\n";
    output << "size_y=" << lattice.spec().sizeY << "\n";
    output << "size_z=" << lattice.spec().sizeZ << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "mass=" << lattice.spec().mass << "\n";
    output << "self_coupling=" << lattice.spec().selfCoupling << "\n";
    output << "total_energy=" << observables.totalEnergy << "\n";
    output << "gradient_energy=" << observables.gradientEnergy << "\n";
    output << "potential_energy=" << observables.potentialEnergy << "\n";
    output << "mean_field=" << observables.meanField << "\n";
    output << "max_amplitude=" << observables.maxAmplitude << "\n";
    return 0;
}

int runGaugeFieldPreview(std::ostream& output) {
    physicsmade::field_theory::U1GaugeFieldLattice lattice({18, 18, 8, 0.25, 1.0});
    lattice.seedStandingWave(0.12, {1.0, 0.5, 0.25});
    lattice.seedMagneticFluxSheet(0.04);
    lattice.step(0.01, 160);
    const auto observables = lattice.observables();

    output << "Physics Made compact U1 gauge preview\n";
    output << "model=compact-u1-gauge-lattice\n";
    output << "size_x=" << lattice.spec().sizeX << "\n";
    output << "size_y=" << lattice.spec().sizeY << "\n";
    output << "size_z=" << lattice.spec().sizeZ << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "coupling=" << lattice.spec().coupling << "\n";
    output << "electric_energy=" << observables.electricEnergy << "\n";
    output << "magnetic_energy=" << observables.magneticEnergy << "\n";
    output << "total_energy=" << observables.totalEnergy << "\n";
    output << "mean_plaquette=" << observables.meanPlaquette << "\n";
    output << "max_electric_field=" << observables.maxElectricField << "\n";
    return 0;
}

int runGaugeSu2Preview(std::ostream& output) {
    physicsmade::field_theory::SU2GaugeFieldLattice lattice({16, 16, 8, 0.25, 1.0});
    lattice.seedStandingWave(0.08, {0.8, 0.4, 0.25}, {0.0, 1.0, 1.0});
    lattice.seedColorMagneticFlux(0.05, {1.0, 0.2, 0.4});
    lattice.step(0.01, 120);
    const auto observables = lattice.observables();

    output << "Physics Made compact SU2 gauge preview\n";
    output << "model=compact-su2-gauge-lattice\n";
    output << "size_x=" << lattice.spec().sizeX << "\n";
    output << "size_y=" << lattice.spec().sizeY << "\n";
    output << "size_z=" << lattice.spec().sizeZ << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "coupling=" << lattice.spec().coupling << "\n";
    output << "electric_energy=" << observables.electricEnergy << "\n";
    output << "magnetic_energy=" << observables.magneticEnergy << "\n";
    output << "total_energy=" << observables.totalEnergy << "\n";
    output << "mean_plaquette_trace=" << observables.meanPlaquetteTrace << "\n";
    output << "max_electric_field=" << observables.maxElectricField << "\n";
    output << "mean_color_field=";
    printVector(output, observables.meanColorField);
    output << "\n";
    return 0;
}

int runStaggeredFermionPreview(std::ostream& output) {
    physicsmade::field_theory::StaggeredFermionLattice lattice({18, 18, 8, 0.2, 0.55, 1.0, 0.85});
    lattice.seedGaussianPacket(1.0, {1.4, 1.2, 0.8}, 0.5, {1.2, 0.7, 0.35}, {0.3, 1.0, 0.4});
    lattice.seedColorFlux(0.08, {1.0, 0.25, 0.45});
    lattice.step(0.002, 220);
    const auto observables = lattice.observables();

    output << "Physics Made staggered fermion preview\n";
    output << "model=staggered-su2-fermion-lattice\n";
    output << "size_x=" << lattice.spec().sizeX << "\n";
    output << "size_y=" << lattice.spec().sizeY << "\n";
    output << "size_z=" << lattice.spec().sizeZ << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "mass=" << lattice.spec().mass << "\n";
    output << "hopping=" << lattice.spec().hopping << "\n";
    output << "gauge_coupling=" << lattice.spec().gaugeCoupling << "\n";
    output << "total_probability=" << observables.totalProbability << "\n";
    output << "kinetic_activity=" << observables.kineticActivity << "\n";
    output << "mass_activity=" << observables.massActivity << "\n";
    output << "total_activity=" << observables.totalActivity << "\n";
    output << "max_site_density=" << observables.maxSiteDensity << "\n";
    output << "staggered_charge=" << observables.staggeredCharge << "\n";
    output << "mean_current=";
    printVector(output, observables.meanCurrent);
    output << "\n";
    return 0;
}

int runCoupledSu2FermionPreview(std::ostream& output) {
    physicsmade::field_theory::CoupledSu2FermionLattice lattice({14, 14, 6, 0.22, 0.55, 1.0, 0.85, 0.3});
    lattice.seedFermionGaussianPacket(1.0, {1.1, 1.0, 0.6}, 0.45, {1.0, 0.6, 0.25}, {0.2, 1.0, 0.5});
    lattice.seedGaugeStandingWave(0.08, {0.9, 0.4, 0.2}, {0.0, 1.0, 1.0});
    lattice.seedColorFlux(0.05, {1.0, 0.25, 0.4});
    lattice.step(0.0025, 160);
    const auto observables = lattice.observables();

    output << "Physics Made coupled SU2 fermion preview\n";
    output << "model=coupled-su2-staggered-fermion-lattice\n";
    output << "size_x=" << lattice.spec().sizeX << "\n";
    output << "size_y=" << lattice.spec().sizeY << "\n";
    output << "size_z=" << lattice.spec().sizeZ << "\n";
    output << "spacing=" << lattice.spec().spacing << "\n";
    output << "mass=" << lattice.spec().mass << "\n";
    output << "hopping=" << lattice.spec().hopping << "\n";
    output << "gauge_coupling=" << lattice.spec().gaugeCoupling << "\n";
    output << "fermion_backreaction=" << lattice.spec().fermionBackreaction << "\n";
    output << "total_probability=" << observables.totalProbability << "\n";
    output << "fermion_kinetic_activity=" << observables.fermionKineticActivity << "\n";
    output << "fermion_mass_activity=" << observables.fermionMassActivity << "\n";
    output << "gauge_electric_energy=" << observables.gaugeElectricEnergy << "\n";
    output << "gauge_magnetic_energy=" << observables.gaugeMagneticEnergy << "\n";
    output << "interaction_activity=" << observables.interactionActivity << "\n";
    output << "total_energy=" << observables.totalEnergy << "\n";
    output << "max_site_density=" << observables.maxSiteDensity << "\n";
    output << "mean_plaquette_trace=" << observables.meanPlaquetteTrace << "\n";
    output << "mean_current=";
    printVector(output, observables.meanCurrent);
    output << "\n";
    output << "mean_color_field=";
    printVector(output, observables.meanColorField);
    output << "\n";
    return 0;
}

int runLaunchTarget(std::string_view id, std::ostream& output) {
    if (id == "sandbox") {
        return runSandbox(output);
    }

    if (id == "relativity-compare") {
        return runRelativityComparison(output);
    }

    if (id == "render-preview") {
        return runRenderPreview(output);
    }

    if (id == "weak-field-preview") {
        return runWeakFieldPreview(output);
    }

    if (id == "kerr-preview") {
        return runKerrPreview(output);
    }

    if (id == "kerr-geodesic-preview") {
        return runKerrGeodesicPreview(output);
    }

    if (id == "schwarzschild-preview") {
        return runSchwarzschildPreview(output);
    }

    if (id == "schwarzschild-geodesic-preview") {
        return runSchwarzschildGeodesicPreview(output);
    }

    if (id == "quantum-chemistry-preview") {
        return runQuantumChemistryPreview(output);
    }

    if (id == "quantum-chemistry-uhf-preview") {
        return runQuantumChemistryUhfPreview(output);
    }

    if (id == "quantum-chemistry-mp2-preview") {
        return runQuantumChemistryMp2Preview(output);
    }

    if (id == "quantum-scalar-preview") {
        return runQuantumScalarPreview(output);
    }

    if (id == "quantum-scalar-grid-preview") {
        return runQuantumScalarGridPreview(output);
    }

    if (id == "gauge-u1-preview") {
        return runGaugeFieldPreview(output);
    }

    if (id == "gauge-su2-preview") {
        return runGaugeSu2Preview(output);
    }

    if (id == "fermion-staggered-preview") {
        return runStaggeredFermionPreview(output);
    }

    if (id == "coupled-su2-fermion-preview") {
        return runCoupledSu2FermionPreview(output);
    }

    constexpr std::string_view objectPrefix = "object:";
    if (id.starts_with(objectPrefix)) {
        const std::string_view objectId = id.substr(objectPrefix.size());
        return runObjectProgram(objectId, defaultRegimeForObject(objectId), "velocity-verlet", output);
    }

    constexpr std::string_view pluginPrefix = "plugin:";
    if (id.starts_with(pluginPrefix)) {
        const std::string_view pluginId = id.substr(pluginPrefix.size());
        return runDiscoveredScenePreview(resolvePluginsRoot(), pluginId, output);
    }

    output << "Unknown launch target: " << id << "\n";
    return 1;
}

}  // namespace physicsmade::runtime


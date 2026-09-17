#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "physicsmade/physics/constraints.hpp"
#include "physicsmade/scene/object_state.hpp"
#include "physicsmade/simulation/simulation_world.hpp"

namespace physicsmade::io {

struct SceneActuatorPreset {
    std::string id;
    double angularStiffness{0.8};
    bool limitsEnabled{false};
    double lowerAngleLimit{-physicsmade::common::kPi};
    double upperAngleLimit{physicsmade::common::kPi};
    bool motorEnabled{false};
    double targetAngularSpeed{0.0};
    double maxMotorTorque{0.0};
};

struct SceneLinkDefinition {
    std::string id;
    physicsmade::scene::ObjectState objectTemplate{};
};

struct SceneLinkInstance {
    std::string id;
    std::string definitionId;
    math::Vector3 translation{};
    math::Quaternion orientation{};
    math::Vector3 linearVelocity{};
    math::Vector3 angularVelocity{};
    bool pinnedOverrideEnabled{false};
    bool pinnedOverride{false};
    double massScale{1.0};
    double radiusScale{1.0};
    double emissiveScale{1.0};
};

struct SceneArticulatedHinge {
    std::string id;
    std::string leftLinkId;
    std::string rightLinkId;
    math::Vector3 leftLocalAnchor{};
    math::Vector3 rightLocalAnchor{};
    math::Vector3 leftLocalAxis{0.0, 0.0, 1.0};
    math::Vector3 rightLocalAxis{0.0, 0.0, 1.0};
    math::Vector3 leftLocalReference{1.0, 0.0, 0.0};
    math::Vector3 rightLocalReference{1.0, 0.0, 0.0};
    double anchorStiffness{0.9};
    double anchorDamping{0.1};
    double angularStiffness{0.85};
    std::string actuatorPresetId{"-"};
};

struct ArticulatedLoopDiagnostic {
    std::string id;
    std::vector<std::string> linkIds{};
    std::vector<std::string> hingeIds{};
    double expectedClosureDistance{0.0};
    double tolerance{1.0e-4};
};

struct ArticulatedLoopDiagnosticResult {
    std::string id;
    double chainClosureDistance{0.0};
    double maxAnchorMismatch{0.0};
    bool withinTolerance{true};
    std::size_t linkCount{0};
    std::size_t hingeCount{0};
};

struct SceneManifest {
    std::string sceneName{"unnamed-scene"};
    std::string kinematicsId{"newtonian"};
    std::string integratorId{"velocity-verlet"};
    int previewSteps{0};
    double previewDtSeconds{0.0};
    std::vector<std::string> physicsSystems{};
    std::vector<SceneActuatorPreset> actuatorPresets{};
    std::vector<SceneLinkDefinition> linkDefinitions{};
    std::vector<SceneLinkInstance> linkInstances{};
    std::vector<SceneArticulatedHinge> articulatedHinges{};
    std::vector<ArticulatedLoopDiagnostic> loopDiagnostics{};
    std::vector<physicsmade::physics::DistanceConstraintSpec> distanceConstraints{};
    std::vector<physicsmade::physics::BallJointConstraintSpec> ballJointConstraints{};
    std::vector<physicsmade::physics::HingeConstraintSpec> hingeConstraints{};
    std::vector<physicsmade::scene::ObjectState> objects{};
};

std::string serializeScene(const SceneManifest& manifest);
SceneManifest deserializeScene(std::string_view text);

void writeSceneFile(const std::filesystem::path& path, const SceneManifest& manifest);
SceneManifest readSceneFile(const std::filesystem::path& path);

SceneManifest snapshotWorld(
    const simulation::SimulationWorld& world,
    std::string sceneName,
    std::string integratorId,
    std::vector<std::string> physicsSystems = {});

simulation::SimulationWorld instantiateScene(
    const SceneManifest& manifest,
    std::vector<std::string>* loadedSystems = nullptr);

std::vector<ArticulatedLoopDiagnosticResult> evaluateLoopDiagnostics(const SceneManifest& manifest);

}  // namespace physicsmade::io

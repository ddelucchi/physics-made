#include "physicsmade/io/scene_serialization.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/physics/electrostatic_system.hpp"
#include "physicsmade/physics/linear_drag_system.hpp"
#include "physicsmade/physics/lorentz_force_system.hpp"
#include "physicsmade/physics/newtonian_gravity_system.hpp"
#include "physicsmade/physics/rigid_body_dynamics.hpp"

namespace {

struct ArticulatedAssembly {
    std::vector<physicsmade::scene::ObjectState> objects{};
    std::unordered_map<std::string, std::size_t> indexById{};
};

void trimLineEnding(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

physicsmade::math::Vector3 parseVector3(
    const std::vector<std::string>& tokens,
    std::size_t startIndex) {
    return {
        std::stod(tokens[startIndex]),
        std::stod(tokens[startIndex + 1]),
        std::stod(tokens[startIndex + 2]),
    };
}

std::string join(const std::vector<std::string>& values, char delimiter) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            stream << delimiter;
        }
        stream << values[index];
    }
    return stream.str();
}

std::string serializeObjectFields(const physicsmade::scene::ObjectState& object) {
    std::ostringstream stream;
    stream << object.name << '|'
           << object.mass << '|'
           << object.charge << '|'
           << object.radius << '|'
           << object.pinned << '|'
           << object.position.x << '|'
           << object.position.y << '|'
           << object.position.z << '|'
           << object.velocity.x << '|'
           << object.velocity.y << '|'
           << object.velocity.z << '|'
           << object.coordinateTimeSeconds << '|'
           << object.properTimeSeconds << '|'
           << object.orientation.w << '|'
           << object.orientation.x << '|'
           << object.orientation.y << '|'
           << object.orientation.z << '|'
           << object.angularVelocity.x << '|'
           << object.angularVelocity.y << '|'
           << object.angularVelocity.z << '|'
           << object.bodyInertiaDiagonal.x << '|'
           << object.bodyInertiaDiagonal.y << '|'
           << object.bodyInertiaDiagonal.z << '|'
           << object.dragCoefficient << '|'
           << object.temperatureKelvin << '|'
           << object.emissiveIntensity << '|'
           << object.restitution << '|'
           << object.frictionCoefficient;
    return stream.str();
}

physicsmade::scene::ObjectState parseObjectTokens(
    const std::vector<std::string>& tokens,
    std::size_t startIndex) {
    const std::size_t remaining = tokens.size() - startIndex;
    if (remaining != 17 && remaining != 28) {
        throw std::runtime_error("invalid object record");
    }

    physicsmade::scene::ObjectState object{};
    object.name = tokens[startIndex + 0];
    object.mass = std::stod(tokens[startIndex + 1]);
    object.charge = std::stod(tokens[startIndex + 2]);
    object.radius = std::stod(tokens[startIndex + 3]);
    object.pinned = std::stoi(tokens[startIndex + 4]) != 0;
    object.position = {std::stod(tokens[startIndex + 5]), std::stod(tokens[startIndex + 6]), std::stod(tokens[startIndex + 7])};
    object.velocity = {std::stod(tokens[startIndex + 8]), std::stod(tokens[startIndex + 9]), std::stod(tokens[startIndex + 10])};
    object.coordinateTimeSeconds = std::stod(tokens[startIndex + 11]);
    object.properTimeSeconds = std::stod(tokens[startIndex + 12]);

    if (remaining == 28) {
        object.orientation = {
            std::stod(tokens[startIndex + 13]),
            std::stod(tokens[startIndex + 14]),
            std::stod(tokens[startIndex + 15]),
            std::stod(tokens[startIndex + 16]),
        };
        object.angularVelocity = parseVector3(tokens, startIndex + 17);
        object.bodyInertiaDiagonal = parseVector3(tokens, startIndex + 20);
        object.dragCoefficient = std::stod(tokens[startIndex + 23]);
        object.temperatureKelvin = std::stod(tokens[startIndex + 24]);
        object.emissiveIntensity = std::stod(tokens[startIndex + 25]);
        object.restitution = std::stod(tokens[startIndex + 26]);
        object.frictionCoefficient = std::stod(tokens[startIndex + 27]);
    } else {
        object.dragCoefficient = std::stod(tokens[startIndex + 13]);
        object.temperatureKelvin = std::stod(tokens[startIndex + 14]);
        object.emissiveIntensity = std::stod(tokens[startIndex + 15]);
        object.restitution = std::stod(tokens[startIndex + 16]);
    }

    return object;
}

std::unique_ptr<physicsmade::physics::StateIntegrator> makeIntegrator(const std::string& id) {
    if (id == "semi-implicit-euler") {
        return std::make_unique<physicsmade::physics::SemiImplicitEulerIntegrator>();
    }

    if (id == "runge-kutta-4") {
        return std::make_unique<physicsmade::physics::RungeKutta4Integrator>();
    }

    return std::make_unique<physicsmade::physics::VelocityVerletIntegrator>();
}

std::unique_ptr<physicsmade::physics::KinematicsModel> makeKinematics(const std::string& id) {
    if (id == "special-relativistic") {
        return std::make_unique<physicsmade::physics::SpecialRelativisticKinematics>();
    }

    return std::make_unique<physicsmade::physics::NewtonianKinematics>();
}

std::unique_ptr<physicsmade::physics::PhysicsSystem> makeSystem(const std::string& id) {
    if (id == "newtonian-gravity") {
        return std::make_unique<physicsmade::physics::NewtonianGravitySystem>();
    }

    if (id == "electrostatic") {
        return std::make_unique<physicsmade::physics::ElectrostaticSystem>();
    }

    if (id == "lorentz-force") {
        return std::make_unique<physicsmade::physics::LorentzForceSystem>();
    }

    if (id == "linear-drag") {
        return std::make_unique<physicsmade::physics::LinearDragSystem>();
    }

    return nullptr;
}

const physicsmade::io::SceneActuatorPreset* findActuatorPreset(
    const physicsmade::io::SceneManifest& manifest,
    const std::string& presetId) {
    if (presetId.empty() || presetId == "-") {
        return nullptr;
    }

    for (const auto& preset : manifest.actuatorPresets) {
        if (preset.id == presetId) {
            return &preset;
        }
    }

    throw std::runtime_error("unknown actuator preset: " + presetId);
}

physicsmade::scene::ObjectState instantiateLinkObject(
    const physicsmade::io::SceneLinkDefinition& definition,
    const physicsmade::io::SceneLinkInstance& instance) {
    auto object = definition.objectTemplate;
    const double safeMassScale = std::max(instance.massScale, 0.0);
    const double safeRadiusScale = std::max(instance.radiusScale, 0.0);
    const double safeEmissiveScale = std::max(instance.emissiveScale, 0.0);

    object.name = instance.id;
    object.position = instance.translation + instance.orientation.rotate(definition.objectTemplate.position);
    object.orientation = (instance.orientation * definition.objectTemplate.orientation).normalized();
    object.velocity = definition.objectTemplate.velocity + instance.linearVelocity;
    object.angularVelocity = definition.objectTemplate.angularVelocity + instance.angularVelocity;
    object.mass *= safeMassScale;
    object.radius *= safeRadiusScale;
    object.emissiveIntensity *= safeEmissiveScale;
    object.bodyInertiaDiagonal *= safeMassScale * safeRadiusScale * safeRadiusScale;
    if (instance.pinnedOverrideEnabled) {
        object.pinned = instance.pinnedOverride;
    }

    return object;
}

ArticulatedAssembly buildArticulatedAssembly(const physicsmade::io::SceneManifest& manifest) {
    ArticulatedAssembly assembly{};
    std::unordered_map<std::string, const physicsmade::io::SceneLinkDefinition*> definitions;
    for (const auto& definition : manifest.linkDefinitions) {
        definitions.emplace(definition.id, &definition);
    }

    for (const auto& instance : manifest.linkInstances) {
        const auto definitionIt = definitions.find(instance.definitionId);
        if (definitionIt == definitions.end()) {
            throw std::runtime_error("unknown link definition: " + instance.definitionId);
        }
        if (!assembly.indexById.emplace(instance.id, assembly.objects.size()).second) {
            throw std::runtime_error("duplicate articulated link instance id: " + instance.id);
        }

        assembly.objects.push_back(instantiateLinkObject(*definitionIt->second, instance));
    }

    return assembly;
}

physicsmade::physics::HingeConstraintSpec buildArticulatedHingeSpec(
    const physicsmade::io::SceneManifest& manifest,
    const physicsmade::io::SceneArticulatedHinge& hinge,
    const std::unordered_map<std::string, std::size_t>& articulatedIndexById,
    std::size_t articulatedBaseIndex) {
    const auto leftIt = articulatedIndexById.find(hinge.leftLinkId);
    const auto rightIt = articulatedIndexById.find(hinge.rightLinkId);
    if (leftIt == articulatedIndexById.end() || rightIt == articulatedIndexById.end()) {
        throw std::runtime_error("articulated hinge references unknown link instance");
    }

    physicsmade::physics::HingeConstraintSpec spec{
        {
            articulatedBaseIndex + leftIt->second,
            articulatedBaseIndex + rightIt->second,
            hinge.leftLocalAnchor,
            hinge.rightLocalAnchor,
            hinge.anchorStiffness,
            hinge.anchorDamping,
        },
        hinge.leftLocalAxis,
        hinge.rightLocalAxis,
    };
    spec.leftLocalReference = hinge.leftLocalReference;
    spec.rightLocalReference = hinge.rightLocalReference;
    spec.angularStiffness = hinge.angularStiffness;

    if (const auto* preset = findActuatorPreset(manifest, hinge.actuatorPresetId)) {
        spec.angularStiffness = preset->angularStiffness;
        spec.limitsEnabled = preset->limitsEnabled;
        spec.lowerAngleLimit = preset->lowerAngleLimit;
        spec.upperAngleLimit = preset->upperAngleLimit;
        spec.motorEnabled = preset->motorEnabled;
        spec.targetAngularSpeed = preset->targetAngularSpeed;
        spec.maxMotorTorque = preset->maxMotorTorque;
    }

    return spec;
}

}  // namespace

namespace physicsmade::io {

std::string serializeScene(const SceneManifest& manifest) {
    std::ostringstream stream;
    stream << "PHYSICSMADE_SCENE 1\n";
    stream << "scene_name=" << manifest.sceneName << "\n";
    stream << "kinematics=" << manifest.kinematicsId << "\n";
    stream << "integrator=" << manifest.integratorId << "\n";
    stream << "preview_steps=" << manifest.previewSteps << "\n";
    stream << "preview_dt_seconds=" << manifest.previewDtSeconds << "\n";
    stream << "physics_systems=" << join(manifest.physicsSystems, ',') << "\n";

    for (const auto& preset : manifest.actuatorPresets) {
        stream << "actuator_preset|"
               << preset.id << '|'
               << preset.angularStiffness << '|'
               << preset.limitsEnabled << '|'
               << preset.lowerAngleLimit << '|'
               << preset.upperAngleLimit << '|'
               << preset.motorEnabled << '|'
               << preset.targetAngularSpeed << '|'
               << preset.maxMotorTorque << "\n";
    }

    for (const auto& definition : manifest.linkDefinitions) {
        stream << "link_definition|" << definition.id << '|' << serializeObjectFields(definition.objectTemplate) << "\n";
    }

    for (const auto& instance : manifest.linkInstances) {
        stream << "link_instance|"
               << instance.id << '|'
               << instance.definitionId << '|'
               << instance.translation.x << '|'
               << instance.translation.y << '|'
               << instance.translation.z << '|'
               << instance.orientation.w << '|'
               << instance.orientation.x << '|'
               << instance.orientation.y << '|'
               << instance.orientation.z << '|'
               << instance.linearVelocity.x << '|'
               << instance.linearVelocity.y << '|'
               << instance.linearVelocity.z << '|'
               << instance.angularVelocity.x << '|'
               << instance.angularVelocity.y << '|'
               << instance.angularVelocity.z << '|'
               << instance.pinnedOverrideEnabled << '|'
               << instance.pinnedOverride << '|'
               << instance.massScale << '|'
               << instance.radiusScale << '|'
               << instance.emissiveScale << "\n";
    }

    for (const auto& hinge : manifest.articulatedHinges) {
        stream << "articulated_hinge|"
               << hinge.id << '|'
               << hinge.leftLinkId << '|'
               << hinge.rightLinkId << '|'
               << hinge.leftLocalAnchor.x << '|'
               << hinge.leftLocalAnchor.y << '|'
               << hinge.leftLocalAnchor.z << '|'
               << hinge.rightLocalAnchor.x << '|'
               << hinge.rightLocalAnchor.y << '|'
               << hinge.rightLocalAnchor.z << '|'
               << hinge.leftLocalAxis.x << '|'
               << hinge.leftLocalAxis.y << '|'
               << hinge.leftLocalAxis.z << '|'
               << hinge.rightLocalAxis.x << '|'
               << hinge.rightLocalAxis.y << '|'
               << hinge.rightLocalAxis.z << '|'
               << hinge.leftLocalReference.x << '|'
               << hinge.leftLocalReference.y << '|'
               << hinge.leftLocalReference.z << '|'
               << hinge.rightLocalReference.x << '|'
               << hinge.rightLocalReference.y << '|'
               << hinge.rightLocalReference.z << '|'
               << hinge.anchorStiffness << '|'
               << hinge.anchorDamping << '|'
               << hinge.angularStiffness << '|'
               << hinge.actuatorPresetId << "\n";
    }

    for (const auto& diagnostic : manifest.loopDiagnostics) {
        stream << "loop_diagnostic|"
               << diagnostic.id << '|'
               << diagnostic.expectedClosureDistance << '|'
               << diagnostic.tolerance << '|'
               << join(diagnostic.linkIds, ',') << '|'
               << join(diagnostic.hingeIds, ',') << "\n";
    }

    for (const auto& constraint : manifest.distanceConstraints) {
        stream << "distance_constraint|"
               << constraint.leftIndex << '|'
               << constraint.rightIndex << '|'
               << constraint.targetDistance << '|'
               << constraint.stiffness << "\n";
    }

    for (const auto& constraint : manifest.ballJointConstraints) {
        stream << "ball_joint|"
               << constraint.leftIndex << '|'
               << constraint.rightIndex << '|'
               << constraint.leftLocalAnchor.x << '|'
               << constraint.leftLocalAnchor.y << '|'
               << constraint.leftLocalAnchor.z << '|'
               << constraint.rightLocalAnchor.x << '|'
               << constraint.rightLocalAnchor.y << '|'
               << constraint.rightLocalAnchor.z << '|'
               << constraint.stiffness << '|'
               << constraint.damping << "\n";
    }

    for (const auto& constraint : manifest.hingeConstraints) {
        stream << "hinge_joint|"
               << constraint.anchor.leftIndex << '|'
               << constraint.anchor.rightIndex << '|'
               << constraint.anchor.leftLocalAnchor.x << '|'
               << constraint.anchor.leftLocalAnchor.y << '|'
               << constraint.anchor.leftLocalAnchor.z << '|'
               << constraint.anchor.rightLocalAnchor.x << '|'
               << constraint.anchor.rightLocalAnchor.y << '|'
               << constraint.anchor.rightLocalAnchor.z << '|'
               << constraint.leftLocalAxis.x << '|'
               << constraint.leftLocalAxis.y << '|'
               << constraint.leftLocalAxis.z << '|'
               << constraint.rightLocalAxis.x << '|'
               << constraint.rightLocalAxis.y << '|'
               << constraint.rightLocalAxis.z << '|'
               << constraint.anchor.stiffness << '|'
               << constraint.anchor.damping << '|'
               << constraint.angularStiffness << '|'
               << constraint.leftLocalReference.x << '|'
               << constraint.leftLocalReference.y << '|'
               << constraint.leftLocalReference.z << '|'
               << constraint.rightLocalReference.x << '|'
               << constraint.rightLocalReference.y << '|'
               << constraint.rightLocalReference.z << '|'
               << constraint.limitsEnabled << '|'
               << constraint.lowerAngleLimit << '|'
               << constraint.upperAngleLimit << '|'
               << constraint.motorEnabled << '|'
               << constraint.targetAngularSpeed << '|'
               << constraint.maxMotorTorque << "\n";
    }

    for (const auto& object : manifest.objects) {
        stream << "object|" << serializeObjectFields(object) << "\n";
    }

    return stream.str();
}

SceneManifest deserializeScene(std::string_view text) {
    SceneManifest manifest{};
    std::stringstream stream{std::string(text)};
    std::string line;
    bool headerSeen = false;

    while (std::getline(stream, line)) {
        trimLineEnding(line);
        if (line.empty()) {
            continue;
        }

        if (!headerSeen) {
            if (line != "PHYSICSMADE_SCENE 1") {
                throw std::runtime_error("invalid scene header");
            }
            headerSeen = true;
            continue;
        }

        if (line.rfind("scene_name=", 0) == 0) {
            manifest.sceneName = line.substr(11);
            continue;
        }

        if (line.rfind("kinematics=", 0) == 0) {
            manifest.kinematicsId = line.substr(11);
            continue;
        }

        if (line.rfind("integrator=", 0) == 0) {
            manifest.integratorId = line.substr(11);
            continue;
        }

        if (line.rfind("preview_steps=", 0) == 0) {
            manifest.previewSteps = std::stoi(line.substr(14));
            continue;
        }

        if (line.rfind("preview_dt_seconds=", 0) == 0) {
            manifest.previewDtSeconds = std::stod(line.substr(19));
            continue;
        }

        if (line.rfind("physics_systems=", 0) == 0) {
            const auto value = line.substr(16);
            manifest.physicsSystems = value.empty() ? std::vector<std::string>{} : split(value, ',');
            continue;
        }

        if (line.rfind("actuator_preset|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 9) {
                throw std::runtime_error("invalid actuator preset record");
            }

            manifest.actuatorPresets.push_back({
                tokens[1],
                std::stod(tokens[2]),
                std::stoi(tokens[3]) != 0,
                std::stod(tokens[4]),
                std::stod(tokens[5]),
                std::stoi(tokens[6]) != 0,
                std::stod(tokens[7]),
                std::stod(tokens[8]),
            });
            continue;
        }

        if (line.rfind("link_definition|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 19 && tokens.size() != 30) {
                throw std::runtime_error("invalid link definition record");
            }

            manifest.linkDefinitions.push_back({tokens[1], parseObjectTokens(tokens, 2)});
            continue;
        }

        if (line.rfind("link_instance|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 21) {
                throw std::runtime_error("invalid link instance record");
            }

            SceneLinkInstance instance{};
            instance.id = tokens[1];
            instance.definitionId = tokens[2];
            instance.translation = parseVector3(tokens, 3);
            instance.orientation = {
                std::stod(tokens[6]),
                std::stod(tokens[7]),
                std::stod(tokens[8]),
                std::stod(tokens[9]),
            };
            instance.linearVelocity = parseVector3(tokens, 10);
            instance.angularVelocity = parseVector3(tokens, 13);
            instance.pinnedOverrideEnabled = std::stoi(tokens[16]) != 0;
            instance.pinnedOverride = std::stoi(tokens[17]) != 0;
            instance.massScale = std::stod(tokens[18]);
            instance.radiusScale = std::stod(tokens[19]);
            instance.emissiveScale = std::stod(tokens[20]);
            manifest.linkInstances.push_back(instance);
            continue;
        }

        if (line.rfind("articulated_hinge|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 26) {
                throw std::runtime_error("invalid articulated hinge record");
            }

            SceneArticulatedHinge hinge{};
            hinge.id = tokens[1];
            hinge.leftLinkId = tokens[2];
            hinge.rightLinkId = tokens[3];
            hinge.leftLocalAnchor = parseVector3(tokens, 4);
            hinge.rightLocalAnchor = parseVector3(tokens, 7);
            hinge.leftLocalAxis = parseVector3(tokens, 10);
            hinge.rightLocalAxis = parseVector3(tokens, 13);
            hinge.leftLocalReference = parseVector3(tokens, 16);
            hinge.rightLocalReference = parseVector3(tokens, 19);
            hinge.anchorStiffness = std::stod(tokens[22]);
            hinge.anchorDamping = std::stod(tokens[23]);
            hinge.angularStiffness = std::stod(tokens[24]);
            hinge.actuatorPresetId = tokens[25];
            manifest.articulatedHinges.push_back(hinge);
            continue;
        }

        if (line.rfind("loop_diagnostic|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 6) {
                throw std::runtime_error("invalid loop diagnostic record");
            }

            ArticulatedLoopDiagnostic diagnostic{};
            diagnostic.id = tokens[1];
            diagnostic.expectedClosureDistance = std::stod(tokens[2]);
            diagnostic.tolerance = std::stod(tokens[3]);
            diagnostic.linkIds = tokens[4].empty() ? std::vector<std::string>{} : split(tokens[4], ',');
            diagnostic.hingeIds = tokens[5].empty() ? std::vector<std::string>{} : split(tokens[5], ',');
            manifest.loopDiagnostics.push_back(std::move(diagnostic));
            continue;
        }

        if (line.rfind("distance_constraint|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 5) {
                throw std::runtime_error("invalid distance constraint record");
            }

            manifest.distanceConstraints.push_back({
                static_cast<std::size_t>(std::stoull(tokens[1])),
                static_cast<std::size_t>(std::stoull(tokens[2])),
                std::stod(tokens[3]),
                std::stod(tokens[4]),
            });
            continue;
        }

        if (line.rfind("ball_joint|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 11) {
                throw std::runtime_error("invalid ball joint record");
            }

            manifest.ballJointConstraints.push_back({
                static_cast<std::size_t>(std::stoull(tokens[1])),
                static_cast<std::size_t>(std::stoull(tokens[2])),
                parseVector3(tokens, 3),
                parseVector3(tokens, 6),
                std::stod(tokens[9]),
                std::stod(tokens[10]),
            });
            continue;
        }

        if (line.rfind("hinge_joint|", 0) == 0) {
            const auto tokens = split(line, '|');
            if (tokens.size() != 18 && tokens.size() != 30) {
                throw std::runtime_error("invalid hinge joint record");
            }

            physicsmade::physics::HingeConstraintSpec spec{
                {
                    static_cast<std::size_t>(std::stoull(tokens[1])),
                    static_cast<std::size_t>(std::stoull(tokens[2])),
                    parseVector3(tokens, 3),
                    parseVector3(tokens, 6),
                    std::stod(tokens[15]),
                    std::stod(tokens[16]),
                },
                parseVector3(tokens, 9),
                parseVector3(tokens, 12),
            };
            spec.angularStiffness = std::stod(tokens[17]);

            if (tokens.size() == 30) {
                spec.leftLocalReference = parseVector3(tokens, 18);
                spec.rightLocalReference = parseVector3(tokens, 21);
                spec.limitsEnabled = std::stoi(tokens[24]) != 0;
                spec.lowerAngleLimit = std::stod(tokens[25]);
                spec.upperAngleLimit = std::stod(tokens[26]);
                spec.motorEnabled = std::stoi(tokens[27]) != 0;
                spec.targetAngularSpeed = std::stod(tokens[28]);
                spec.maxMotorTorque = std::stod(tokens[29]);
            }

            manifest.hingeConstraints.push_back(spec);
            continue;
        }

        if (line.rfind("object|", 0) == 0) {
            const auto tokens = split(line, '|');
            manifest.objects.push_back(parseObjectTokens(tokens, 1));
        }
    }

    if (!headerSeen) {
        throw std::runtime_error("scene header missing");
    }

    return manifest;
}

void writeSceneFile(const std::filesystem::path& path, const SceneManifest& manifest) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open scene file for writing");
    }

    stream << serializeScene(manifest);
}

SceneManifest readSceneFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open scene file for reading");
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return deserializeScene(buffer.str());
}

SceneManifest snapshotWorld(
    const simulation::SimulationWorld& world,
    std::string sceneName,
    std::string integratorId,
    std::vector<std::string> physicsSystems) {
    SceneManifest manifest{};
    manifest.sceneName = std::move(sceneName);
    manifest.kinematicsId = world.kinematicsModel().name();
    manifest.integratorId = std::move(integratorId);
    manifest.physicsSystems = std::move(physicsSystems);
    manifest.objects = world.objects();
    return manifest;
}

simulation::SimulationWorld instantiateScene(
    const SceneManifest& manifest,
    std::vector<std::string>* loadedSystems) {
    simulation::SimulationWorld world;
    world.setKinematicsModel(makeKinematics(manifest.kinematicsId));
    world.setIntegrator(makeIntegrator(manifest.integratorId));

    for (const auto& systemId : manifest.physicsSystems) {
        if (auto system = makeSystem(systemId)) {
            world.addPhysicsSystem(std::move(system));
            if (loadedSystems) {
                loadedSystems->push_back(systemId);
            }
        }
    }

    for (const auto& object : manifest.objects) {
        world.addObject(object);
    }

    const auto articulatedAssembly = buildArticulatedAssembly(manifest);
    const std::size_t articulatedBaseIndex = world.objects().size();
    for (const auto& object : articulatedAssembly.objects) {
        world.addObject(object);
    }

    for (const auto& constraint : manifest.distanceConstraints) {
        world.addConstraint(std::make_unique<physicsmade::physics::DistanceConstraint>(constraint));
    }

    for (const auto& constraint : manifest.ballJointConstraints) {
        world.addConstraint(std::make_unique<physicsmade::physics::BallJointConstraint>(constraint));
    }

    for (const auto& constraint : manifest.hingeConstraints) {
        world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(constraint));
    }

    for (const auto& hinge : manifest.articulatedHinges) {
        world.addConstraint(std::make_unique<physicsmade::physics::HingeConstraint>(
            buildArticulatedHingeSpec(manifest, hinge, articulatedAssembly.indexById, articulatedBaseIndex)));
    }

    return world;
}

std::vector<ArticulatedLoopDiagnosticResult> evaluateLoopDiagnostics(const SceneManifest& manifest) {
    std::vector<ArticulatedLoopDiagnosticResult> results;
    if (manifest.loopDiagnostics.empty()) {
        return results;
    }

    const auto articulatedAssembly = buildArticulatedAssembly(manifest);
    std::unordered_map<std::string, physicsmade::physics::HingeConstraintSpec> hingesById;
    for (const auto& hinge : manifest.articulatedHinges) {
        hingesById.emplace(hinge.id, buildArticulatedHingeSpec(manifest, hinge, articulatedAssembly.indexById, 0));
    }

    for (const auto& diagnostic : manifest.loopDiagnostics) {
        ArticulatedLoopDiagnosticResult result{};
        result.id = diagnostic.id;
        result.linkCount = diagnostic.linkIds.size();
        result.hingeCount = diagnostic.hingeIds.size();

        if (!diagnostic.hingeIds.empty()) {
            const auto firstIt = hingesById.find(diagnostic.hingeIds.front());
            const auto lastIt = hingesById.find(diagnostic.hingeIds.back());
            if (firstIt == hingesById.end() || lastIt == hingesById.end()) {
                throw std::runtime_error("loop diagnostic references unknown articulated hinge");
            }

            const auto firstLeftAnchor = physicsmade::physics::worldPoint(
                articulatedAssembly.objects[firstIt->second.anchor.leftIndex],
                firstIt->second.anchor.leftLocalAnchor);
            const auto lastRightAnchor = physicsmade::physics::worldPoint(
                articulatedAssembly.objects[lastIt->second.anchor.rightIndex],
                lastIt->second.anchor.rightLocalAnchor);
            result.chainClosureDistance = (lastRightAnchor - firstLeftAnchor).norm();

            for (const auto& hingeId : diagnostic.hingeIds) {
                const auto hingeIt = hingesById.find(hingeId);
                if (hingeIt == hingesById.end()) {
                    throw std::runtime_error("loop diagnostic references unknown articulated hinge");
                }

                const auto leftAnchor = physicsmade::physics::worldPoint(
                    articulatedAssembly.objects[hingeIt->second.anchor.leftIndex],
                    hingeIt->second.anchor.leftLocalAnchor);
                const auto rightAnchor = physicsmade::physics::worldPoint(
                    articulatedAssembly.objects[hingeIt->second.anchor.rightIndex],
                    hingeIt->second.anchor.rightLocalAnchor);
                result.maxAnchorMismatch = std::max(result.maxAnchorMismatch, (rightAnchor - leftAnchor).norm());
            }
        } else if (diagnostic.linkIds.size() >= 2) {
            const auto firstIt = articulatedAssembly.indexById.find(diagnostic.linkIds.front());
            const auto lastIt = articulatedAssembly.indexById.find(diagnostic.linkIds.back());
            if (firstIt == articulatedAssembly.indexById.end() || lastIt == articulatedAssembly.indexById.end()) {
                throw std::runtime_error("loop diagnostic references unknown articulated link");
            }

            result.chainClosureDistance =
                (articulatedAssembly.objects[lastIt->second].position - articulatedAssembly.objects[firstIt->second].position).norm();
        }

        result.withinTolerance =
            std::abs(result.chainClosureDistance - diagnostic.expectedClosureDistance) <= diagnostic.tolerance &&
            result.maxAnchorMismatch <= diagnostic.tolerance;
        results.push_back(result);
    }

    return results;
}

}  // namespace physicsmade::io

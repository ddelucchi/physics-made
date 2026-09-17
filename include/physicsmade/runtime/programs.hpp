#pragma once

#include <iosfwd>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "physicsmade/physics/kinematics_model.hpp"

namespace physicsmade::runtime {

struct ObjectProgramDescriptor {
    std::string id;
    std::string summary;
    physics::KinematicRegime defaultRegime{physics::KinematicRegime::NonRelativistic};
};

struct IntegratorDescriptor {
    std::string id;
    std::string summary;
};

struct LaunchTargetDescriptor {
    std::string id;
    std::string summary;
};

std::vector<ObjectProgramDescriptor> availableObjectPrograms();
std::vector<IntegratorDescriptor> availableIntegrators();
std::vector<LaunchTargetDescriptor> availableLaunchTargets();

int runObjectProgram(
    std::string_view id,
    physics::KinematicRegime regime,
    std::string_view integratorId,
    std::ostream& output);

inline int runObjectProgram(
    std::string_view id,
    physics::KinematicRegime regime,
    std::ostream& output) {
    return runObjectProgram(id, regime, "velocity-verlet", output);
}

int runSandbox(std::ostream& output);
int runRelativityComparison(std::ostream& output);
int runDiscoveredScenePreview(const std::filesystem::path& pluginsRoot, std::string_view pluginId, std::ostream& output);
int runKerrPreview(std::ostream& output);
int runKerrGeodesicPreview(std::ostream& output);
int runSchwarzschildPreview(std::ostream& output);
int runSchwarzschildGeodesicPreview(std::ostream& output);
int runQuantumChemistryPreview(std::ostream& output);
int runQuantumChemistryUhfPreview(std::ostream& output);
int runQuantumChemistryMp2Preview(std::ostream& output);
int runQuantumScalarPreview(std::ostream& output);
int runQuantumScalarGridPreview(std::ostream& output);
int runGaugeFieldPreview(std::ostream& output);
int runGaugeSu2Preview(std::ostream& output);
int runStaggeredFermionPreview(std::ostream& output);
int runCoupledSu2FermionPreview(std::ostream& output);
int runLaunchTarget(std::string_view id, std::ostream& output);

}  // namespace physicsmade::runtime

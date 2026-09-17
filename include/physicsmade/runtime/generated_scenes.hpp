#pragma once

#include <functional>
#include <optional>
#include <string_view>

#include "physicsmade/io/scene_serialization.hpp"
#include "physicsmade/simulation/simulation_world.hpp"

namespace physicsmade::runtime {

using GeneratedViewerSceneStepper = std::function<void(simulation::SimulationWorld&, double)>;

struct GeneratedViewerScene {
	io::SceneManifest manifest{};
	int suggestedFollowIndex{-1};
	GeneratedViewerSceneStepper step{};
};

std::optional<io::SceneManifest> buildGeneratedScene(std::string_view id);
std::optional<GeneratedViewerScene> buildGeneratedViewerScene(std::string_view id);

}  // namespace physicsmade::runtime
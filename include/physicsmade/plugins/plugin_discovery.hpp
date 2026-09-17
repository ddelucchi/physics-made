#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace physicsmade::plugins {

enum class PluginKind {
    Scene,
    ObjectProgram,
};

struct PluginManifest {
    std::string id;
    PluginKind kind{PluginKind::Scene};
    std::string summary;
    std::filesystem::path manifestPath;
    std::filesystem::path entryPath;
    std::string builtinSceneId;
};

std::vector<PluginManifest> discoverPlugins(const std::filesystem::path& pluginsRoot);
std::optional<PluginManifest> findPlugin(const std::filesystem::path& pluginsRoot, const std::string& id);
const char* toString(PluginKind kind) noexcept;

}  // namespace physicsmade::plugins

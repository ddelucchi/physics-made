#include "physicsmade/plugins/plugin_discovery.hpp"

#include <fstream>
#include <sstream>

namespace {

void trimLineEnding(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

std::optional<physicsmade::plugins::PluginManifest> parseManifest(const std::filesystem::path& manifestPath) {
    std::ifstream stream(manifestPath, std::ios::binary);
    if (!stream) {
        return std::nullopt;
    }

    physicsmade::plugins::PluginManifest manifest{};
    std::string line;
    bool headerSeen = false;

    while (std::getline(stream, line)) {
        trimLineEnding(line);
        if (line.empty()) {
            continue;
        }

        if (!headerSeen) {
            if (line != "PHYSICSMADE_PLUGIN 1") {
                return std::nullopt;
            }
            headerSeen = true;
            continue;
        }

        if (line.rfind("id=", 0) == 0) {
            manifest.id = line.substr(3);
            continue;
        }

        if (line.rfind("kind=", 0) == 0) {
            const auto value = line.substr(5);
            manifest.kind = (value == "object-program")
                                ? physicsmade::plugins::PluginKind::ObjectProgram
                                : physicsmade::plugins::PluginKind::Scene;
            continue;
        }

        if (line.rfind("summary=", 0) == 0) {
            manifest.summary = line.substr(8);
            continue;
        }

        if (line.rfind("entry=", 0) == 0) {
            const auto value = line.substr(6);
            if (value.rfind("builtin:", 0) == 0) {
                manifest.builtinSceneId = value.substr(8);
            } else {
                manifest.entryPath = manifestPath.parent_path() / value;
            }
        }
    }

    if (!headerSeen || manifest.id.empty() || (manifest.entryPath.empty() && manifest.builtinSceneId.empty())) {
        return std::nullopt;
    }

    manifest.manifestPath = manifestPath;
    return manifest;
}

}  // namespace

namespace physicsmade::plugins {

std::vector<PluginManifest> discoverPlugins(const std::filesystem::path& pluginsRoot) {
    std::vector<PluginManifest> manifests;
    if (!std::filesystem::exists(pluginsRoot)) {
        return manifests;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(pluginsRoot)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".pmplugin") {
            continue;
        }

        if (auto manifest = parseManifest(entry.path())) {
            manifests.push_back(std::move(*manifest));
        }
    }

    return manifests;
}

std::optional<PluginManifest> findPlugin(const std::filesystem::path& pluginsRoot, const std::string& id) {
    for (const auto& manifest : discoverPlugins(pluginsRoot)) {
        if (manifest.id == id) {
            return manifest;
        }
    }

    return std::nullopt;
}

const char* toString(PluginKind kind) noexcept {
    switch (kind) {
        case PluginKind::Scene:
            return "scene";
        case PluginKind::ObjectProgram:
            return "object-program";
    }

    return "unknown";
}

}  // namespace physicsmade::plugins

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "instanced_renderer.hpp"

#include "physicsmade/camera/orbit_camera.hpp"
#include "physicsmade/cuda/scene_buffers.hpp"
#include "physicsmade/io/scene_serialization.hpp"
#include "physicsmade/plugins/plugin_discovery.hpp"
#include "physicsmade/runtime/generated_scenes.hpp"

namespace {

using physicsmade::camera::OrbitCamera;
using physicsmade::math::Vector3;

struct ViewerConfig {
    std::optional<std::filesystem::path> scenePath{};
    std::string pluginId{};
    int followIndex{-1};
    int width{1280};
    int height{800};
};

struct ViewerState {
    physicsmade::simulation::SimulationWorld world{};
    std::unique_ptr<physicsmade::cuda::SceneBufferManager> sceneBuffers{};
    std::function<void(physicsmade::simulation::SimulationWorld&, double)> liveStep{};
    OrbitCamera camera{};
    std::string titleBase{"Physics Made Viewer"};
    double fixedDtSeconds{0.01};
    double displayTimeSeconds{0.0};
    int followIndex{-1};
    bool paused{false};
    bool dragging{false};
    bool pauseLatch{false};
    bool resetLatch{false};
    double lastCursorX{0.0};
    double lastCursorY{0.0};
    std::chrono::steady_clock::time_point lastTitleUpdate{};
};

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

std::string chooseDefaultPluginId(const std::filesystem::path& pluginsRoot) {
    if (physicsmade::plugins::findPlugin(pluginsRoot, "four-bar-loop")) {
        return "four-bar-loop";
    }

    if (physicsmade::plugins::findPlugin(pluginsRoot, "sample-orbit")) {
        return "sample-orbit";
    }

    const auto manifests = physicsmade::plugins::discoverPlugins(pluginsRoot);
    if (manifests.empty()) {
        throw std::runtime_error("no plugin scenes found under plugins/");
    }

    return manifests.front().id;
}

void printUsage() {
    std::cout
        << "Usage: physicsmade_viewer [--list] [--plugin <id>] [--scene <path>] [--follow <index>] [--width <pixels>] [--height <pixels>]\n"
        << "Controls: left-drag orbit, mouse wheel zoom, arrow keys orbit, W/S zoom, Space pause, R reset, Esc exit.\n";
}

void listPlugins() {
    const auto pluginsRoot = resolvePluginsRoot();
    std::cout << "Available plugins:\n";
    for (const auto& manifest : physicsmade::plugins::discoverPlugins(pluginsRoot)) {
        std::cout << "  " << manifest.id << " [" << physicsmade::plugins::toString(manifest.kind) << "]\n";
        std::cout << "    " << manifest.summary << "\n";
    }
}

ViewerConfig parseArguments(int argc, char** argv, bool& listOnly) {
    ViewerConfig config{};
    listOnly = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--list") {
            listOnly = true;
            continue;
        }

        if (argument == "--plugin" && (index + 1) < argc) {
            config.pluginId = argv[++index];
            continue;
        }

        if (argument == "--scene" && (index + 1) < argc) {
            config.scenePath = std::filesystem::path(argv[++index]);
            continue;
        }

        if (argument == "--follow" && (index + 1) < argc) {
            config.followIndex = std::stoi(argv[++index]);
            continue;
        }

        if (argument == "--width" && (index + 1) < argc) {
            config.width = std::stoi(argv[++index]);
            continue;
        }

        if (argument == "--height" && (index + 1) < argc) {
            config.height = std::stoi(argv[++index]);
            continue;
        }

        printUsage();
        throw std::runtime_error("invalid viewer argument");
    }

    return config;
}

struct LoadedViewerScene {
    physicsmade::io::SceneManifest manifest{};
    std::function<void(physicsmade::simulation::SimulationWorld&, double)> liveStep{};
    int suggestedFollowIndex{-1};
};

LoadedViewerScene loadViewerScene(const ViewerConfig& config, std::string& titleSuffix) {
    if (config.scenePath) {
        titleSuffix = config.scenePath->filename().string();
        return {physicsmade::io::readSceneFile(*config.scenePath), {}, -1};
    }

    const auto pluginsRoot = resolvePluginsRoot();
    const auto pluginId = config.pluginId.empty() ? chooseDefaultPluginId(pluginsRoot) : config.pluginId;
    const auto manifest = physicsmade::plugins::findPlugin(pluginsRoot, pluginId);
    if (!manifest) {
        throw std::runtime_error("unknown plugin scene: " + pluginId);
    }
    if (manifest->kind != physicsmade::plugins::PluginKind::Scene && manifest->kind != physicsmade::plugins::PluginKind::ObjectProgram) {
        throw std::runtime_error("unsupported plugin kind");
    }

    titleSuffix = manifest->id;
    if (!manifest->builtinSceneId.empty()) {
        if (const auto liveScene = physicsmade::runtime::buildGeneratedViewerScene(manifest->builtinSceneId)) {
            return {liveScene->manifest, liveScene->step, liveScene->suggestedFollowIndex};
        }

        const auto generatedScene = physicsmade::runtime::buildGeneratedScene(manifest->builtinSceneId);
        if (!generatedScene) {
            throw std::runtime_error("unknown generated plugin scene: " + manifest->builtinSceneId);
        }
        return {*generatedScene, {}, -1};
    }
    return {physicsmade::io::readSceneFile(manifest->entryPath), {}, -1};
}

Vector3 focusPoint(const ViewerState& state) {
    if (state.followIndex >= 0 && static_cast<std::size_t>(state.followIndex) < state.world.objects().size()) {
        return state.world.objects()[static_cast<std::size_t>(state.followIndex)].position;
    }

    return state.world.worldDiagnostics().centerOfMass;
}

double sceneExtent(const physicsmade::cuda::SceneInteropView& interop) {
    return physicsmade::cuda::renderSceneExtent(interop.boundsMin, interop.boundsMax);
}

void updateSceneBuffers(ViewerState& state) {
    state.sceneBuffers->reserve(state.world.objects().size());
    state.sceneBuffers->update(state.world.objects());
}

void updateWindowTitle(GLFWwindow* window, ViewerState& state) {
    const auto now = std::chrono::steady_clock::now();
    if (state.lastTitleUpdate.time_since_epoch().count() != 0 &&
        std::chrono::duration<double>(now - state.lastTitleUpdate).count() < 0.25) {
        return;
    }

    std::ostringstream title;
    title << state.titleBase << " | " << (state.paused ? "paused" : "running")
            << " | t=" << std::fixed << std::setprecision(3) << state.displayTimeSeconds << " s";
    glfwSetWindowTitle(window, title.str().c_str());
    state.lastTitleUpdate = now;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr || button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }

    state->dragging = (action == GLFW_PRESS);
    if (state->dragging) {
        glfwGetCursorPos(window, &state->lastCursorX, &state->lastCursorY);
    }
}

void cursorPositionCallback(GLFWwindow* window, double x, double y) {
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr || !state->dragging) {
        return;
    }

    const double deltaX = x - state->lastCursorX;
    const double deltaY = y - state->lastCursorY;
    state->camera.orbit(-0.004 * deltaX, -0.004 * deltaY);
    state->lastCursorX = x;
    state->lastCursorY = y;
}

void scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
    (void)xOffset;
    auto* state = static_cast<ViewerState*>(glfwGetWindowUserPointer(window));
    if (state == nullptr) {
        return;
    }

    state->camera.setRadius(state->camera.radius() * std::pow(0.9, yOffset));
}

void handleKeyboard(GLFWwindow* window, ViewerState& state, double frameSeconds, double defaultRadius) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    const double orbitRate = 1.6 * frameSeconds;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        state.camera.orbit(-orbitRate, 0.0);
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        state.camera.orbit(orbitRate, 0.0);
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        state.camera.orbit(0.0, -orbitRate);
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        state.camera.orbit(0.0, orbitRate);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        state.camera.setRadius(state.camera.radius() * std::pow(0.35, frameSeconds));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        state.camera.setRadius(state.camera.radius() * std::pow(1.0 / 0.35, frameSeconds));
    }

    const bool pausePressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (pausePressed && !state.pauseLatch) {
        state.paused = !state.paused;
    }
    state.pauseLatch = pausePressed;

    const bool resetPressed = glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS;
    if (resetPressed && !state.resetLatch) {
        state.camera = OrbitCamera(defaultRadius, 0.35, 0.35);
    }
    state.resetLatch = resetPressed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool listOnly = false;
        const auto config = parseArguments(argc, argv, listOnly);
        if (listOnly) {
            listPlugins();
            return 0;
        }

        std::string titleSuffix;
        const auto loadedScene = loadViewerScene(config, titleSuffix);
        ViewerState state{};
        state.world = physicsmade::io::instantiateScene(loadedScene.manifest);
        state.sceneBuffers = physicsmade::cuda::createSceneBufferManager();
        state.liveStep = loadedScene.liveStep;
        state.fixedDtSeconds = (loadedScene.manifest.previewDtSeconds > 0.0) ? loadedScene.manifest.previewDtSeconds : 0.01;
        state.followIndex = (config.followIndex >= 0) ? config.followIndex : loadedScene.suggestedFollowIndex;
        state.displayTimeSeconds = state.world.simulationTimeSeconds();
        state.titleBase = "Physics Made Viewer - " + titleSuffix;
        updateSceneBuffers(state);

        state.sceneBuffers->setViewerOrigin(focusPoint(state));
        const auto initialInterop = state.sceneBuffers->interopView();
        const double defaultRadius = std::max(3.0, 1.8 * sceneExtent(initialInterop));
        state.camera = OrbitCamera(defaultRadius, 0.35, 0.35);

        if (state.world.objects().empty()) {
            throw std::runtime_error("viewer scene contains no objects");
        }

        if (!glfwInit()) {
            throw std::runtime_error("failed to initialize GLFW");
        }

        struct GlfwTerminator {
            ~GlfwTerminator() {
                glfwTerminate();
            }
        } glfwTerminator;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_SAMPLES, 4);
        GLFWwindow* window = glfwCreateWindow(config.width, config.height, state.titleBase.c_str(), nullptr, nullptr);
        if (window == nullptr) {
            glfwDefaultWindowHints();
            glfwWindowHint(GLFW_SAMPLES, 4);
            window = glfwCreateWindow(config.width, config.height, state.titleBase.c_str(), nullptr, nullptr);
        }
        if (window == nullptr) {
            throw std::runtime_error("failed to create viewer window");
        }

        struct WindowDestroyer {
            GLFWwindow* handle;
            ~WindowDestroyer() {
                if (handle != nullptr) {
                    glfwDestroyWindow(handle);
                }
            }
        } windowDestroyer{window};

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(window, &state);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetScrollCallback(window, scrollCallback);

        physicsmade::viewer::InstancedSceneRenderer renderer(window);

        auto lastFrame = std::chrono::steady_clock::now();
        double accumulatorSeconds = 0.0;
        constexpr double wallStepSeconds = 1.0 / 60.0;

        while (!glfwWindowShouldClose(window)) {
            const auto now = std::chrono::steady_clock::now();
            double frameSeconds = std::chrono::duration<double>(now - lastFrame).count();
            lastFrame = now;
            frameSeconds = std::clamp(frameSeconds, 0.0, 0.1);
            accumulatorSeconds = std::min(accumulatorSeconds + frameSeconds, 4.0 * wallStepSeconds);

            handleKeyboard(window, state, frameSeconds, defaultRadius);

            if (!state.paused) {
                int substeps = 0;
                while (accumulatorSeconds >= wallStepSeconds && substeps < 4) {
                    if (state.liveStep) {
                        state.liveStep(state.world, state.fixedDtSeconds);
                        state.displayTimeSeconds += state.fixedDtSeconds;
                    } else {
                        state.world.step(state.fixedDtSeconds);
                        state.displayTimeSeconds = state.world.simulationTimeSeconds();
                    }
                    accumulatorSeconds -= wallStepSeconds;
                    ++substeps;
                }
                if (substeps > 0) {
                    updateSceneBuffers(state);
                }
            }

            const auto focus = focusPoint(state);
            state.sceneBuffers->setViewerOrigin(focus);
            const auto interop = state.sceneBuffers->interopView();
            if (!renderer.uploadInstances(interop, {})) {
                const auto cpuInstances = state.sceneBuffers->downloadGpuRenderInstances();
                if (!renderer.uploadInstances(interop, cpuInstances)) {
                    throw std::runtime_error("failed to upload viewer instances");
                }
            }

            const auto pose = state.camera.pose(focus);
            const auto relativeCamera = pose.position - interop.viewerOrigin;
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            renderer.render(relativeCamera, pose.up, interop, framebufferWidth, framebufferHeight);
            updateWindowTitle(window, state);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "viewer error: " << exception.what() << "\n";
        return 1;
    }
}
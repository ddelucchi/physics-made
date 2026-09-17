#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glext.h>

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#if defined(PHYSICSMADE_WITH_CUDA)
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#endif

#include "physicsmade/common/config.hpp"
#include "physicsmade/cuda/render_instances.hpp"
#include "physicsmade/cuda/scene_buffers.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::viewer {

namespace detail {

struct Mat4 {
    std::array<float, 16> value{};
};

inline Mat4 identityMatrix() noexcept {
    return {{{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    }}};
}

inline Mat4 multiply(const Mat4& left, const Mat4& right) noexcept {
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += left.value[static_cast<std::size_t>(row + (k * 4))] * right.value[static_cast<std::size_t>(k + (column * 4))];
            }
            result.value[static_cast<std::size_t>(row + (column * 4))] = sum;
        }
    }
    return result;
}

inline Mat4 perspectiveMatrix(float verticalFovRadians, float aspectRatio, float nearPlane, float farPlane) noexcept {
    const float tanHalfFov = std::tan(0.5f * verticalFovRadians);
    Mat4 result{};
    result.value[0] = 1.0f / (aspectRatio * tanHalfFov);
    result.value[5] = 1.0f / tanHalfFov;
    result.value[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
    result.value[11] = -1.0f;
    result.value[14] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    return result;
}

inline Mat4 lookAtMatrix(
    const physicsmade::math::Vector3& eye,
    const physicsmade::math::Vector3& target,
    const physicsmade::math::Vector3& up) noexcept {
    const auto forward = (target - eye).normalized();
    const auto right = forward.cross(up).normalized();
    const auto correctedUp = right.cross(forward).normalized();

    Mat4 result = identityMatrix();
    result.value[0] = static_cast<float>(right.x);
    result.value[1] = static_cast<float>(correctedUp.x);
    result.value[2] = static_cast<float>(-forward.x);
    result.value[4] = static_cast<float>(right.y);
    result.value[5] = static_cast<float>(correctedUp.y);
    result.value[6] = static_cast<float>(-forward.y);
    result.value[8] = static_cast<float>(right.z);
    result.value[9] = static_cast<float>(correctedUp.z);
    result.value[10] = static_cast<float>(-forward.z);
    result.value[12] = -static_cast<float>(right.dot(eye));
    result.value[13] = -static_cast<float>(correctedUp.dot(eye));
    result.value[14] = static_cast<float>(forward.dot(eye));
    return result;
}

template <typename ProcType>
ProcType loadProc(const char* name) {
    const auto proc = reinterpret_cast<ProcType>(glfwGetProcAddress(name));
    if (proc == nullptr) {
        throw std::runtime_error(std::string("missing OpenGL entry point: ") + name);
    }
    return proc;
}

struct GlApi {
    PFNGLGENVERTEXARRAYSPROC genVertexArrays{};
    PFNGLBINDVERTEXARRAYPROC bindVertexArray{};
    PFNGLDELETEVERTEXARRAYSPROC deleteVertexArrays{};
    PFNGLGENBUFFERSPROC genBuffers{};
    PFNGLBINDBUFFERPROC bindBuffer{};
    PFNGLBUFFERDATAPROC bufferData{};
    PFNGLBUFFERSUBDATAPROC bufferSubData{};
    PFNGLDELETEBUFFERSPROC deleteBuffers{};
    PFNGLENABLEVERTEXATTRIBARRAYPROC enableVertexAttribArray{};
    PFNGLVERTEXATTRIBPOINTERPROC vertexAttribPointer{};
    PFNGLVERTEXATTRIBDIVISORPROC vertexAttribDivisor{};
    PFNGLCREATESHADERPROC createShader{};
    PFNGLSHADERSOURCEPROC shaderSource{};
    PFNGLCOMPILESHADERPROC compileShader{};
    PFNGLGETSHADERIVPROC getShaderiv{};
    PFNGLGETSHADERINFOLOGPROC getShaderInfoLog{};
    PFNGLCREATEPROGRAMPROC createProgram{};
    PFNGLATTACHSHADERPROC attachShader{};
    PFNGLLINKPROGRAMPROC linkProgram{};
    PFNGLGETPROGRAMIVPROC getProgramiv{};
    PFNGLGETPROGRAMINFOLOGPROC getProgramInfoLog{};
    PFNGLUSEPROGRAMPROC useProgram{};
    PFNGLDELETESHADERPROC deleteShader{};
    PFNGLDELETEPROGRAMPROC deleteProgram{};
    PFNGLGETUNIFORMLOCATIONPROC getUniformLocation{};
    PFNGLUNIFORMMATRIX4FVPROC uniformMatrix4fv{};
    PFNGLUNIFORM3FPROC uniform3f{};
    PFNGLDRAWELEMENTSINSTANCEDPROC drawElementsInstanced{};
    PFNGLDRAWARRAYSINSTANCEDPROC drawArraysInstanced{};

    void load() {
        genVertexArrays = loadProc<PFNGLGENVERTEXARRAYSPROC>("glGenVertexArrays");
        bindVertexArray = loadProc<PFNGLBINDVERTEXARRAYPROC>("glBindVertexArray");
        deleteVertexArrays = loadProc<PFNGLDELETEVERTEXARRAYSPROC>("glDeleteVertexArrays");
        genBuffers = loadProc<PFNGLGENBUFFERSPROC>("glGenBuffers");
        bindBuffer = loadProc<PFNGLBINDBUFFERPROC>("glBindBuffer");
        bufferData = loadProc<PFNGLBUFFERDATAPROC>("glBufferData");
        bufferSubData = loadProc<PFNGLBUFFERSUBDATAPROC>("glBufferSubData");
        deleteBuffers = loadProc<PFNGLDELETEBUFFERSPROC>("glDeleteBuffers");
        enableVertexAttribArray = loadProc<PFNGLENABLEVERTEXATTRIBARRAYPROC>("glEnableVertexAttribArray");
        vertexAttribPointer = loadProc<PFNGLVERTEXATTRIBPOINTERPROC>("glVertexAttribPointer");
        vertexAttribDivisor = loadProc<PFNGLVERTEXATTRIBDIVISORPROC>("glVertexAttribDivisor");
        createShader = loadProc<PFNGLCREATESHADERPROC>("glCreateShader");
        shaderSource = loadProc<PFNGLSHADERSOURCEPROC>("glShaderSource");
        compileShader = loadProc<PFNGLCOMPILESHADERPROC>("glCompileShader");
        getShaderiv = loadProc<PFNGLGETSHADERIVPROC>("glGetShaderiv");
        getShaderInfoLog = loadProc<PFNGLGETSHADERINFOLOGPROC>("glGetShaderInfoLog");
        createProgram = loadProc<PFNGLCREATEPROGRAMPROC>("glCreateProgram");
        attachShader = loadProc<PFNGLATTACHSHADERPROC>("glAttachShader");
        linkProgram = loadProc<PFNGLLINKPROGRAMPROC>("glLinkProgram");
        getProgramiv = loadProc<PFNGLGETPROGRAMIVPROC>("glGetProgramiv");
        getProgramInfoLog = loadProc<PFNGLGETPROGRAMINFOLOGPROC>("glGetProgramInfoLog");
        useProgram = loadProc<PFNGLUSEPROGRAMPROC>("glUseProgram");
        deleteShader = loadProc<PFNGLDELETESHADERPROC>("glDeleteShader");
        deleteProgram = loadProc<PFNGLDELETEPROGRAMPROC>("glDeleteProgram");
        getUniformLocation = loadProc<PFNGLGETUNIFORMLOCATIONPROC>("glGetUniformLocation");
        uniformMatrix4fv = loadProc<PFNGLUNIFORMMATRIX4FVPROC>("glUniformMatrix4fv");
        uniform3f = loadProc<PFNGLUNIFORM3FPROC>("glUniform3f");
        drawElementsInstanced = loadProc<PFNGLDRAWELEMENTSINSTANCEDPROC>("glDrawElementsInstanced");
        drawArraysInstanced = loadProc<PFNGLDRAWARRAYSINSTANCEDPROC>("glDrawArraysInstanced");
    }
};

struct SphereVertex {
    float position[3];
    float normal[3];
};

struct AxisVertex {
    float position[3];
    float color[3];
};

inline GLuint compileShader(const GlApi& gl, GLenum type, const char* source) {
    const GLuint shader = gl.createShader(type);
    gl.shaderSource(shader, 1, &source, nullptr);
    gl.compileShader(shader);

    GLint status = GL_FALSE;
    gl.getShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_TRUE) {
        return shader;
    }

    GLint length = 0;
    gl.getShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    gl.getShaderInfoLog(shader, length, nullptr, log.data());
    gl.deleteShader(shader);
    throw std::runtime_error("shader compilation failed: " + log);
}

inline GLuint createProgram(const GlApi& gl, const char* vertexSource, const char* fragmentSource) {
    const GLuint vertexShader = compileShader(gl, GL_VERTEX_SHADER, vertexSource);
    const GLuint fragmentShader = compileShader(gl, GL_FRAGMENT_SHADER, fragmentSource);
    const GLuint program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    GLint status = GL_FALSE;
    gl.getProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_TRUE) {
        return program;
    }

    GLint length = 0;
    gl.getProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
    gl.getProgramInfoLog(program, length, nullptr, log.data());
    gl.deleteProgram(program);
    throw std::runtime_error("shader link failed: " + log);
}

inline void appendSphereMesh(
    std::vector<SphereVertex>& vertices,
    std::vector<unsigned int>& indices,
    int latitudeBands,
    int longitudeBands) {
    for (int latitude = 0; latitude <= latitudeBands; ++latitude) {
        const float theta = static_cast<float>(physicsmade::common::kPi * static_cast<double>(latitude) / static_cast<double>(latitudeBands));
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);

        for (int longitude = 0; longitude <= longitudeBands; ++longitude) {
            const float phi = static_cast<float>((2.0 * physicsmade::common::kPi) * static_cast<double>(longitude) / static_cast<double>(longitudeBands));
            const float sinPhi = std::sin(phi);
            const float cosPhi = std::cos(phi);
            const float x = sinTheta * cosPhi;
            const float y = cosTheta;
            const float z = sinTheta * sinPhi;
            vertices.push_back({{x, y, z}, {x, y, z}});
        }
    }

    for (int latitude = 0; latitude < latitudeBands; ++latitude) {
        for (int longitude = 0; longitude < longitudeBands; ++longitude) {
            const unsigned int first = static_cast<unsigned int>((latitude * (longitudeBands + 1)) + longitude);
            const unsigned int second = first + static_cast<unsigned int>(longitudeBands) + 1U;
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1U);
            indices.push_back(second);
            indices.push_back(second + 1U);
            indices.push_back(first + 1U);
        }
    }
}

class CudaInteropBridge {
  public:
    ~CudaInteropBridge() {
        reset();
    }

    bool upload(GLuint buffer, std::size_t capacityBytes, std::uintptr_t sourceDeviceAddress, std::size_t bytes) {
#if defined(PHYSICSMADE_WITH_CUDA)
        if (bytes == 0) {
            return true;
        }

        if (!ensureRegistered(buffer, capacityBytes)) {
            return false;
        }

        cudaError_t status = cudaGraphicsMapResources(1, &resource_);
        if (status != cudaSuccess) {
            reset();
            return false;
        }

        void* mappedPointer = nullptr;
        std::size_t mappedBytes = 0;
        status = cudaGraphicsResourceGetMappedPointer(&mappedPointer, &mappedBytes, resource_);
        if (status != cudaSuccess || mappedBytes < bytes) {
            cudaGraphicsUnmapResources(1, &resource_);
            reset();
            return false;
        }

        status = cudaMemcpy(mappedPointer, reinterpret_cast<const void*>(sourceDeviceAddress), bytes, cudaMemcpyDeviceToDevice);
        const cudaError_t unmapStatus = cudaGraphicsUnmapResources(1, &resource_);
        if (status != cudaSuccess || unmapStatus != cudaSuccess) {
            reset();
            return false;
        }

        return true;
#else
        (void)buffer;
        (void)capacityBytes;
        (void)sourceDeviceAddress;
        (void)bytes;
        return false;
#endif
    }

  private:
    bool ensureRegistered(GLuint buffer, std::size_t capacityBytes) {
#if defined(PHYSICSMADE_WITH_CUDA)
        if (resource_ != nullptr && registeredBuffer_ == buffer && registeredCapacityBytes_ == capacityBytes) {
            return true;
        }

        reset();
        const auto status = cudaGraphicsGLRegisterBuffer(&resource_, buffer, cudaGraphicsRegisterFlagsWriteDiscard);
        if (status != cudaSuccess) {
            resource_ = nullptr;
            return false;
        }

        registeredBuffer_ = buffer;
        registeredCapacityBytes_ = capacityBytes;
        return true;
#else
        (void)buffer;
        (void)capacityBytes;
        return false;
#endif
    }

    void reset() {
#if defined(PHYSICSMADE_WITH_CUDA)
        if (resource_ != nullptr) {
            cudaGraphicsUnregisterResource(resource_);
            resource_ = nullptr;
        }
#endif
        registeredBuffer_ = 0;
        registeredCapacityBytes_ = 0;
    }

#if defined(PHYSICSMADE_WITH_CUDA)
    cudaGraphicsResource* resource_{nullptr};
#endif
    GLuint registeredBuffer_{0};
    std::size_t registeredCapacityBytes_{0};
};

}  // namespace detail

class InstancedSceneRenderer {
  public:
    explicit InstancedSceneRenderer(GLFWwindow* window) {
        if (window == nullptr) {
            throw std::runtime_error("renderer requires a valid GLFW window");
        }

        (void)window;
        gl_.load();
        createPrograms();
        createBuffers();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glLineWidth(1.5f);
    }

    ~InstancedSceneRenderer() {
        if (sphereProgram_ != 0U) {
            gl_.deleteProgram(sphereProgram_);
        }
        if (axisProgram_ != 0U) {
            gl_.deleteProgram(axisProgram_);
        }
        if (sphereEbo_ != 0U) {
            gl_.deleteBuffers(1, &sphereEbo_);
        }
        if (sphereVbo_ != 0U) {
            gl_.deleteBuffers(1, &sphereVbo_);
        }
        if (axisVbo_ != 0U) {
            gl_.deleteBuffers(1, &axisVbo_);
        }
        if (instanceVbo_ != 0U) {
            gl_.deleteBuffers(1, &instanceVbo_);
        }
        if (sphereVao_ != 0U) {
            gl_.deleteVertexArrays(1, &sphereVao_);
        }
        if (axisVao_ != 0U) {
            gl_.deleteVertexArrays(1, &axisVao_);
        }
    }

    bool uploadInstances(
        const physicsmade::cuda::SceneInteropView& interop,
        const std::vector<physicsmade::cuda::GpuRenderInstance>& cpuInstances) {
        const std::size_t instanceCount = interop.viewerInstanceBuffer.count;
        ensureInstanceCapacity(std::max<std::size_t>(instanceCount, 1));

        const std::size_t byteCount = instanceCount * sizeof(physicsmade::cuda::GpuRenderInstance);
        if (interop.deviceResident && interop.viewerInstanceBuffer.deviceAddress != 0U) {
            if (interopBridge_.upload(instanceVbo_, instanceCapacity_ * sizeof(physicsmade::cuda::GpuRenderInstance), interop.viewerInstanceBuffer.deviceAddress, byteCount)) {
                return true;
            }
        }

        if (instanceCount == 0) {
            return true;
        }

        if (cpuInstances.size() < instanceCount) {
            return false;
        }

        gl_.bindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
        gl_.bufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(byteCount), cpuInstances.data());
        return true;
    }

    void render(
        const physicsmade::math::Vector3& relativeCameraPosition,
        const physicsmade::math::Vector3& up,
        const physicsmade::cuda::SceneInteropView& interop,
        int framebufferWidth,
        int framebufferHeight) {
        const float aspectRatio = framebufferHeight > 0 ? static_cast<float>(framebufferWidth) / static_cast<float>(framebufferHeight) : 1.0f;
        const double extent = physicsmade::cuda::renderSceneExtent(interop.boundsMin, interop.boundsMax);
        const double cameraRadius = relativeCameraPosition.norm();
        const float nearPlane = static_cast<float>(std::max(0.01, 0.0005 * std::max(cameraRadius, extent)));
        const float farPlane = static_cast<float>(std::max(50.0, cameraRadius + (8.0 * extent)));

        const auto projection = detail::perspectiveMatrix(static_cast<float>(45.0 * physicsmade::common::kPi / 180.0), aspectRatio, nearPlane, farPlane);
        const auto view = detail::lookAtMatrix(relativeCameraPosition, {0.0, 0.0, 0.0}, up);
        const auto viewProjection = detail::multiply(projection, view);
        const auto lightDirection = physicsmade::math::Vector3{0.8 * extent, 1.2 * extent, 1.5 * extent}.normalized();

        glViewport(0, 0, framebufferWidth, framebufferHeight);
        glClearColor(0.04f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        gl_.useProgram(sphereProgram_);
        gl_.uniformMatrix4fv(sphereViewProjectionLocation_, 1, GL_FALSE, viewProjection.value.data());
        gl_.uniform3f(sphereLightDirectionLocation_, static_cast<float>(lightDirection.x), static_cast<float>(lightDirection.y), static_cast<float>(lightDirection.z));
        gl_.bindVertexArray(sphereVao_);
        gl_.drawElementsInstanced(GL_TRIANGLES, sphereIndexCount_, GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(interop.viewerInstanceBuffer.count));

        gl_.useProgram(axisProgram_);
        gl_.uniformMatrix4fv(axisViewProjectionLocation_, 1, GL_FALSE, viewProjection.value.data());
        gl_.bindVertexArray(axisVao_);
        gl_.drawArraysInstanced(GL_LINES, 0, axisVertexCount_, static_cast<GLsizei>(interop.viewerInstanceBuffer.count));

        renderFocusAxes(viewProjection, extent);
    }

  private:
    void createPrograms() {
        static constexpr const char* kSphereVertexShader = R"GLSL(
            #version 330 core
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aNormal;
            layout(location = 2) in vec4 iPositionRadius;
            layout(location = 3) in vec4 iOrientation;
            layout(location = 4) in vec4 iColor;
            layout(location = 5) in vec4 iDynamics;

            uniform mat4 uViewProjection;

            out vec3 vNormal;
            out vec3 vColor;
            out float vEmissive;

            vec3 quatRotate(vec4 q, vec3 v) {
                return v + (2.0 * cross(q.xyz, cross(q.xyz, v) + (q.w * v)));
            }

            void main() {
                vec3 worldPosition = quatRotate(iOrientation, aPosition * iPositionRadius.w) + iPositionRadius.xyz;
                vNormal = normalize(quatRotate(iOrientation, aNormal));
                vColor = iColor.rgb;
                vEmissive = iDynamics.w;
                gl_Position = uViewProjection * vec4(worldPosition, 1.0);
            }
        )GLSL";

        static constexpr const char* kSphereFragmentShader = R"GLSL(
            #version 330 core
            uniform vec3 uLightDirection;
            in vec3 vNormal;
            in vec3 vColor;
            in float vEmissive;
            out vec4 fragColor;

            void main() {
                float diffuse = max(dot(normalize(vNormal), normalize(uLightDirection)), 0.0);
                float lighting = 0.22 + (0.78 * diffuse);
                float emissive = clamp(vEmissive, 0.0, 8.0);
                vec3 color = clamp(vColor * (lighting + (0.18 * emissive)), 0.0, 1.2);
                fragColor = vec4(color, 1.0);
            }
        )GLSL";

        static constexpr const char* kAxisVertexShader = R"GLSL(
            #version 330 core
            layout(location = 0) in vec3 aPosition;
            layout(location = 1) in vec3 aColor;
            layout(location = 2) in vec4 iPositionRadius;
            layout(location = 3) in vec4 iOrientation;
            layout(location = 4) in vec4 iColor;
            layout(location = 5) in vec4 iDynamics;

            uniform mat4 uViewProjection;
            out vec3 vColor;

            vec3 quatRotate(vec4 q, vec3 v) {
                return v + (2.0 * cross(q.xyz, cross(q.xyz, v) + (q.w * v)));
            }

            void main() {
                vec3 worldPosition = quatRotate(iOrientation, aPosition * iDynamics.x) + iPositionRadius.xyz;
                vColor = aColor;
                gl_Position = uViewProjection * vec4(worldPosition, 1.0);
            }
        )GLSL";

        static constexpr const char* kAxisFragmentShader = R"GLSL(
            #version 330 core
            in vec3 vColor;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(vColor, 1.0);
            }
        )GLSL";

        sphereProgram_ = detail::createProgram(gl_, kSphereVertexShader, kSphereFragmentShader);
        axisProgram_ = detail::createProgram(gl_, kAxisVertexShader, kAxisFragmentShader);
        sphereViewProjectionLocation_ = gl_.getUniformLocation(sphereProgram_, "uViewProjection");
        sphereLightDirectionLocation_ = gl_.getUniformLocation(sphereProgram_, "uLightDirection");
        axisViewProjectionLocation_ = gl_.getUniformLocation(axisProgram_, "uViewProjection");
    }

    void createBuffers() {
        std::vector<detail::SphereVertex> sphereVertices;
        std::vector<unsigned int> sphereIndices;
        detail::appendSphereMesh(sphereVertices, sphereIndices, 18, 24);
        sphereIndexCount_ = static_cast<GLsizei>(sphereIndices.size());

        static constexpr detail::AxisVertex kAxisVertices[]{
            {{0.0f, 0.0f, 0.0f}, {0.95f, 0.30f, 0.25f}},
            {{1.0f, 0.0f, 0.0f}, {0.95f, 0.30f, 0.25f}},
            {{0.0f, 0.0f, 0.0f}, {0.20f, 0.85f, 0.35f}},
            {{0.0f, 1.0f, 0.0f}, {0.20f, 0.85f, 0.35f}},
            {{0.0f, 0.0f, 0.0f}, {0.30f, 0.55f, 0.95f}},
            {{0.0f, 0.0f, 1.0f}, {0.30f, 0.55f, 0.95f}},
        };
        axisVertexCount_ = static_cast<GLsizei>(std::size(kAxisVertices));

        gl_.genVertexArrays(1, &sphereVao_);
        gl_.genVertexArrays(1, &axisVao_);
        gl_.genBuffers(1, &sphereVbo_);
        gl_.genBuffers(1, &sphereEbo_);
        gl_.genBuffers(1, &axisVbo_);
        gl_.genBuffers(1, &instanceVbo_);

        gl_.bindVertexArray(sphereVao_);
        gl_.bindBuffer(GL_ARRAY_BUFFER, sphereVbo_);
        gl_.bufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sphereVertices.size() * sizeof(detail::SphereVertex)), sphereVertices.data(), GL_STATIC_DRAW);
        gl_.bindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEbo_);
        gl_.bufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(sphereIndices.size() * sizeof(unsigned int)), sphereIndices.data(), GL_STATIC_DRAW);
        gl_.enableVertexAttribArray(0);
        gl_.vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(detail::SphereVertex), reinterpret_cast<const void*>(offsetof(detail::SphereVertex, position)));
        gl_.enableVertexAttribArray(1);
        gl_.vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(detail::SphereVertex), reinterpret_cast<const void*>(offsetof(detail::SphereVertex, normal)));

        gl_.bindVertexArray(axisVao_);
        gl_.bindBuffer(GL_ARRAY_BUFFER, axisVbo_);
        gl_.bufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(kAxisVertices)), kAxisVertices, GL_STATIC_DRAW);
        gl_.enableVertexAttribArray(0);
        gl_.vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(detail::AxisVertex), reinterpret_cast<const void*>(offsetof(detail::AxisVertex, position)));
        gl_.enableVertexAttribArray(1);
        gl_.vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(detail::AxisVertex), reinterpret_cast<const void*>(offsetof(detail::AxisVertex, color)));

        ensureInstanceCapacity(64);
        configureInstanceAttributes(sphereVao_);
        configureInstanceAttributes(axisVao_);
        gl_.bindVertexArray(0);
    }

    void configureInstanceAttributes(GLuint vao) {
        gl_.bindVertexArray(vao);
        gl_.bindBuffer(GL_ARRAY_BUFFER, instanceVbo_);

        gl_.enableVertexAttribArray(2);
        gl_.vertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(physicsmade::cuda::GpuRenderInstance), reinterpret_cast<const void*>(offsetof(physicsmade::cuda::GpuRenderInstance, positionRadius)));
        gl_.vertexAttribDivisor(2, 1);
        gl_.enableVertexAttribArray(3);
        gl_.vertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(physicsmade::cuda::GpuRenderInstance), reinterpret_cast<const void*>(offsetof(physicsmade::cuda::GpuRenderInstance, orientation)));
        gl_.vertexAttribDivisor(3, 1);
        gl_.enableVertexAttribArray(4);
        gl_.vertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(physicsmade::cuda::GpuRenderInstance), reinterpret_cast<const void*>(offsetof(physicsmade::cuda::GpuRenderInstance, color)));
        gl_.vertexAttribDivisor(4, 1);
        gl_.enableVertexAttribArray(5);
        gl_.vertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(physicsmade::cuda::GpuRenderInstance), reinterpret_cast<const void*>(offsetof(physicsmade::cuda::GpuRenderInstance, dynamics)));
        gl_.vertexAttribDivisor(5, 1);
    }

    void ensureInstanceCapacity(std::size_t minimumCapacity) {
        if (minimumCapacity <= instanceCapacity_) {
            return;
        }

        instanceCapacity_ = std::max(instanceCapacity_ * 2, minimumCapacity);
        gl_.bindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
        gl_.bufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(instanceCapacity_ * sizeof(physicsmade::cuda::GpuRenderInstance)),
            nullptr,
            GL_DYNAMIC_DRAW);
    }

    void renderFocusAxes(const detail::Mat4& viewProjection, double extent) {
        const float axisLength = static_cast<float>(std::max(0.3, 0.15 * extent));
        physicsmade::cuda::GpuRenderInstance focusInstance{};
        focusInstance.orientation[3] = 1.0f;
        focusInstance.color[3] = 1.0f;
        focusInstance.dynamics[0] = axisLength;

        glDisable(GL_DEPTH_TEST);
        gl_.bindBuffer(GL_ARRAY_BUFFER, instanceVbo_);
        gl_.bufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(focusInstance)), &focusInstance);
        gl_.useProgram(axisProgram_);
        gl_.uniformMatrix4fv(axisViewProjectionLocation_, 1, GL_FALSE, viewProjection.value.data());
        gl_.bindVertexArray(axisVao_);
        gl_.drawArraysInstanced(GL_LINES, 0, axisVertexCount_, 1);
        glEnable(GL_DEPTH_TEST);
    }

    detail::GlApi gl_{};
    detail::CudaInteropBridge interopBridge_{};
    GLuint sphereProgram_{0};
    GLuint axisProgram_{0};
    GLuint sphereVao_{0};
    GLuint sphereVbo_{0};
    GLuint sphereEbo_{0};
    GLuint axisVao_{0};
    GLuint axisVbo_{0};
    GLuint instanceVbo_{0};
    GLint sphereViewProjectionLocation_{-1};
    GLint sphereLightDirectionLocation_{-1};
    GLint axisViewProjectionLocation_{-1};
    GLsizei sphereIndexCount_{0};
    GLsizei axisVertexCount_{0};
    std::size_t instanceCapacity_{0};
};

}  // namespace physicsmade::viewer
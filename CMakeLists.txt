cmake_minimum_required(VERSION 3.25)

project(
    PhysicsMade
    VERSION 0.1.0
    DESCRIPTION "Extensible spacetime physics framework in C++ with CUDA acceleration"
    LANGUAGES CXX
)

option(PHYSICSMADE_ENABLE_CUDA "Enable CUDA field solvers when a CUDA toolchain is available" ON)
option(PHYSICSMADE_BUILD_APPS "Build example applications" ON)
option(PHYSICSMADE_BUILD_TESTS "Build tests" ON)
option(PHYSICSMADE_BUILD_VIEWER "Build the GLFW/OpenGL live viewer" ON)

include(CTest)
include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PHYSICSMADE_CUDA_ENABLED OFF)
if(PHYSICSMADE_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(PHYSICSMADE_CUDA_ENABLED ON)
        set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
        endif()
    endif()
endif()

add_library(physicsmade)
add_library(physicsmade::physicsmade ALIAS physicsmade)

target_sources(
    physicsmade
    PRIVATE
        src/spacetime/metric_tensor.cpp
        src/spacetime/geodesic_integrator.cpp
        src/spacetime/kerr_metric.cpp
        src/spacetime/schwarzschild_metric.cpp
        src/spacetime/weak_field_metric.cpp
        src/physics/kinematics_model.cpp
        src/physics/integrators.cpp
        src/physics/rigid_body_dynamics.cpp
        src/physics/constraints.cpp
        src/physics/newtonian_gravity_system.cpp
        src/physics/electrostatic_system.cpp
        src/physics/lorentz_force_system.cpp
        src/physics/linear_drag_system.cpp
        src/physics/hookean_spring_system.cpp
        src/simulation/simulation_world.cpp
        src/camera/orbit_camera.cpp
        src/io/scene_serialization.cpp
        src/plugins/plugin_discovery.cpp
        src/field_theory/scalar_field_lattice.cpp
        src/field_theory/rectangular_scalar_field_lattice.cpp
        src/field_theory/u1_gauge_field_lattice.cpp
        src/field_theory/su2_gauge_field_lattice.cpp
        src/field_theory/staggered_fermion_lattice.cpp
        src/field_theory/coupled_su2_fermion_lattice.cpp
        src/quantum_chemistry/restricted_hartree_fock.cpp
        src/runtime/generated_scenes.cpp
        src/runtime/programs.cpp
)

if(PHYSICSMADE_CUDA_ENABLED)
    target_sources(
        physicsmade
        PRIVATE
            src/cuda/gravity_field_kernels.cu
            src/cuda/gravity_body_forces.cu
            src/cuda/render_instances.cu
            src/cuda/scene_buffers.cu
    )
    target_link_libraries(physicsmade PUBLIC CUDA::cudart)
    set_target_properties(
        physicsmade
        PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
    target_compile_definitions(physicsmade PUBLIC PHYSICSMADE_WITH_CUDA=1)
else()
    target_sources(
        physicsmade
        PRIVATE
            src/cuda/gravity_field_kernels_stub.cpp
            src/cuda/gravity_body_forces_stub.cpp
            src/cuda/render_instances_stub.cpp
            src/cuda/scene_buffers_stub.cpp
    )
endif()

target_include_directories(
    physicsmade
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(physicsmade PUBLIC cxx_std_20)
target_compile_definitions(physicsmade PUBLIC NOMINMAX)

physicsmade_enable_strict_warnings(physicsmade)

function(physicsmade_add_app target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE physicsmade::physicsmade)
    if(PHYSICSMADE_CUDA_ENABLED)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    endif()
    physicsmade_enable_strict_warnings(${target_name})
endfunction()

set(PHYSICSMADE_VIEWER_AVAILABLE OFF)
set(PHYSICSMADE_VIEWER_LIBS)
if(PHYSICSMADE_BUILD_VIEWER)
    find_package(glfw3 CONFIG QUIET)
    find_package(OpenGL QUIET COMPONENTS OpenGL GLU)

    if(TARGET glfw)
        list(APPEND PHYSICSMADE_VIEWER_LIBS glfw)
    elseif(EXISTS "C:/vcpkg/installed/x64-windows/include/GLFW/glfw3.h" AND EXISTS "C:/vcpkg/installed/x64-windows/lib/glfw3dll.lib")
        add_library(physicsmade_glfw INTERFACE)
        target_include_directories(physicsmade_glfw INTERFACE "C:/vcpkg/installed/x64-windows/include")
        target_link_libraries(
            physicsmade_glfw
            INTERFACE
                optimized "C:/vcpkg/installed/x64-windows/lib/glfw3dll.lib"
                debug "C:/vcpkg/installed/x64-windows/debug/lib/glfw3dll.lib"
        )
        list(APPEND PHYSICSMADE_VIEWER_LIBS physicsmade_glfw)
    endif()

    if(TARGET OpenGL::GL AND TARGET OpenGL::GLU)
        list(APPEND PHYSICSMADE_VIEWER_LIBS OpenGL::GL OpenGL::GLU)
    elseif(WIN32)
        add_library(physicsmade_opengl INTERFACE)
        target_link_libraries(physicsmade_opengl INTERFACE opengl32 glu32)
        list(APPEND PHYSICSMADE_VIEWER_LIBS physicsmade_opengl)
    endif()

    list(LENGTH PHYSICSMADE_VIEWER_LIBS PHYSICSMADE_VIEWER_LIB_COUNT)
    if(PHYSICSMADE_VIEWER_LIB_COUNT GREATER 1)
        set(PHYSICSMADE_VIEWER_AVAILABLE ON)
    else()
        message(STATUS "Skipping physicsmade_viewer: missing glfw3 or OpenGL development packages")
    endif()
endif()

if(PHYSICSMADE_BUILD_APPS)
    physicsmade_add_app(physicsmade_sandbox apps/sandbox/main.cpp)
    physicsmade_add_app(physicsmade_object_runner apps/object_runner/main.cpp)
    physicsmade_add_app(physicsmade_launcher apps/launcher/main.cpp)
    if(PHYSICSMADE_VIEWER_AVAILABLE)
        physicsmade_add_app(physicsmade_viewer apps/viewer/main.cpp)
        target_link_libraries(physicsmade_viewer PRIVATE ${PHYSICSMADE_VIEWER_LIBS})
    endif()
endif()

if(BUILD_TESTING AND PHYSICSMADE_BUILD_TESTS)
    add_executable(physicsmade_tests tests/main.cpp)
    target_link_libraries(physicsmade_tests PRIVATE physicsmade::physicsmade)
    if(PHYSICSMADE_CUDA_ENABLED)
        target_link_libraries(physicsmade_tests PRIVATE CUDA::cudart)
    endif()
    physicsmade_enable_strict_warnings(physicsmade_tests)
    add_test(NAME physicsmade_tests COMMAND physicsmade_tests)
endif()

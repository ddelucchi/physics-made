# Physics Made

Physics Made is a C++ and CUDA foundation for high-rigor spacetime simulation. It does not pretend that a finite codebase can already contain every physical law, but it is structured so new laws, metrics, solvers, and objects can be added without rewriting the engine.

## What the repository already provides

- A spacetime abstraction with a metric tensor interface and a Minkowski baseline implementation
- A weak-field relativistic metric, an exact isotropic Schwarzschild metric, a rotating Kerr metric in Cartesian Kerr-Schild coordinates, and a reusable numeric geodesic integrator for curved-spacetime trajectories
- Explicit kinematics models for both nonrelativistic and special-relativistic time evolution
- A world model that accepts separately coded objects and advances them through simulation time, including quaternion-based rigid-body orientation updates
- A pluggable physics system API now seeded with gravity, Lorentz-force electromagnetism, electrostatics, drag, and spring-network systems
- A constraint and contact layer with distance constraints, ball joints, hinge joints, hinge motors, angular limits, closed-loop articulated linkages, and iterative sequential-impulse sphere contact resolution with frictional spin generation
- A pluggable integrator API now seeded with semi-implicit Euler, velocity Verlet, and RK4
- CUDA-backed gravity field sampling, pairwise gravity body-force evaluation, persistent scene buffers, renderer-ready instance packing, and viewer-oriented GPU instance buffers with CPU fallbacks
- Scene serialization and plugin discovery so scenes and object programs can be loaded from manifest files instead of only hard-coded entry points, including articulated link definitions, reusable link instances, actuator presets, rigid-body state, loop diagnostics, and joint constraints
- Lattice field-theory modules spanning 1D scalar fields, higher-dimensional rectangular scalar lattices, compact U(1) gauge-field scaffolding, compact SU(2) non-Abelian gauge scaffolding, staggered fermion transport on a lattice, and an explicit coupled SU(2)+staggered-fermion backreaction slice
- A minimal quantum-chemistry surface for H/He systems via restricted and unrestricted Hartree-Fock plus a closed-shell MP2 correction on the same Gaussian integral layer, including live viewer sampling of actual Gaussian-basis electron density and orbital structure
- An orbit camera that rotates around a chosen object independently from the object implementation
- A live GLFW/OpenGL viewer that consumes the orbit camera and persistent scene-buffer pipeline through shader-based instanced mesh rendering, with direct CUDA/OpenGL buffer upload when device-resident scene buffers are available, including generated plugin scenes that can drive object state from live solver callbacks instead of only static manifests
- A standalone object runner so individually authored objects can be simulated on their own
- A launcher executable that lists built-in programs and runs them from one entry point, including discovered file-backed and generated plugin scenes
- Tests and executables showing how to wire custom objects into the world

## Repository layout

```text
include/physicsmade/   Public engine interfaces
src/                   Engine implementations
apps/sandbox/          Minimal demo application
apps/object_runner/    Standalone object execution
apps/launcher/         Launcher entry point
apps/viewer/           Live GLFW/OpenGL scene viewer
tests/                 Engine regression tests
docs/                  Architecture and extension notes
plugins/               Sample plugin manifests and serialized scene assets
```

## Build

### CPU-only

```powershell
cmake -S . -B build -DPHYSICSMADE_ENABLE_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

### CUDA-enabled

```powershell
cmake -S . -B build -DPHYSICSMADE_ENABLE_CUDA=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

If CMake does not detect a CUDA compiler, the project still builds with the CPU fallback implementation for field sampling.

If GLFW/OpenGL development packages are available, CMake also builds `physicsmade_viewer`. The current workspace falls back to the installed vcpkg GLFW/OpenGL import libraries automatically.

## Run the software

```powershell
.\build\Debug\physicsmade_launcher.exe
.\build\Debug\physicsmade_object_runner.exe --list
.\build\Debug\physicsmade_object_runner.exe --object rigid-spinner --integrator runge-kutta-4
.\build\Debug\physicsmade_object_runner.exe --object hinge-rotor --integrator velocity-verlet
.\build\Debug\physicsmade_object_runner.exe --object motorized-hinge-limit --integrator velocity-verlet
.\build\Debug\physicsmade_object_runner.exe --object four-bar-loop --integrator velocity-verlet
.\build\Debug\physicsmade_object_runner.exe --object magnetic-pair --integrator runge-kutta-4
.\build\Debug\physicsmade_object_runner.exe --object relativistic-probe --regime special-relativistic --integrator runge-kutta-4
.\build\Debug\physicsmade_launcher.exe plugin:sample-orbit
.\build\Debug\physicsmade_launcher.exe plugin:four-bar-loop
.\build\Debug\physicsmade_launcher.exe plugin:kerr-geodesic
.\build\Debug\physicsmade_launcher.exe plugin:coupled-su2-fermion-state
.\build\Debug\physicsmade_launcher.exe plugin:quantum-chemistry-live
.\build\Debug\physicsmade_launcher.exe kerr-preview
.\build\Debug\physicsmade_launcher.exe kerr-geodesic-preview
.\build\Debug\physicsmade_launcher.exe schwarzschild-preview
.\build\Debug\physicsmade_launcher.exe schwarzschild-geodesic-preview
.\build\Debug\physicsmade_launcher.exe quantum-chemistry-preview
.\build\Debug\physicsmade_launcher.exe quantum-chemistry-uhf-preview
.\build\Debug\physicsmade_launcher.exe quantum-chemistry-mp2-preview
.\build\Debug\physicsmade_launcher.exe quantum-scalar-preview
.\build\Debug\physicsmade_launcher.exe quantum-scalar-grid-preview
.\build\Debug\physicsmade_launcher.exe gauge-u1-preview
.\build\Debug\physicsmade_launcher.exe gauge-su2-preview
.\build\Debug\physicsmade_launcher.exe fermion-staggered-preview
.\build\Debug\physicsmade_launcher.exe coupled-su2-fermion-preview
.\build\Debug\physicsmade_viewer.exe --plugin four-bar-loop
.\build\Debug\physicsmade_viewer.exe --plugin kerr-geodesic
.\build\Debug\physicsmade_viewer.exe --plugin coupled-su2-fermion-state
.\build\Debug\physicsmade_viewer.exe --plugin quantum-chemistry-live
.\build\Debug\physicsmade_sandbox.exe
```

## Design stance

This project is intentionally honest about scope:

- No finite repository can literally pre-encode every physical theory.
- A rigorous simulation stack still needs explicit abstractions for metrics, state variables, integrators, constraints, units, and numerical error control.
- The current implementation focuses on a strong base architecture plus working reference models for gravity, Lorentz-force electromagnetism, drag, springs, rotational rigid bodies, motorized and limited joints, closed-loop articulated linkages, authored articulated assemblies, special relativity, weak-field/Schwarzschild/Kerr geometry, numeric geodesic integration, scene serialization, plugin discovery, generated viewer scenes with optional live solver-driven animation, scalar fields, compact Abelian and non-Abelian gauge scaffolding, staggered fermions, a coupled SU(2)-fermion lattice slice, restricted and unrestricted minimal Hartree-Fock quantum chemistry, live sampled Hartree-Fock electron density and orbital structure in the viewer, post-Hartree-Fock MP2 correction, and CUDA-assisted scene residency.
- The repository is now prepared for both relativistic and nonrelativistic execution paths, but more advanced theories still need explicit, testable system implementations.

## Next domains to add

- Gauge-field and tensor formulations beyond the current weak-field/Lorentz approximations and current compact U(1)/SU(2)/coupled-staggered-fermion scaffolds
- PDE solvers for continuum mechanics and relativistic hydrodynamics
- Adaptive and implicit integrators
- Larger Gaussian basis sets, higher-level post-Hartree-Fock chemistry, and renormalization workflows

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/EXTENDING.md](docs/EXTENDING.md) for the extension model.


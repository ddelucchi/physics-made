# Physics Made

Physics Made is an extensible C++20 scientific-computing framework for spacetime, classical dynamics, field models, compact lattice experiments, optional CUDA acceleration, and interactive visualization.

## Status

This repository is a research and engineering framework. It contains executable implementations and tests, but it is not presented as a validated replacement for specialist numerical-relativity, lattice-gauge, molecular-dynamics, or quantum-chemistry packages.

## Implemented surface

### Spacetime

- metric-tensor abstraction
- Minkowski baseline
- weak-field metric
- isotropic Schwarzschild metric
- rotating Kerr metric in Cartesian Kerr-Schild coordinates
- numerical geodesic integration

### Dynamics

- Newtonian and special-relativistic kinematics models
- semi-implicit Euler, velocity Verlet, and RK4 integration
- rigid-body orientation using quaternions
- distance, ball-joint, hinge, motor, and angular-limit constraints
- iterative sphere-contact resolution

### Forces and fields

- Newtonian gravity
- electrostatics
- Lorentz-force electromagnetism
- linear drag
- Hookean spring networks
- CPU implementations with optional CUDA paths for gravity-field sampling, body-force evaluation, and scene buffers

### Field-theory experiments

- scalar lattices
- rectangular scalar lattices
- compact U(1) gauge scaffolding
- compact SU(2) gauge scaffolding
- staggered fermion transport
- a coupled SU(2) plus staggered-fermion slice

### Quantum-chemistry surface

- compact H/He-oriented Gaussian-basis infrastructure
- restricted and unrestricted Hartree-Fock paths
- closed-shell MP2 correction hooks
- density/orbital sampling for visualization

### Runtime and visualization

- scene serialization
- plugin discovery
- generated scene/program registry
- standalone object runner
- launcher
- optional GLFW/OpenGL viewer
- CPU fallbacks when CUDA is unavailable

## Repository map

- `include/physicsmade/` - public interfaces
- `src/` - implementations
- `apps/` - sandbox, object runner, launcher, and optional viewer
- `tests/main.cpp` - executable regression suite
- `docs/ARCHITECTURE.md` - system architecture
- `docs/EXTENDING.md` - extension guidance

## Portable CPU build

The most reproducible public path disables CUDA and the optional viewer:

```bash
cmake -S . -B build \
  -DPHYSICSMADE_ENABLE_CUDA=OFF \
  -DPHYSICSMADE_BUILD_VIEWER=OFF \
  -DPHYSICSMADE_BUILD_APPS=ON \
  -DPHYSICSMADE_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## CUDA build

If a CUDA toolchain is available, leave `PHYSICSMADE_ENABLE_CUDA=ON`. The build automatically falls back to CPU stub implementations when CUDA is unavailable.

The live viewer additionally requires GLFW and OpenGL development packages. Viewer dependencies are discovered through CMake rather than hard-coded workstation paths.

## Validation philosophy

Tests exercise mathematical primitives, lightlike intervals, gravity, field sampling, rigid-body behavior, constraints, spacetime metrics, runtime wiring, and CPU/CUDA-style interfaces. Passing these tests establishes software regressions and local invariants; it does not by itself establish research-grade physical accuracy for every model.

## Scope

The broad module surface is intentional: this project explores how independently authored physical systems can coexist behind a common runtime. Individual modules should be judged by their own equations, discretization choices, convergence tests, and validation references rather than by the existence of a common interface.

## License

Source is publicly viewable for portfolio and technical evaluation. See [LICENSE](LICENSE).

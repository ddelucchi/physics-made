# Architecture

## Goal

Physics Made is organized around a simple principle: geometry, matter state, dynamics, and observation are separate concerns.

## Layers

### 1. Geometry

The spacetime layer defines the metric tensor used to evaluate intervals and to encode the background geometry in which physics evolves.

Current implementation:

- `MetricTensor` interface
- `MinkowskiMetric` reference metric
- `WeakFieldSphericalMassMetric` reference curved approximation
- `SchwarzschildMetric` exact static vacuum solution in isotropic coordinates
- `KerrMetric` rotating vacuum solution in Cartesian Kerr-Schild coordinates
- `NumericGeodesicIntegrator` for reusable metric-driven timelike trajectory integration
- `SpacetimeModel` wrapper for geometry plus physical constants

### 2. Kinematics

The kinematics layer separates the definition of motion from the force laws themselves.

Current implementation:

- `KinematicsModel` interface
- `NewtonianKinematics`
- `SpecialRelativisticKinematics`

This is what makes the same scene state and many of the same systems usable in both nonrelativistic and relativistic runs.

### 3. Scene State

The scene layer stores object state independently from object authoring. A user can create new object classes that generate initial state without changing the engine core.

Current implementation:

- `ObjectState`
- `SimObject`

The scene state now carries translational and rotational degrees of freedom:

- position and velocity
- orientation quaternion and angular velocity
- diagonal body-frame inertia for rigid-body response

### 4. Dynamics

The dynamics layer is split in two parts:

- `PhysicsSystem` computes force or field contributions from a given law
- `StateIntegrator` advances object state in time

This split keeps the law of motion independent from the numerical scheme.

Current reference systems:

- Newtonian gravity
- electrostatics
- Lorentz-force electromagnetism with electric and magnetic terms
- drag
- spring networks

Current constraint layer:

- distance constraints
- ball joints with local anchors
- hinge joints with anchor and axis alignment
- hinge motors with target angular speed and capped drive torque
- hinge angular limits with signed reference-angle enforcement
- closed-loop articulated mechanisms built from multi-hinge rigid links
- iterative sequential-impulse sphere contact resolution with angular friction response

Current reference integrators:

- semi-implicit Euler
- velocity Verlet
- RK4

### 5. Acceleration

The CUDA layer evaluates sampled fields over a grid. It is isolated from the rest of the engine so that:

- CPU fallback remains available
- additional GPU kernels can be added without changing the world API
- field solvers can evolve faster than the scene stepping code

Current CUDA-backed paths:

- gravity field grids
- pairwise gravity body forces
- render-instance packing for renderer-facing scene data, including orientation and angular-speed metadata
- viewer-oriented GPU render-instance packing for direct instanced rendering uploads
- persistent scene buffers exposing interop-facing device views and a stable viewer-instance buffer

### 6. Data-Driven Scenes

The IO and plugin layers allow scenes to be defined and discovered without recompiling the engine.

Current implementation:

- line-based scene serialization for object state, systems, rigid-body joints, articulated link definitions, reusable link instances, actuator presets, and loop diagnostics
- manifest-based plugin discovery for scenes and object-program presets
- sample and generated plugin assets under the top-level `plugins/` directory
- generated viewer scenes can optionally attach a per-frame solver callback so the live renderer consumes evolving state rather than only a static serialized manifest

### 7. Quantum Chemistry

The quantum-chemistry layer adds bound-state electronic structure models that sit between discrete particles and full field-theory lattices.

Current implementation:

- closed-shell restricted Hartree-Fock solver in atomic units
- open-shell unrestricted Hartree-Fock solver in atomic units
- closed-shell MP2 correlation correction on top of the restricted Hartree-Fock reference
- minimal STO-3G-style 1s Gaussian basis support for H/He nuclei
- one-electron overlap, kinetic, and nuclear-attraction integrals
- two-electron Coulomb/exchange integrals for s-type contracted Gaussians
- stateful restricted Hartree-Fock iteration support for stepping SCF convergence frame by frame
- basis-function, molecular-orbital, and electron-density evaluators so live viewer scenes can sample actual electronic structure instead of authored proxies

This is a minimal ab initio chemistry surface, not a substitute for a production electronic-structure package.

### 8. Field Theory

The field-theory layer extends the engine past point particles and into discretized fields.

Current implementation:

- one-dimensional scalar lattice field evolution
- rectangular higher-dimensional scalar lattice evolution
- compact U(1) gauge-field lattice scaffolding with plaquette observables
- compact SU(2) gauge-field lattice scaffolding with quaternion-valued links and color plaquette observables
- staggered fermion lattice evolution with SU(2)-transported color doublets and explicit observables for norm, energy, and chiral density proxy
- coupled SU(2) gauge plus staggered fermion lattice evolution with explicit fermion-current backreaction on gauge electric fields
- lattice observables for total energy, amplitude, mean field, and nearest-neighbor correlation

This is a foundation for broader QFT-style work, not a claim of full nonperturbative quantum field theory coverage.

### 9. Observation

The orbit camera is a separate observer model. It rotates around a selected object using spherical coordinates and returns a full camera basis:

- position
- forward
- right
- up

That observer model now feeds both headless previews and a live GLFW/OpenGL viewer that renders instanced meshes from scene-buffer instance data in real time, using direct device-buffer upload when CUDA scene residency is available.

Generated plugin scenes can also provide live scene steppers, which lets the viewer render evolving geodesic and lattice solver state directly instead of only replaying a fixed snapshot world.

### 10. Runtime Surface

The runtime layer exposes the engine through executable entry points:

- sandbox program
- standalone object runner
- launcher catalog
- discovered plugin scene previews
- generated plugin scene for live Hartree-Fock H2 density and virtual-orbital visualization
- generated plugin scenes for Kerr geodesics and coupled lattice states
- Kerr metric preview target
- Kerr geodesic preview target
- Schwarzschild metric preview target
- Schwarzschild geodesic preview target
- restricted Hartree-Fock quantum-chemistry preview target
- unrestricted Hartree-Fock quantum-chemistry preview target
- restricted MP2 quantum-chemistry preview target
- scalar field preview targets
- gauge-field preview targets for both U(1) and SU(2)
- staggered-fermion preview targets
- coupled SU(2)-fermion preview target
- live viewer executable for plugin scenes and serialized scenes

This keeps authored objects individually runnable while still giving the user a single launcher experience.

## Current reference implementation

The repository currently ships one reference law:

- Newtonian pairwise gravity
- electrostatic interactions
- Lorentz-force interactions with magnetic terms
- drag forces
- spring-network forces
- distance constraints, ball joints, hinge joints, motorized/limited hinges, closed-loop articulated mechanisms, and sphere contacts

And two reference kinematic regimes:

- Newtonian
- special relativistic

Those laws are not a claim of completeness. They are the validated seed from which more advanced relativistic, continuum, and field-theory systems can be added.

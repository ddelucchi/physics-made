# Extending Physics Made

## Add a new object type

Create a class deriving from `physicsmade::scene::SimObject` and return an `ObjectState` from `initialState()`.

Use this when the object needs its own authoring logic, parameters, or validation rules.

If you want the object to be individually runnable from the shipped runtime, add it to the built-in runtime catalog in `src/runtime/programs.cpp` or mirror that pattern in your own launcher.

## Add a new kinematics regime

Create a class deriving from `physicsmade::physics::KinematicsModel` and set it on `SimulationWorld`.

Use this when the meaning of momentum, kinetic energy, time dilation, or force-to-acceleration mapping changes.

Examples that fit this interface:

- Galilean kinematics with custom unit systems
- special relativity in curved coordinates approximations
- post-Newtonian approximations
- custom capped-speed gameplay models

## Add a new physics law

Create a class deriving from `physicsmade::physics::PhysicsSystem` and implement `accumulateForces(...)`.

If the system is conservative, also implement `potentialEnergy(...)` so the world diagnostics can report total energy consistently.

Examples that fit this interface:

- electrostatics
- Lorentz-force electromagnetic systems
- drag fields
- spring networks
- coarse collision response

If the law is naturally expressed as a field source instead of a pure pair force, keep the host API in terms of `ObjectState` and let internal helpers build the tensor, grid, or lattice representation.

## Add a new constraint or contact solver

Create a class deriving from `physicsmade::physics::Constraint` for a single relation, or derive from `ConstraintSolver` if you need a different global solve strategy.

Examples that fit this interface:

- rod or hinge constraints
- ball joints and swing constraints
- joint limits
- joint motors and servos
- closed-loop linkage assemblies
- rigid contact manifolds
- projected Gauss-Seidel or XPBD solvers

If a constraint needs rotational response, use the rigid-body helper functions so impulses are applied through both linear and angular channels instead of only pushing object centers.

## Add a new integrator

Create a class deriving from `physicsmade::physics::StateIntegrator` and set it on `SimulationWorld`.

Integrators worth adding next:

- symplectic leapfrog
- adaptive embedded Runge-Kutta
- implicit midpoint
- Gauss-Legendre collocation

## Add a new spacetime metric

Create a new `MetricTensor` implementation and construct `SimulationWorld` with a `SpacetimeModel` that owns it.

If the metric is intended to be exact rather than perturbative, keep the coordinate chart explicit in the type and document the domain where the chart is valid, as done by the isotropic Schwarzschild metric.

If the metric is rotating or carries off-diagonal time-space terms, keep those frame-dragging terms explicit in the chosen chart instead of hiding them behind a coordinate transform, as in the Cartesian Kerr-Schild Kerr implementation.

If you also need test-particle trajectories in that geometry, follow the `NumericGeodesicIntegrator` pattern so Christoffel evaluation and affine stepping stay separate from the metric definition itself.

## Add a new quantum-chemistry model

Follow the pattern in `include/physicsmade/quantum_chemistry/restricted_hartree_fock.hpp` when the physics lives in an electronic-structure basis rather than directly in object state or a lattice field.

Keep the unit system explicit, expose physically meaningful observables such as orbital energies and total energy, and separate basis construction from the actual solver so larger basis sets or post-Hartree-Fock methods can reuse the integral layer.

If the chemistry needs to drive the live viewer, expose basis-evaluation helpers and, when appropriate, a stateful SCF iteration surface so generated viewer scenes can sample actual electron density or orbital amplitudes from solver state instead of inventing ad hoc particles.

If the target system is open-shell, keep the spin populations explicit in the API, as in the unrestricted Hartree-Fock extension, instead of hard-coding closed-shell occupancy assumptions into the solver core.

If you are adding post-Hartree-Fock methods, keep the Hartree-Fock reference result reusable and layer the correlation correction on top of the same integral and orbital data instead of recomputing unrelated chemistry state in a separate stack.

## Add a new CUDA field solver

Follow the pattern in `src/cuda/gravity_field_kernels.cu` and expose a host API through `include/physicsmade/cuda/field_grid.hpp`.

Keep the public API stable and allow a CPU fallback for environments where CUDA is unavailable.

If the goal is renderer throughput rather than only numerical sampling, mirror the pattern in `include/physicsmade/cuda/render_instances.hpp` so scene state can be packed into renderer-facing buffers efficiently.

If the goal is persistent renderer interop rather than one-shot packing, extend `SceneBufferManager` so device pointers, bounds, viewer origins, and download paths stay stable across frames.

## Add a new serialized scene or plugin

Use `physicsmade::io::SceneManifest` for line-based scene data and add a `.pmplugin` manifest under `plugins/` when you want the launcher or viewer to discover it automatically.

If the scene should be generated from a solver rather than stored as a static file, point the plugin manifest at a generated scene id and implement the builder through `physicsmade::runtime::buildGeneratedScene(...)` so the launcher and viewer resolve the same scene source.

If the viewer should render that generated scene live, also implement `physicsmade::runtime::buildGeneratedViewerScene(...)` so the plugin can supply a per-frame world-state updater instead of stopping at a static snapshot.

This is the right place to add authored presets that combine object state, rigid-body orientation/spin, reusable articulated link definitions, instantiated link placements, motorized or limited hinges, closed loops, loop diagnostics, systems, and preview parameters without recompiling the runtime.

## Add a new field-theory module

Follow the pattern in `include/physicsmade/field_theory/scalar_field_lattice.hpp`, `include/physicsmade/field_theory/rectangular_scalar_field_lattice.hpp`, `include/physicsmade/field_theory/u1_gauge_field_lattice.hpp`, `include/physicsmade/field_theory/su2_gauge_field_lattice.hpp`, or `include/physicsmade/field_theory/staggered_fermion_lattice.hpp` when the simulated quantity is distributed over a lattice instead of attached to a discrete object.

Keep observables explicit so runtime previews and tests can validate energy growth, correlations, and amplitude bounds.

For gauge-field work, expose plaquette or Wilson-style observables directly so numerical drift can be tracked from the runtime without additional tooling.

If the field lives on a non-Abelian group, keep the link representation explicit and normalized so stepping and diagnostics can share the same group-valued state.

If the field carries fermionic degrees of freedom, keep the transport operator, mass term, and any staggered or Wilson-style phase factors explicit in the API so the runtime can report physically meaningful observables instead of only raw lattice buffers.

If the module couples gauge and matter sectors directly, expose both gauge observables and fermion observables from the same type and keep the backreaction scale explicit so tests can validate the interaction path rather than only the uncoupled limits.

## Add a new launch target

Expose it through `include/physicsmade/runtime/programs.hpp`, implement it in `src/runtime/programs.cpp`, and it will become available to the launcher. If the target should be visualizable in real time, mirror it as a serialized scene or plugin so `physicsmade_viewer` can load it directly.

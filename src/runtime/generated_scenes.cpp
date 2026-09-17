#include "physicsmade/runtime/generated_scenes.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/field_theory/coupled_su2_fermion_lattice.hpp"
#include "physicsmade/quantum_chemistry/restricted_hartree_fock.hpp"
#include "physicsmade/spacetime/geodesic_integrator.hpp"
#include "physicsmade/spacetime/kerr_metric.hpp"

namespace {

physicsmade::scene::ObjectState makePinnedObject(
    std::string name,
    const physicsmade::math::Vector3& position,
    double radius,
    double charge,
    double temperatureKelvin,
    double emissiveIntensity) {
    physicsmade::scene::ObjectState object{};
    object.name = std::move(name);
    object.mass = 1.0;
    object.charge = charge;
    object.radius = radius;
    object.pinned = true;
    object.position = position;
    object.temperatureKelvin = temperatureKelvin;
    object.emissiveIntensity = emissiveIntensity;
    return object;
}

double safeFraction(double numerator, double denominator) {
    return numerator / std::max(denominator, 1.0e-12);
}

physicsmade::math::Vector3 scaledCoordinateVelocity(
    const physicsmade::spacetime::GeodesicState& state,
    double coordinateScale) {
    const double temporal = std::max(std::abs(state.tangent.t), physicsmade::common::kEpsilon);
    return (coordinateScale * physicsmade::common::kSpeedOfLight / temporal) * state.tangent.spatial();
}

struct KerrLiveSceneState {
    std::shared_ptr<physicsmade::spacetime::KerrMetric> metric{};
    physicsmade::spacetime::SpacetimeModel spacetime{};
    physicsmade::spacetime::NumericGeodesicIntegrator integrator{};
    physicsmade::spacetime::GeodesicState current{};
    std::vector<physicsmade::math::Vector3> trail{};
    double coordinateScale{1.0};
};

struct CoupledLatticeLiveSceneState {
    physicsmade::field_theory::CoupledSu2FermionLattice lattice{};
    std::vector<std::size_t> sampleIndices{};
    physicsmade::math::Vector3 centerOffset{};
};

struct QuantumChemistryProbeSet {
    std::vector<physicsmade::math::Vector3> densitySamplePoints{};
    std::vector<physicsmade::math::Vector3> virtualOrbitalSamplePoints{};
    std::size_t virtualOrbitalIndex{0};
    double densityReference{0.0};
    double orbitalReference{0.0};
};

struct QuantumChemistryLiveSceneState {
    physicsmade::quantum_chemistry::Molecule molecule{};
    physicsmade::quantum_chemistry::RestrictedHartreeFockState scf{};
    QuantumChemistryProbeSet probes{};
    double sceneScale{1.35};
    double resetDelaySeconds{0.0};
};

std::vector<physicsmade::math::Vector3> downsamplePoints(
    const std::vector<physicsmade::math::Vector3>& points,
    std::size_t maxCount) {
    if (points.size() <= maxCount || maxCount == 0) {
        return points;
    }

    const std::size_t stride = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(
        static_cast<double>(points.size()) / static_cast<double>(maxCount))));
    std::vector<physicsmade::math::Vector3> reduced;
    reduced.reserve(maxCount);
    for (std::size_t index = 0; index < points.size(); index += stride) {
        reduced.push_back(points[index]);
    }
    return reduced;
}

std::vector<physicsmade::math::Vector3> buildQuantumChemistryGrid() {
    std::vector<physicsmade::math::Vector3> grid;
    for (double x = -2.6; x <= 2.600001; x += 0.35) {
        for (double y = -1.75; y <= 1.750001; y += 0.35) {
            for (double z = -1.75; z <= 1.750001; z += 0.35) {
                grid.push_back({x, y, z});
            }
        }
    }
    return grid;
}

physicsmade::quantum_chemistry::RestrictedHartreeFockState convergeRestrictedHartreeFockState(
    const physicsmade::quantum_chemistry::Molecule& molecule) {
    auto state = physicsmade::quantum_chemistry::initializeRestrictedHartreeFockState(molecule);
    while (!state.converged && state.iterations < state.options.maxIterations) {
        physicsmade::quantum_chemistry::stepRestrictedHartreeFock(state);
    }
    return state;
}

QuantumChemistryProbeSet buildQuantumChemistryProbeSet(
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& referenceState) {
    QuantumChemistryProbeSet probes{};
    const auto grid = buildQuantumChemistryGrid();
    std::vector<double> densityValues;
    densityValues.reserve(grid.size());
    for (const auto& samplePoint : grid) {
        const double density = physicsmade::quantum_chemistry::evaluateElectronDensity(
            referenceState.basis,
            referenceState.densityMatrix,
            samplePoint);
        densityValues.push_back(density);
        probes.densityReference = std::max(probes.densityReference, density);
    }

    for (std::size_t sampleIndex = 0; sampleIndex < grid.size(); ++sampleIndex) {
        if (densityValues[sampleIndex] >= (0.045 * probes.densityReference)) {
            probes.densitySamplePoints.push_back(grid[sampleIndex]);
        }
    }
    if (probes.densitySamplePoints.empty() && !grid.empty()) {
        const auto maxDensityIt = std::max_element(densityValues.begin(), densityValues.end());
        probes.densitySamplePoints.push_back(grid[static_cast<std::size_t>(maxDensityIt - densityValues.begin())]);
    }
    probes.densitySamplePoints = downsamplePoints(probes.densitySamplePoints, 220);

    const std::size_t occupiedCount = static_cast<std::size_t>(std::max(referenceState.occupiedOrbitals, 0));
    probes.virtualOrbitalIndex = occupiedCount < referenceState.basis.size()
                                     ? occupiedCount
                                     : (referenceState.basis.empty() ? 0 : referenceState.basis.size() - 1);

    std::vector<double> orbitalValues;
    orbitalValues.reserve(grid.size());
    for (std::size_t sampleIndex = 0; sampleIndex < grid.size(); ++sampleIndex) {
        const double orbitalValue = std::abs(physicsmade::quantum_chemistry::evaluateMolecularOrbital(
            referenceState.basis,
            referenceState.coefficientMatrix,
            probes.virtualOrbitalIndex,
            grid[sampleIndex]));
        orbitalValues.push_back(orbitalValue);
        probes.orbitalReference = std::max(probes.orbitalReference, orbitalValue);
    }

    for (std::size_t sampleIndex = 0; sampleIndex < grid.size(); ++sampleIndex) {
        if (orbitalValues[sampleIndex] >= (0.19 * probes.orbitalReference) &&
            densityValues[sampleIndex] <= (0.38 * probes.densityReference)) {
            probes.virtualOrbitalSamplePoints.push_back(grid[sampleIndex]);
        }
    }
    if (probes.virtualOrbitalSamplePoints.empty() && !grid.empty()) {
        const auto maxOrbitalIt = std::max_element(orbitalValues.begin(), orbitalValues.end());
        probes.virtualOrbitalSamplePoints.push_back(grid[static_cast<std::size_t>(maxOrbitalIt - orbitalValues.begin())]);
    }
    probes.virtualOrbitalSamplePoints = downsamplePoints(probes.virtualOrbitalSamplePoints, 160);
    return probes;
}

physicsmade::scene::ObjectState makeQuantumChemistryNucleusObject(
    const physicsmade::quantum_chemistry::Nucleus& nucleus,
    std::size_t nucleusIndex,
    double sceneScale) {
    return makePinnedObject(
        "qc-nucleus-" + std::to_string(nucleusIndex),
        sceneScale * nucleus.position,
        0.16 + (0.03 * nucleus.charge),
        2.0e-6 * nucleus.charge,
        5800.0 + (600.0 * nucleus.charge),
        4.0 + (0.8 * nucleus.charge));
}

physicsmade::scene::ObjectState makeQuantumChemistryDensityObject(
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& scf,
    const QuantumChemistryProbeSet& probes,
    const physicsmade::math::Vector3& samplePoint,
    std::size_t sampleIndex,
    double sceneScale) {
    const double density = physicsmade::quantum_chemistry::evaluateElectronDensity(
        scf.basis,
        scf.densityMatrix,
        samplePoint);
    const double fraction = std::clamp(safeFraction(density, probes.densityReference), 0.0, 1.0);
    auto object = makePinnedObject(
        "qc-density-" + std::to_string(sampleIndex),
        sceneScale * samplePoint,
        0.03 + (0.12 * std::sqrt(fraction)),
        -1.8e-6 * fraction,
        320.0 + (4200.0 * fraction),
        0.12 + (5.2 * fraction));
    object.mass = 0.0;
    return object;
}

physicsmade::scene::ObjectState makeQuantumChemistryVirtualOrbitalObject(
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& scf,
    const QuantumChemistryProbeSet& probes,
    const physicsmade::math::Vector3& samplePoint,
    std::size_t sampleIndex,
    double sceneScale) {
    const double amplitude = physicsmade::quantum_chemistry::evaluateMolecularOrbital(
        scf.basis,
        scf.coefficientMatrix,
        probes.virtualOrbitalIndex,
        samplePoint);
    const double fraction = std::clamp(safeFraction(std::abs(amplitude), probes.orbitalReference), 0.0, 1.0);
    auto object = makePinnedObject(
        "qc-virtual-orbital-" + std::to_string(sampleIndex),
        sceneScale * samplePoint,
        0.022 + (0.085 * std::sqrt(fraction)),
        (amplitude >= 0.0 ? 1.6e-6 : -1.6e-6) * fraction,
        900.0 + (2400.0 * fraction),
        0.18 + (4.0 * fraction));
    object.mass = 0.0;
    return object;
}

void appendQuantumChemistryObjects(
    std::vector<physicsmade::scene::ObjectState>& objects,
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& scf,
    const QuantumChemistryProbeSet& probes,
    double sceneScale) {
    for (std::size_t nucleusIndex = 0; nucleusIndex < scf.molecule.nuclei.size(); ++nucleusIndex) {
        objects.push_back(makeQuantumChemistryNucleusObject(scf.molecule.nuclei[nucleusIndex], nucleusIndex, sceneScale));
    }

    for (std::size_t sampleIndex = 0; sampleIndex < probes.densitySamplePoints.size(); ++sampleIndex) {
        objects.push_back(makeQuantumChemistryDensityObject(
            scf,
            probes,
            probes.densitySamplePoints[sampleIndex],
            sampleIndex,
            sceneScale));
    }

    for (std::size_t sampleIndex = 0; sampleIndex < probes.virtualOrbitalSamplePoints.size(); ++sampleIndex) {
        objects.push_back(makeQuantumChemistryVirtualOrbitalObject(
            scf,
            probes,
            probes.virtualOrbitalSamplePoints[sampleIndex],
            sampleIndex,
            sceneScale));
    }
}

void updateQuantumChemistryObjects(
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& scf,
    const QuantumChemistryProbeSet& probes,
    double sceneScale,
    std::vector<physicsmade::scene::ObjectState>& objects,
    double dtSeconds) {
    std::size_t objectIndex = 0;
    for (std::size_t nucleusIndex = 0; nucleusIndex < scf.molecule.nuclei.size(); ++nucleusIndex, ++objectIndex) {
        auto& nucleusObject = objects[objectIndex];
        nucleusObject.position = sceneScale * scf.molecule.nuclei[nucleusIndex].position;
        nucleusObject.coordinateTimeSeconds += dtSeconds;
        nucleusObject.properTimeSeconds += dtSeconds;
    }

    for (std::size_t sampleIndex = 0; sampleIndex < probes.densitySamplePoints.size(); ++sampleIndex, ++objectIndex) {
        const double density = physicsmade::quantum_chemistry::evaluateElectronDensity(
            scf.basis,
            scf.densityMatrix,
            probes.densitySamplePoints[sampleIndex]);
        const double fraction = std::clamp(safeFraction(density, probes.densityReference), 0.0, 1.0);
        auto& densityObject = objects[objectIndex];
        densityObject.position = sceneScale * probes.densitySamplePoints[sampleIndex];
        densityObject.radius = 0.03 + (0.12 * std::sqrt(fraction));
        densityObject.charge = -1.8e-6 * fraction;
        densityObject.temperatureKelvin = 320.0 + (4200.0 * fraction);
        densityObject.emissiveIntensity = 0.12 + (5.2 * fraction);
        densityObject.coordinateTimeSeconds += dtSeconds;
        densityObject.properTimeSeconds += dtSeconds;
    }

    for (std::size_t sampleIndex = 0; sampleIndex < probes.virtualOrbitalSamplePoints.size(); ++sampleIndex, ++objectIndex) {
        const double amplitude = physicsmade::quantum_chemistry::evaluateMolecularOrbital(
            scf.basis,
            scf.coefficientMatrix,
            probes.virtualOrbitalIndex,
            probes.virtualOrbitalSamplePoints[sampleIndex]);
        const double fraction = std::clamp(safeFraction(std::abs(amplitude), probes.orbitalReference), 0.0, 1.0);
        auto& orbitalObject = objects[objectIndex];
        orbitalObject.position = sceneScale * probes.virtualOrbitalSamplePoints[sampleIndex];
        orbitalObject.radius = 0.022 + (0.085 * std::sqrt(fraction));
        orbitalObject.charge = (amplitude >= 0.0 ? 1.6e-6 : -1.6e-6) * fraction;
        orbitalObject.temperatureKelvin = 900.0 + (2400.0 * fraction);
        orbitalObject.emissiveIntensity = 0.18 + (4.0 * fraction);
        orbitalObject.coordinateTimeSeconds += dtSeconds;
        orbitalObject.properTimeSeconds += dtSeconds;
    }
}

physicsmade::io::SceneManifest buildQuantumChemistryScene() {
    const auto referenceState = convergeRestrictedHartreeFockState(physicsmade::quantum_chemistry::makeHydrogenMolecule());
    const auto probes = buildQuantumChemistryProbeSet(referenceState);

    physicsmade::io::SceneManifest manifest{};
    manifest.sceneName = "quantum-chemistry-live";
    manifest.kinematicsId = "newtonian";
    manifest.integratorId = "velocity-verlet";
    appendQuantumChemistryObjects(manifest.objects, referenceState, probes, 1.35);
    return manifest;
}

physicsmade::runtime::GeneratedViewerScene buildQuantumChemistryLiveViewerScene() {
    auto liveState = std::make_shared<QuantumChemistryLiveSceneState>();
    liveState->molecule = physicsmade::quantum_chemistry::makeHydrogenMolecule();
    const auto referenceState = convergeRestrictedHartreeFockState(liveState->molecule);
    liveState->probes = buildQuantumChemistryProbeSet(referenceState);
    liveState->scf = physicsmade::quantum_chemistry::initializeRestrictedHartreeFockState(liveState->molecule);

    physicsmade::runtime::GeneratedViewerScene scene{};
    scene.manifest.sceneName = "quantum-chemistry-live";
    scene.manifest.kinematicsId = "newtonian";
    scene.manifest.integratorId = "velocity-verlet";
    scene.manifest.previewDtSeconds = 0.05;
    appendQuantumChemistryObjects(scene.manifest.objects, liveState->scf, liveState->probes, liveState->sceneScale);
    scene.suggestedFollowIndex = 0;
    scene.step = [liveState](physicsmade::simulation::SimulationWorld& world, double dtSeconds) {
        if (liveState->scf.converged) {
            liveState->resetDelaySeconds += dtSeconds;
            if (liveState->resetDelaySeconds >= 1.0) {
                liveState->scf = physicsmade::quantum_chemistry::initializeRestrictedHartreeFockState(liveState->molecule);
                liveState->resetDelaySeconds = 0.0;
            }
        } else {
            const int iterations = std::max(1, static_cast<int>(std::lround(dtSeconds / 0.05)));
            for (int iteration = 0; iteration < iterations; ++iteration) {
                if (liveState->scf.converged) {
                    break;
                }
                physicsmade::quantum_chemistry::stepRestrictedHartreeFock(liveState->scf);
            }
            if (liveState->scf.converged) {
                liveState->resetDelaySeconds = 0.0;
            }
        }

        updateQuantumChemistryObjects(
            liveState->scf,
            liveState->probes,
            liveState->sceneScale,
            world.objects(),
            dtSeconds);
    };
    return scene;
}

physicsmade::io::SceneManifest buildKerrGeodesicScene() {
    constexpr double solarMass = 1.98847e30;
    const double centralMass = 4.154e6 * solarMass;
    const auto metric = std::make_shared<physicsmade::spacetime::KerrMetric>(centralMass, 0.78);
    const physicsmade::spacetime::SpacetimeModel spacetime(metric);
    const double scale = 1.0 / metric->gravitationalRadius();
    const double orbitalRadius = 85.0 * metric->gravitationalRadius();
    const double orbitalSpeed = std::sqrt(physicsmade::common::kGravitationalConstant * centralMass / orbitalRadius);
    const auto initialState = physicsmade::spacetime::makeTimelikeGeodesicState(
        spacetime,
        {0.0, orbitalRadius, 0.0, 0.0},
        {0.0, orbitalSpeed, 0.0});
    const physicsmade::spacetime::NumericGeodesicIntegrator integrator(spacetime, {600.0, 1.0, 1.0e-6});
    const auto trajectory = integrator.integrate(initialState, 180);

    physicsmade::io::SceneManifest manifest{};
    manifest.sceneName = "kerr-geodesic";
    manifest.kinematicsId = "newtonian";
    manifest.integratorId = "velocity-verlet";
    manifest.objects.push_back(makePinnedObject(
        "kerr-horizon",
        {0.0, 0.0, 0.0},
        metric->eventHorizonRadius() * scale,
        0.0,
        1800.0,
        7.0));
    manifest.objects.push_back(makePinnedObject(
        "spin-axis-north",
        {0.0, 0.0, 5.0},
        0.35,
        3.0e-6,
        6500.0,
        4.0));
    manifest.objects.push_back(makePinnedObject(
        "spin-axis-south",
        {0.0, 0.0, -5.0},
        0.35,
        -3.0e-6,
        3500.0,
        4.0));

    const std::size_t stride = 2;
    for (std::size_t index = 0; index < trajectory.size(); index += stride) {
        const auto& state = trajectory[index];
        const double fraction = static_cast<double>(index) / static_cast<double>(trajectory.size() - 1);
        const auto position = scale * state.position.spatial();
        const double emissive = 1.2 + (5.0 * fraction);
        const double temperature = 500.0 + (5500.0 * fraction);
        const double charge = (fraction < 0.5) ? 2.0e-6 : -2.0e-6;
        manifest.objects.push_back(makePinnedObject(
            "geodesic-sample-" + std::to_string(index),
            position,
            0.18,
            charge,
            temperature,
            emissive));
    }

    const auto& finalState = trajectory.back();
    manifest.objects.push_back(makePinnedObject(
        "geodesic-end",
        scale * finalState.position.spatial(),
        0.28,
        -4.0e-6,
        6200.0,
        7.5));
    return manifest;
}

physicsmade::runtime::GeneratedViewerScene buildKerrLiveViewerScene() {
    constexpr double solarMass = 1.98847e30;
    const double centralMass = 4.154e6 * solarMass;
    auto liveState = std::make_shared<KerrLiveSceneState>();
    liveState->metric = std::make_shared<physicsmade::spacetime::KerrMetric>(centralMass, 0.78);
    liveState->spacetime = physicsmade::spacetime::SpacetimeModel(liveState->metric);
    liveState->coordinateScale = 1.0 / liveState->metric->gravitationalRadius();

    const double orbitalRadius = 85.0 * liveState->metric->gravitationalRadius();
    const double orbitalSpeed = std::sqrt(physicsmade::common::kGravitationalConstant * centralMass / orbitalRadius);
    liveState->current = physicsmade::spacetime::makeTimelikeGeodesicState(
        liveState->spacetime,
        {0.0, orbitalRadius, 0.0, 0.0},
        {0.0, orbitalSpeed, 0.0});
    liveState->integrator = physicsmade::spacetime::NumericGeodesicIntegrator(liveState->spacetime, {600.0, 1.0, 1.0e-6});

    constexpr std::size_t trailCount = 48;
    liveState->trail.assign(trailCount, liveState->coordinateScale * liveState->current.position.spatial());

    physicsmade::runtime::GeneratedViewerScene scene{};
    scene.manifest.sceneName = "kerr-geodesic";
    scene.manifest.kinematicsId = "newtonian";
    scene.manifest.integratorId = "velocity-verlet";
    scene.manifest.previewDtSeconds = liveState->integrator.options().affineStep;
    scene.manifest.objects.push_back(makePinnedObject(
        "kerr-horizon",
        {0.0, 0.0, 0.0},
        liveState->metric->eventHorizonRadius() * liveState->coordinateScale,
        0.0,
        1800.0,
        7.0));
    scene.manifest.objects.push_back(makePinnedObject(
        "spin-axis-north",
        {0.0, 0.0, 5.0},
        0.35,
        3.0e-6,
        6500.0,
        4.0));
    scene.manifest.objects.push_back(makePinnedObject(
        "spin-axis-south",
        {0.0, 0.0, -5.0},
        0.35,
        -3.0e-6,
        3500.0,
        4.0));

    auto traveler = makePinnedObject(
        "geodesic-traveler",
        liveState->trail.front(),
        0.28,
        -4.0e-6,
        6200.0,
        7.5);
    traveler.pinned = false;
    traveler.velocity = scaledCoordinateVelocity(liveState->current, liveState->coordinateScale);
    traveler.coordinateTimeSeconds = liveState->current.position.t / physicsmade::common::kSpeedOfLight;
    traveler.properTimeSeconds = liveState->current.affineParameter;
    scene.manifest.objects.push_back(traveler);

    for (std::size_t index = 0; index < trailCount; ++index) {
        const double ageFraction = safeFraction(static_cast<double>(index), static_cast<double>(trailCount - 1));
        scene.manifest.objects.push_back(makePinnedObject(
            "geodesic-trail-" + std::to_string(index),
            liveState->trail[index],
            0.08 + (0.1 * (1.0 - ageFraction)),
            (ageFraction < 0.5) ? 2.0e-6 : -2.0e-6,
            700.0 + (4200.0 * (1.0 - ageFraction)),
            0.7 + (4.0 * (1.0 - ageFraction))));
    }

    scene.suggestedFollowIndex = 3;
    scene.step = [liveState](physicsmade::simulation::SimulationWorld& world, double dtSeconds) {
        const int substeps = std::max(1, static_cast<int>(std::lround(dtSeconds / liveState->integrator.options().affineStep)));
        for (int stepIndex = 0; stepIndex < substeps; ++stepIndex) {
            liveState->current = liveState->integrator.step(liveState->current);
        }

        for (std::size_t index = liveState->trail.size() - 1; index > 0; --index) {
            liveState->trail[index] = liveState->trail[index - 1];
        }
        liveState->trail.front() = liveState->coordinateScale * liveState->current.position.spatial();

        auto& objects = world.objects();
        auto& travelerObject = objects[3];
        travelerObject.position = liveState->trail.front();
        travelerObject.velocity = scaledCoordinateVelocity(liveState->current, liveState->coordinateScale);
        travelerObject.coordinateTimeSeconds = liveState->current.position.t / physicsmade::common::kSpeedOfLight;
        travelerObject.properTimeSeconds = liveState->current.affineParameter;

        for (std::size_t index = 0; index < liveState->trail.size(); ++index) {
            auto& trailObject = objects[4 + index];
            const double ageFraction = safeFraction(static_cast<double>(index), static_cast<double>(liveState->trail.size() - 1));
            trailObject.position = liveState->trail[index];
            trailObject.radius = 0.08 + (0.1 * (1.0 - ageFraction));
            trailObject.temperatureKelvin = 700.0 + (4200.0 * (1.0 - ageFraction));
            trailObject.emissiveIntensity = 0.7 + (4.0 * (1.0 - ageFraction));
            trailObject.coordinateTimeSeconds = travelerObject.coordinateTimeSeconds;
            trailObject.properTimeSeconds = travelerObject.properTimeSeconds;
        }
    };
    return scene;
}

physicsmade::io::SceneManifest buildCoupledSu2FermionScene() {
    physicsmade::field_theory::CoupledSu2FermionLattice lattice({14, 14, 6, 0.22, 0.55, 1.0, 0.85, 0.3});
    lattice.seedFermionGaussianPacket(1.0, {1.1, 1.0, 0.6}, 0.45, {1.0, 0.6, 0.25}, {0.2, 1.0, 0.5});
    lattice.seedGaugeStandingWave(0.08, {0.9, 0.4, 0.2}, {0.0, 1.0, 1.0});
    lattice.seedColorFlux(0.05, {1.0, 0.25, 0.4});
    lattice.step(0.0025, 160);

    const auto samples = lattice.sampleSites();
    double maxDensity = 0.0;
    double maxElectric = 0.0;
    for (const auto& sample : samples) {
        maxDensity = std::max(maxDensity, sample.density);
        maxElectric = std::max(maxElectric, sample.electricField.norm());
    }

    const physicsmade::math::Vector3 centerOffset{
        0.5 * lattice.spec().spacing * static_cast<double>(lattice.spec().sizeX - 1),
        0.5 * lattice.spec().spacing * static_cast<double>(lattice.spec().sizeY - 1),
        0.5 * lattice.spec().spacing * static_cast<double>(lattice.spec().sizeZ - 1),
    };

    physicsmade::io::SceneManifest manifest{};
    manifest.sceneName = "coupled-su2-fermion-state";
    manifest.kinematicsId = "newtonian";
    manifest.integratorId = "velocity-verlet";

    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& sample = samples[index];
        const double densityFraction = sample.density / std::max(maxDensity, 1.0e-12);
        const double electricFraction = sample.electricField.norm() / std::max(maxElectric, 1.0e-12);
        const auto centeredPosition = sample.position - centerOffset;

        if (densityFraction >= 0.06) {
            manifest.objects.push_back(makePinnedObject(
                "fermion-site-" + std::to_string(index),
                centeredPosition,
                0.08 + (0.18 * densityFraction),
                2.0e-6 * sample.current.x,
                500.0 + (4200.0 * densityFraction),
                1.2 + (5.0 * densityFraction)));
        }

        if (electricFraction >= 0.35) {
            manifest.objects.push_back(makePinnedObject(
                "gauge-site-" + std::to_string(index),
                centeredPosition + physicsmade::math::Vector3{0.06, 0.06, 0.06},
                0.05 + (0.12 * electricFraction),
                -2.0e-6 * sample.plaquetteTrace,
                800.0 + (2800.0 * (1.0 - sample.plaquetteTrace)),
                0.8 + (4.0 * electricFraction)));
        }
    }

    if (manifest.objects.empty() && !samples.empty()) {
        const auto& sample = *std::max_element(samples.begin(), samples.end(), [](const auto& left, const auto& right) {
            return left.density < right.density;
        });
        manifest.objects.push_back(makePinnedObject(
            "fermion-site-max",
            sample.position - centerOffset,
            0.2,
            0.0,
            4200.0,
            6.0));
    }

    return manifest;
}

physicsmade::runtime::GeneratedViewerScene buildCoupledLatticeLiveViewerScene() {
    auto liveState = std::make_shared<CoupledLatticeLiveSceneState>();
    liveState->lattice = physicsmade::field_theory::CoupledSu2FermionLattice({14, 14, 6, 0.22, 0.55, 1.0, 0.85, 0.3});
    liveState->lattice.seedFermionGaussianPacket(1.0, {1.1, 1.0, 0.6}, 0.45, {1.0, 0.6, 0.25}, {0.2, 1.0, 0.5});
    liveState->lattice.seedGaugeStandingWave(0.08, {0.9, 0.4, 0.2}, {0.0, 1.0, 1.0});
    liveState->lattice.seedColorFlux(0.05, {1.0, 0.25, 0.4});
    liveState->centerOffset = {
        0.5 * liveState->lattice.spec().spacing * static_cast<double>(liveState->lattice.spec().sizeX - 1),
        0.5 * liveState->lattice.spec().spacing * static_cast<double>(liveState->lattice.spec().sizeY - 1),
        0.5 * liveState->lattice.spec().spacing * static_cast<double>(liveState->lattice.spec().sizeZ - 1),
    };

    for (std::size_t z = 0; z < liveState->lattice.spec().sizeZ; z += 2) {
        for (std::size_t y = 0; y < liveState->lattice.spec().sizeY; y += 2) {
            for (std::size_t x = 0; x < liveState->lattice.spec().sizeX; x += 2) {
                liveState->sampleIndices.push_back(
                    x + (liveState->lattice.spec().sizeX * (y + (liveState->lattice.spec().sizeY * z))));
            }
        }
    }

    const auto initialSamples = liveState->lattice.sampleSites();
    physicsmade::runtime::GeneratedViewerScene scene{};
    scene.manifest.sceneName = "coupled-su2-fermion-state";
    scene.manifest.kinematicsId = "newtonian";
    scene.manifest.integratorId = "velocity-verlet";
    scene.manifest.previewDtSeconds = 0.0025;

    for (std::size_t sampleOrdinal = 0; sampleOrdinal < liveState->sampleIndices.size(); ++sampleOrdinal) {
        const auto& sample = initialSamples[liveState->sampleIndices[sampleOrdinal]];
        const auto centeredPosition = sample.position - liveState->centerOffset;
        scene.manifest.objects.push_back(makePinnedObject(
            "fermion-sample-" + std::to_string(sampleOrdinal),
            centeredPosition,
            0.12,
            2.0e-6 * sample.current.x,
            1800.0,
            2.5));
        scene.manifest.objects.push_back(makePinnedObject(
            "gauge-sample-" + std::to_string(sampleOrdinal),
            centeredPosition + physicsmade::math::Vector3{0.05, 0.05, 0.05},
            0.08,
            -2.0e-6 * sample.plaquetteTrace,
            1400.0,
            1.8));
    }

    scene.step = [liveState](physicsmade::simulation::SimulationWorld& world, double dtSeconds) {
        const int iterations = std::max(1, static_cast<int>(std::lround(dtSeconds / 0.0025)));
        liveState->lattice.step(0.0025, iterations);
        const auto samples = liveState->lattice.sampleSites();

        double maxDensity = 0.0;
        double maxElectric = 0.0;
        for (const auto sampleIndex : liveState->sampleIndices) {
            maxDensity = std::max(maxDensity, samples[sampleIndex].density);
            maxElectric = std::max(maxElectric, samples[sampleIndex].electricField.norm());
        }

        auto& objects = world.objects();
        for (std::size_t sampleOrdinal = 0; sampleOrdinal < liveState->sampleIndices.size(); ++sampleOrdinal) {
            const auto& sample = samples[liveState->sampleIndices[sampleOrdinal]];
            const double densityFraction = safeFraction(sample.density, maxDensity);
            const double electricFraction = safeFraction(sample.electricField.norm(), maxElectric);
            const auto centeredPosition = sample.position - liveState->centerOffset;

            auto& fermionObject = objects[2 * sampleOrdinal];
            fermionObject.position = centeredPosition;
            fermionObject.radius = 0.05 + (0.18 * densityFraction);
            fermionObject.charge = 2.0e-6 * sample.current.x;
            fermionObject.temperatureKelvin = 450.0 + (4500.0 * densityFraction);
            fermionObject.emissiveIntensity = 0.6 + (5.4 * densityFraction);
            fermionObject.coordinateTimeSeconds += dtSeconds;
            fermionObject.properTimeSeconds += dtSeconds;

            auto& gaugeObject = objects[(2 * sampleOrdinal) + 1];
            const auto electricOffset = 0.03 * sample.electricField.normalized();
            gaugeObject.position = centeredPosition + physicsmade::math::Vector3{0.05, 0.05, 0.05} + electricOffset;
            gaugeObject.radius = 0.04 + (0.14 * electricFraction);
            gaugeObject.charge = -2.0e-6 * sample.plaquetteTrace;
            gaugeObject.temperatureKelvin = 850.0 + (2800.0 * (1.0 - sample.plaquetteTrace));
            gaugeObject.emissiveIntensity = 0.5 + (4.5 * electricFraction);
            gaugeObject.coordinateTimeSeconds += dtSeconds;
            gaugeObject.properTimeSeconds += dtSeconds;
        }
    };
    return scene;
}

}  // namespace

namespace physicsmade::runtime {

std::optional<io::SceneManifest> buildGeneratedScene(std::string_view id) {
    if (id == "quantum-chemistry-live") {
        return buildQuantumChemistryScene();
    }

    if (id == "kerr-geodesic") {
        return buildKerrGeodesicScene();
    }

    if (id == "coupled-su2-fermion-state") {
        return buildCoupledSu2FermionScene();
    }

    return std::nullopt;
}

std::optional<GeneratedViewerScene> buildGeneratedViewerScene(std::string_view id) {
    if (id == "quantum-chemistry-live") {
        return buildQuantumChemistryLiveViewerScene();
    }

    if (id == "kerr-geodesic") {
        return buildKerrLiveViewerScene();
    }

    if (id == "coupled-su2-fermion-state") {
        return buildCoupledLatticeLiveViewerScene();
    }

    return std::nullopt;
}

}  // namespace physicsmade::runtime
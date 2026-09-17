#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "physicsmade/math/vector3.hpp"

namespace physicsmade::quantum_chemistry {

struct Nucleus {
    std::string symbol{"H"};
    int atomicNumber{1};
    double charge{1.0};
    math::Vector3 position{};
};

struct Molecule {
    std::vector<Nucleus> nuclei{};
    int electronCount{0};
};

struct PrimitiveGaussian {
    double exponent{1.0};
    double coefficient{1.0};
};

struct ContractedGaussianOrbital {
    std::size_t nucleusIndex{0};
    math::Vector3 center{};
    std::vector<PrimitiveGaussian> primitives{};
};

struct RestrictedHartreeFockOptions {
    int maxIterations{64};
    double energyTolerance{1.0e-10};
    double densityTolerance{1.0e-8};
};

struct RestrictedHartreeFockResult {
    bool converged{false};
    int iterations{0};
    std::size_t basisCount{0};
    int occupiedOrbitals{0};
    double electronicEnergy{0.0};
    double nuclearRepulsionEnergy{0.0};
    double totalEnergy{0.0};
    std::vector<double> orbitalEnergies{};
    std::vector<double> densityMatrix{};
    std::vector<double> coefficientMatrix{};
};

struct RestrictedHartreeFockState {
    Molecule molecule{};
    RestrictedHartreeFockOptions options{};
    std::vector<ContractedGaussianOrbital> basis{};
    int occupiedOrbitals{0};
    double nuclearRepulsionEnergy{0.0};
    std::vector<double> overlapMatrix{};
    std::vector<double> coreHamiltonianMatrix{};
    std::vector<double> electronRepulsionIntegrals{};
    std::vector<double> orthogonalizerMatrix{};
    std::vector<double> densityMatrix{};
    std::vector<double> coefficientMatrix{};
    std::vector<double> orbitalEnergies{};
    int iterations{0};
    double electronicEnergy{0.0};
    double totalEnergy{0.0};
    double previousElectronicEnergy{0.0};
    double energyDelta{0.0};
    double densityDelta{0.0};
    bool converged{false};
};

struct UnrestrictedHartreeFockOptions {
    int maxIterations{64};
    int alphaElectronCount{-1};
    int betaElectronCount{-1};
    double energyTolerance{1.0e-10};
    double densityTolerance{1.0e-8};
};

struct UnrestrictedHartreeFockResult {
    bool converged{false};
    int iterations{0};
    std::size_t basisCount{0};
    int alphaElectronCount{0};
    int betaElectronCount{0};
    double electronicEnergy{0.0};
    double nuclearRepulsionEnergy{0.0};
    double totalEnergy{0.0};
    std::vector<double> alphaOrbitalEnergies{};
    std::vector<double> betaOrbitalEnergies{};
    std::vector<double> alphaDensityMatrix{};
    std::vector<double> betaDensityMatrix{};
    std::vector<double> alphaCoefficientMatrix{};
    std::vector<double> betaCoefficientMatrix{};
};

struct RestrictedMollerPlesset2Options {
    RestrictedHartreeFockOptions hartreeFock{};
    double denominatorTolerance{1.0e-10};
};

struct RestrictedMollerPlesset2Result {
    bool scfConverged{false};
    int scfIterations{0};
    std::size_t basisCount{0};
    int occupiedOrbitals{0};
    double hartreeFockElectronicEnergy{0.0};
    double nuclearRepulsionEnergy{0.0};
    double hartreeFockTotalEnergy{0.0};
    double correlationEnergy{0.0};
    double totalEnergy{0.0};
    std::vector<double> orbitalEnergies{};
};

Molecule makeHydrogenMolecule(double bondLengthBohr = 1.4);
Molecule makeHeliumHydrideCation(double bondLengthBohr = 1.4632);
Molecule makeHydrogenMolecularCation(double bondLengthBohr = 2.0);

std::vector<ContractedGaussianOrbital> buildMinimalSto3gBasis(const Molecule& molecule);
double evaluateContractedGaussian(const ContractedGaussianOrbital& orbital, const math::Vector3& position);
double evaluateMolecularOrbital(
    const std::vector<ContractedGaussianOrbital>& basis,
    const std::vector<double>& coefficientMatrix,
    std::size_t orbitalIndex,
    const math::Vector3& position);
double evaluateElectronDensity(
    const std::vector<ContractedGaussianOrbital>& basis,
    const std::vector<double>& densityMatrix,
    const math::Vector3& position);
double nuclearRepulsionEnergy(const Molecule& molecule);

RestrictedHartreeFockState initializeRestrictedHartreeFockState(
    const Molecule& molecule,
    const RestrictedHartreeFockOptions& options = {});

bool stepRestrictedHartreeFock(RestrictedHartreeFockState& state);

RestrictedHartreeFockResult solveRestrictedHartreeFock(
    const Molecule& molecule,
    const RestrictedHartreeFockOptions& options = {});

UnrestrictedHartreeFockResult solveUnrestrictedHartreeFock(
    const Molecule& molecule,
    const UnrestrictedHartreeFockOptions& options = {});

RestrictedMollerPlesset2Result solveRestrictedMollerPlesset2(
    const Molecule& molecule,
    const RestrictedMollerPlesset2Options& options = {});

}  // namespace physicsmade::quantum_chemistry
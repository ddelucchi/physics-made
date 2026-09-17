#include "physicsmade/quantum_chemistry/restricted_hartree_fock.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

using Matrix = std::vector<double>;

constexpr double kPi = 3.14159265358979323846;
constexpr double kIntegralEpsilon = 1.0e-12;

std::size_t matrixIndex(std::size_t dimension, std::size_t row, std::size_t column) {
    return (row * dimension) + column;
}

double& matrixEntry(Matrix& matrix, std::size_t dimension, std::size_t row, std::size_t column) {
    return matrix[matrixIndex(dimension, row, column)];
}

double matrixEntry(const Matrix& matrix, std::size_t dimension, std::size_t row, std::size_t column) {
    return matrix[matrixIndex(dimension, row, column)];
}

Matrix zeroMatrix(std::size_t dimension) {
    return Matrix(dimension * dimension, 0.0);
}

Matrix identityMatrix(std::size_t dimension) {
    auto matrix = zeroMatrix(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        matrixEntry(matrix, dimension, index, index) = 1.0;
    }
    return matrix;
}

Matrix transpose(const Matrix& matrix, std::size_t dimension) {
    auto result = zeroMatrix(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            matrixEntry(result, dimension, column, row) = matrixEntry(matrix, dimension, row, column);
        }
    }
    return result;
}

Matrix multiply(const Matrix& left, const Matrix& right, std::size_t dimension) {
    auto result = zeroMatrix(dimension);
    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < dimension; ++inner) {
                sum += matrixEntry(left, dimension, row, inner) * matrixEntry(right, dimension, inner, column);
            }
            matrixEntry(result, dimension, row, column) = sum;
        }
    }
    return result;
}

double maxAbsoluteDifference(const Matrix& left, const Matrix& right) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

double boysFunction0(double argument) {
    if (argument <= 1.0e-10) {
        return 1.0 - (argument / 3.0);
    }

    const double root = std::sqrt(argument);
    return 0.5 * std::sqrt(kPi / argument) * std::erf(root);
}

double primitiveNormalization(double exponent) {
    return std::pow((2.0 * exponent) / kPi, 0.75);
}

double primitiveGaussianValue(
    double exponent,
    const physicsmade::math::Vector3& center,
    const physicsmade::math::Vector3& position) {
    return primitiveNormalization(exponent) * std::exp(-exponent * (position - center).normSquared());
}

physicsmade::math::Vector3 gaussianProductCenter(
    double leftExponent,
    const physicsmade::math::Vector3& leftCenter,
    double rightExponent,
    const physicsmade::math::Vector3& rightCenter) {
    const double inverse = 1.0 / (leftExponent + rightExponent);
    return ((leftCenter * leftExponent) + (rightCenter * rightExponent)) * inverse;
}

double primitiveOverlap(
    double leftExponent,
    const physicsmade::math::Vector3& leftCenter,
    double rightExponent,
    const physicsmade::math::Vector3& rightCenter) {
    const double sumExponent = leftExponent + rightExponent;
    const double reducedExponent = (leftExponent * rightExponent) / sumExponent;
    const double distanceSquared = (leftCenter - rightCenter).normSquared();
    return primitiveNormalization(leftExponent) *
           primitiveNormalization(rightExponent) *
           std::pow(kPi / sumExponent, 1.5) *
           std::exp(-reducedExponent * distanceSquared);
}

double primitiveKinetic(
    double leftExponent,
    const physicsmade::math::Vector3& leftCenter,
    double rightExponent,
    const physicsmade::math::Vector3& rightCenter) {
    const double sumExponent = leftExponent + rightExponent;
    const double reducedExponent = (leftExponent * rightExponent) / sumExponent;
    const double distanceSquared = (leftCenter - rightCenter).normSquared();
    const double overlap = primitiveOverlap(leftExponent, leftCenter, rightExponent, rightCenter);
    return reducedExponent * (3.0 - (2.0 * reducedExponent * distanceSquared)) * overlap;
}

double primitiveNuclearAttraction(
    double leftExponent,
    const physicsmade::math::Vector3& leftCenter,
    double rightExponent,
    const physicsmade::math::Vector3& rightCenter,
    const physicsmade::quantum_chemistry::Nucleus& nucleus) {
    const double sumExponent = leftExponent + rightExponent;
    const double reducedExponent = (leftExponent * rightExponent) / sumExponent;
    const double distanceSquared = (leftCenter - rightCenter).normSquared();
    const auto productCenter = gaussianProductCenter(leftExponent, leftCenter, rightExponent, rightCenter);
    const double centerDistanceSquared = (productCenter - nucleus.position).normSquared();
    const double prefactor = -nucleus.charge * (2.0 * kPi / sumExponent);
    return prefactor *
           primitiveNormalization(leftExponent) *
           primitiveNormalization(rightExponent) *
           std::exp(-reducedExponent * distanceSquared) *
           boysFunction0(sumExponent * centerDistanceSquared);
}

double primitiveElectronRepulsion(
    double leftExponentA,
    const physicsmade::math::Vector3& leftCenterA,
    double rightExponentA,
    const physicsmade::math::Vector3& rightCenterA,
    double leftExponentB,
    const physicsmade::math::Vector3& leftCenterB,
    double rightExponentB,
    const physicsmade::math::Vector3& rightCenterB) {
    const double sumExponentA = leftExponentA + rightExponentA;
    const double sumExponentB = leftExponentB + rightExponentB;
    const double reducedExponentA = (leftExponentA * rightExponentA) / sumExponentA;
    const double reducedExponentB = (leftExponentB * rightExponentB) / sumExponentB;
    const double distanceSquaredA = (leftCenterA - rightCenterA).normSquared();
    const double distanceSquaredB = (leftCenterB - rightCenterB).normSquared();
    const auto productCenterA = gaussianProductCenter(leftExponentA, leftCenterA, rightExponentA, rightCenterA);
    const auto productCenterB = gaussianProductCenter(leftExponentB, leftCenterB, rightExponentB, rightCenterB);
    const double productDistanceSquared = (productCenterA - productCenterB).normSquared();
    const double rho = (sumExponentA * sumExponentB) / (sumExponentA + sumExponentB);
    const double prefactor = 2.0 * std::pow(kPi, 2.5) /
                             (sumExponentA * sumExponentB * std::sqrt(sumExponentA + sumExponentB));
    return prefactor *
           primitiveNormalization(leftExponentA) *
           primitiveNormalization(rightExponentA) *
           primitiveNormalization(leftExponentB) *
           primitiveNormalization(rightExponentB) *
           std::exp((-reducedExponentA * distanceSquaredA) - (reducedExponentB * distanceSquaredB)) *
           boysFunction0(rho * productDistanceSquared);
}

double contractedOverlap(
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& left,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& right) {
    double value = 0.0;
    for (const auto& leftPrimitive : left.primitives) {
        for (const auto& rightPrimitive : right.primitives) {
            value += leftPrimitive.coefficient *
                     rightPrimitive.coefficient *
                     primitiveOverlap(leftPrimitive.exponent, left.center, rightPrimitive.exponent, right.center);
        }
    }
    return value;
}

double contractedKinetic(
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& left,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& right) {
    double value = 0.0;
    for (const auto& leftPrimitive : left.primitives) {
        for (const auto& rightPrimitive : right.primitives) {
            value += leftPrimitive.coefficient *
                     rightPrimitive.coefficient *
                     primitiveKinetic(leftPrimitive.exponent, left.center, rightPrimitive.exponent, right.center);
        }
    }
    return value;
}

double contractedNuclearAttraction(
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& left,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& right,
    const physicsmade::quantum_chemistry::Molecule& molecule) {
    double value = 0.0;
    for (const auto& nucleus : molecule.nuclei) {
        for (const auto& leftPrimitive : left.primitives) {
            for (const auto& rightPrimitive : right.primitives) {
                value += leftPrimitive.coefficient *
                         rightPrimitive.coefficient *
                         primitiveNuclearAttraction(
                             leftPrimitive.exponent,
                             left.center,
                             rightPrimitive.exponent,
                             right.center,
                             nucleus);
            }
        }
    }
    return value;
}

double contractedElectronRepulsion(
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& leftA,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& rightA,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& leftB,
    const physicsmade::quantum_chemistry::ContractedGaussianOrbital& rightB) {
    double value = 0.0;
    for (const auto& primitiveLeftA : leftA.primitives) {
        for (const auto& primitiveRightA : rightA.primitives) {
            for (const auto& primitiveLeftB : leftB.primitives) {
                for (const auto& primitiveRightB : rightB.primitives) {
                    value += primitiveLeftA.coefficient *
                             primitiveRightA.coefficient *
                             primitiveLeftB.coefficient *
                             primitiveRightB.coefficient *
                             primitiveElectronRepulsion(
                                 primitiveLeftA.exponent,
                                 leftA.center,
                                 primitiveRightA.exponent,
                                 rightA.center,
                                 primitiveLeftB.exponent,
                                 leftB.center,
                                 primitiveRightB.exponent,
                                 rightB.center);
                }
            }
        }
    }
    return value;
}

struct SymmetricEigenDecomposition {
    std::vector<double> eigenvalues{};
    Matrix eigenvectors{};
};

SymmetricEigenDecomposition diagonalizeSymmetric(const Matrix& input, std::size_t dimension) {
    auto matrix = input;
    auto eigenvectors = identityMatrix(dimension);

    const std::size_t maxIterations = 64U * dimension * dimension;
    for (std::size_t iteration = 0; iteration < maxIterations; ++iteration) {
        double maxOffDiagonal = 0.0;
        std::size_t pivotRow = 0;
        std::size_t pivotColumn = 0;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = row + 1; column < dimension; ++column) {
                const double value = std::abs(matrixEntry(matrix, dimension, row, column));
                if (value > maxOffDiagonal) {
                    maxOffDiagonal = value;
                    pivotRow = row;
                    pivotColumn = column;
                }
            }
        }

        if (maxOffDiagonal <= 1.0e-12) {
            break;
        }

        const double diagonalLeft = matrixEntry(matrix, dimension, pivotRow, pivotRow);
        const double diagonalRight = matrixEntry(matrix, dimension, pivotColumn, pivotColumn);
        const double offDiagonal = matrixEntry(matrix, dimension, pivotRow, pivotColumn);
        const double angle = 0.5 * std::atan2(2.0 * offDiagonal, diagonalRight - diagonalLeft);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        for (std::size_t index = 0; index < dimension; ++index) {
            if (index == pivotRow || index == pivotColumn) {
                continue;
            }

            const double leftValue = matrixEntry(matrix, dimension, index, pivotRow);
            const double rightValue = matrixEntry(matrix, dimension, index, pivotColumn);
            const double rotatedLeft = (cosine * leftValue) - (sine * rightValue);
            const double rotatedRight = (sine * leftValue) + (cosine * rightValue);
            matrixEntry(matrix, dimension, index, pivotRow) = rotatedLeft;
            matrixEntry(matrix, dimension, pivotRow, index) = rotatedLeft;
            matrixEntry(matrix, dimension, index, pivotColumn) = rotatedRight;
            matrixEntry(matrix, dimension, pivotColumn, index) = rotatedRight;
        }

        const double rotatedDiagonalLeft =
            (cosine * cosine * diagonalLeft) - (2.0 * sine * cosine * offDiagonal) + (sine * sine * diagonalRight);
        const double rotatedDiagonalRight =
            (sine * sine * diagonalLeft) + (2.0 * sine * cosine * offDiagonal) + (cosine * cosine * diagonalRight);
        matrixEntry(matrix, dimension, pivotRow, pivotRow) = rotatedDiagonalLeft;
        matrixEntry(matrix, dimension, pivotColumn, pivotColumn) = rotatedDiagonalRight;
        matrixEntry(matrix, dimension, pivotRow, pivotColumn) = 0.0;
        matrixEntry(matrix, dimension, pivotColumn, pivotRow) = 0.0;

        for (std::size_t row = 0; row < dimension; ++row) {
            const double leftValue = matrixEntry(eigenvectors, dimension, row, pivotRow);
            const double rightValue = matrixEntry(eigenvectors, dimension, row, pivotColumn);
            matrixEntry(eigenvectors, dimension, row, pivotRow) = (cosine * leftValue) - (sine * rightValue);
            matrixEntry(eigenvectors, dimension, row, pivotColumn) = (sine * leftValue) + (cosine * rightValue);
        }
    }

    std::vector<std::pair<double, std::size_t>> ordering;
    ordering.reserve(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        ordering.push_back({matrixEntry(matrix, dimension, index, index), index});
    }
    std::sort(ordering.begin(), ordering.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });

    SymmetricEigenDecomposition result{};
    result.eigenvalues.resize(dimension);
    result.eigenvectors = zeroMatrix(dimension);
    for (std::size_t newColumn = 0; newColumn < dimension; ++newColumn) {
        result.eigenvalues[newColumn] = ordering[newColumn].first;
        const std::size_t oldColumn = ordering[newColumn].second;
        for (std::size_t row = 0; row < dimension; ++row) {
            matrixEntry(result.eigenvectors, dimension, row, newColumn) =
                matrixEntry(eigenvectors, dimension, row, oldColumn);
        }
    }

    return result;
}

Matrix symmetricInverseSquareRoot(const Matrix& overlap, std::size_t dimension) {
    const auto decomposition = diagonalizeSymmetric(overlap, dimension);
    auto diagonal = zeroMatrix(dimension);
    for (std::size_t index = 0; index < dimension; ++index) {
        if (decomposition.eigenvalues[index] <= kIntegralEpsilon) {
            throw std::runtime_error("overlap matrix is singular or ill-conditioned");
        }
        matrixEntry(diagonal, dimension, index, index) = 1.0 / std::sqrt(decomposition.eigenvalues[index]);
    }

    return multiply(
        multiply(decomposition.eigenvectors, diagonal, dimension),
        transpose(decomposition.eigenvectors, dimension),
        dimension);
}

std::vector<physicsmade::quantum_chemistry::PrimitiveGaussian> sto3gPrimitivesForAtomicNumber(int atomicNumber) {
    if (atomicNumber == 1) {
        return {
            {3.425250914, 0.1543289673},
            {0.6239137298, 0.5353281423},
            {0.1688554040, 0.4446345422},
        };
    }

    if (atomicNumber == 2) {
        return {
            {6.362421394, 0.1543289673},
            {1.158923000, 0.5353281423},
            {0.3136497915, 0.4446345422},
        };
    }

    throw std::runtime_error("minimal STO-3G support currently covers only H and He nuclei");
}

std::size_t eriIndex(std::size_t dimension, std::size_t mu, std::size_t nu, std::size_t lambda, std::size_t sigma) {
    return (((mu * dimension + nu) * dimension + lambda) * dimension) + sigma;
}

struct MolecularIntegralSet {
    std::vector<physicsmade::quantum_chemistry::ContractedGaussianOrbital> basis{};
    Matrix overlap{};
    Matrix coreHamiltonian{};
    std::vector<double> electronRepulsion{};
    double nuclearRepulsion{0.0};
};

MolecularIntegralSet buildIntegralSet(const physicsmade::quantum_chemistry::Molecule& molecule) {
    MolecularIntegralSet integrals{};
    integrals.basis = physicsmade::quantum_chemistry::buildMinimalSto3gBasis(molecule);
    const std::size_t dimension = integrals.basis.size();
    if (dimension == 0) {
        throw std::runtime_error("Hartree-Fock requires at least one basis function");
    }

    auto overlap = zeroMatrix(dimension);
    auto kinetic = zeroMatrix(dimension);
    auto nuclearAttraction = zeroMatrix(dimension);
    auto coreHamiltonian = zeroMatrix(dimension);
    std::vector<double> electronRepulsion(dimension * dimension * dimension * dimension, 0.0);

    for (std::size_t row = 0; row < dimension; ++row) {
        for (std::size_t column = 0; column < dimension; ++column) {
            const double overlapIntegral = contractedOverlap(integrals.basis[row], integrals.basis[column]);
            const double kineticIntegral = contractedKinetic(integrals.basis[row], integrals.basis[column]);
            const double attractionIntegral = contractedNuclearAttraction(integrals.basis[row], integrals.basis[column], molecule);
            matrixEntry(overlap, dimension, row, column) = overlapIntegral;
            matrixEntry(kinetic, dimension, row, column) = kineticIntegral;
            matrixEntry(nuclearAttraction, dimension, row, column) = attractionIntegral;
            matrixEntry(coreHamiltonian, dimension, row, column) = kineticIntegral + attractionIntegral;

            for (std::size_t lambda = 0; lambda < dimension; ++lambda) {
                for (std::size_t sigma = 0; sigma < dimension; ++sigma) {
                    electronRepulsion[eriIndex(dimension, row, column, lambda, sigma)] =
                        contractedElectronRepulsion(
                            integrals.basis[row],
                            integrals.basis[column],
                            integrals.basis[lambda],
                            integrals.basis[sigma]);
                }
            }
        }
    }

    integrals.overlap = std::move(overlap);
    integrals.coreHamiltonian = std::move(coreHamiltonian);
    integrals.electronRepulsion = std::move(electronRepulsion);
    integrals.nuclearRepulsion = physicsmade::quantum_chemistry::nuclearRepulsionEnergy(molecule);
    return integrals;
}

void validateSquareMatrixSize(const Matrix& matrix, std::size_t dimension, const char* label) {
    if (matrix.size() != (dimension * dimension)) {
        throw std::runtime_error(std::string(label) + " size does not match the basis dimension");
    }
}

Matrix buildRestrictedFockMatrix(
    const Matrix& coreHamiltonian,
    const std::vector<double>& electronRepulsion,
    const Matrix& density,
    std::size_t dimension) {
    auto fock = coreHamiltonian;
    for (std::size_t mu = 0; mu < dimension; ++mu) {
        for (std::size_t nu = 0; nu < dimension; ++nu) {
            double interaction = 0.0;
            for (std::size_t lambda = 0; lambda < dimension; ++lambda) {
                for (std::size_t sigma = 0; sigma < dimension; ++sigma) {
                    interaction += matrixEntry(density, dimension, lambda, sigma) *
                                   (electronRepulsion[eriIndex(dimension, mu, nu, lambda, sigma)] -
                                    0.5 * electronRepulsion[eriIndex(dimension, mu, lambda, nu, sigma)]);
                }
            }
            matrixEntry(fock, dimension, mu, nu) += interaction;
        }
    }
    return fock;
}

physicsmade::quantum_chemistry::RestrictedHartreeFockResult makeRestrictedHartreeFockResult(
    const physicsmade::quantum_chemistry::RestrictedHartreeFockState& state) {
    physicsmade::quantum_chemistry::RestrictedHartreeFockResult result{};
    result.converged = state.converged;
    result.iterations = state.iterations;
    result.basisCount = state.basis.size();
    result.occupiedOrbitals = state.occupiedOrbitals;
    result.electronicEnergy = state.electronicEnergy;
    result.nuclearRepulsionEnergy = state.nuclearRepulsionEnergy;
    result.totalEnergy = state.totalEnergy;
    result.orbitalEnergies = state.orbitalEnergies;
    result.densityMatrix = state.densityMatrix;
    result.coefficientMatrix = state.coefficientMatrix;
    return result;
}

double molecularOrbitalIntegral(
    const MolecularIntegralSet& integrals,
    const Matrix& coefficients,
    std::size_t leftOccupied,
    std::size_t rightOccupied,
    std::size_t leftVirtual,
    std::size_t rightVirtual) {
    const std::size_t dimension = integrals.basis.size();
    double value = 0.0;
    for (std::size_t mu = 0; mu < dimension; ++mu) {
        for (std::size_t nu = 0; nu < dimension; ++nu) {
            for (std::size_t lambda = 0; lambda < dimension; ++lambda) {
                for (std::size_t sigma = 0; sigma < dimension; ++sigma) {
                    value +=
                        matrixEntry(coefficients, dimension, mu, leftOccupied) *
                        matrixEntry(coefficients, dimension, nu, rightOccupied) *
                        matrixEntry(coefficients, dimension, lambda, leftVirtual) *
                        matrixEntry(coefficients, dimension, sigma, rightVirtual) *
                        integrals.electronRepulsion[eriIndex(dimension, mu, nu, lambda, sigma)];
                }
            }
        }
    }
    return value;
}

std::pair<int, int> determineSpinPopulation(
    int electronCount,
    int requestedAlpha,
    int requestedBeta) {
    const int alpha = (requestedAlpha >= 0) ? requestedAlpha : ((electronCount + 1) / 2);
    const int beta = (requestedBeta >= 0) ? requestedBeta : (electronCount / 2);
    if (alpha < 0 || beta < 0 || (alpha + beta) != electronCount) {
        throw std::runtime_error("unrestricted Hartree-Fock spin populations must be nonnegative and sum to the molecular electron count");
    }
    return {alpha, beta};
}

}  // namespace

namespace physicsmade::quantum_chemistry {

Molecule makeHydrogenMolecule(double bondLengthBohr) {
    const double halfBond = 0.5 * bondLengthBohr;
    return {
        {
            {"H", 1, 1.0, {-halfBond, 0.0, 0.0}},
            {"H", 1, 1.0, {halfBond, 0.0, 0.0}},
        },
        2,
    };
}

Molecule makeHeliumHydrideCation(double bondLengthBohr) {
    const double halfBond = 0.5 * bondLengthBohr;
    return {
        {
            {"He", 2, 2.0, {-halfBond, 0.0, 0.0}},
            {"H", 1, 1.0, {halfBond, 0.0, 0.0}},
        },
        2,
    };
}

Molecule makeHydrogenMolecularCation(double bondLengthBohr) {
    const double halfBond = 0.5 * bondLengthBohr;
    return {
        {
            {"H", 1, 1.0, {-halfBond, 0.0, 0.0}},
            {"H", 1, 1.0, {halfBond, 0.0, 0.0}},
        },
        1,
    };
}

std::vector<ContractedGaussianOrbital> buildMinimalSto3gBasis(const Molecule& molecule) {
    std::vector<ContractedGaussianOrbital> basis;
    basis.reserve(molecule.nuclei.size());
    for (std::size_t nucleusIndex = 0; nucleusIndex < molecule.nuclei.size(); ++nucleusIndex) {
        const auto& nucleus = molecule.nuclei[nucleusIndex];
        basis.push_back({
            nucleusIndex,
            nucleus.position,
            sto3gPrimitivesForAtomicNumber(nucleus.atomicNumber),
        });
    }
    return basis;
}

double evaluateContractedGaussian(const ContractedGaussianOrbital& orbital, const math::Vector3& position) {
    double value = 0.0;
    for (const auto& primitive : orbital.primitives) {
        value += primitive.coefficient * primitiveGaussianValue(primitive.exponent, orbital.center, position);
    }
    return value;
}

double evaluateMolecularOrbital(
    const std::vector<ContractedGaussianOrbital>& basis,
    const std::vector<double>& coefficientMatrix,
    std::size_t orbitalIndex,
    const math::Vector3& position) {
    const std::size_t dimension = basis.size();
    if (orbitalIndex >= dimension) {
        throw std::runtime_error("molecular orbital index exceeds the basis dimension");
    }

    validateSquareMatrixSize(coefficientMatrix, dimension, "coefficient matrix");
    double value = 0.0;
    for (std::size_t basisIndex = 0; basisIndex < dimension; ++basisIndex) {
        value += matrixEntry(coefficientMatrix, dimension, basisIndex, orbitalIndex) *
                 evaluateContractedGaussian(basis[basisIndex], position);
    }
    return value;
}

double evaluateElectronDensity(
    const std::vector<ContractedGaussianOrbital>& basis,
    const std::vector<double>& densityMatrix,
    const math::Vector3& position) {
    const std::size_t dimension = basis.size();
    validateSquareMatrixSize(densityMatrix, dimension, "density matrix");

    std::vector<double> basisValues(dimension, 0.0);
    for (std::size_t basisIndex = 0; basisIndex < dimension; ++basisIndex) {
        basisValues[basisIndex] = evaluateContractedGaussian(basis[basisIndex], position);
    }

    double density = 0.0;
    for (std::size_t mu = 0; mu < dimension; ++mu) {
        for (std::size_t nu = 0; nu < dimension; ++nu) {
            density += matrixEntry(densityMatrix, dimension, mu, nu) * basisValues[mu] * basisValues[nu];
        }
    }
    return std::max(density, 0.0);
}

double nuclearRepulsionEnergy(const Molecule& molecule) {
    double energy = 0.0;
    for (std::size_t left = 0; left < molecule.nuclei.size(); ++left) {
        for (std::size_t right = left + 1; right < molecule.nuclei.size(); ++right) {
            const double distance = std::max((molecule.nuclei[right].position - molecule.nuclei[left].position).norm(), kIntegralEpsilon);
            energy += (molecule.nuclei[left].charge * molecule.nuclei[right].charge) / distance;
        }
    }
    return energy;
}

RestrictedHartreeFockState initializeRestrictedHartreeFockState(
    const Molecule& molecule,
    const RestrictedHartreeFockOptions& options) {
    if (molecule.electronCount < 0 || (molecule.electronCount % 2) != 0) {
        throw std::runtime_error("restricted Hartree-Fock requires a nonnegative, closed-shell even electron count");
    }

    const auto integrals = buildIntegralSet(molecule);
    const std::size_t dimension = integrals.basis.size();
    const int occupiedOrbitals = molecule.electronCount / 2;
    if (occupiedOrbitals > static_cast<int>(dimension)) {
        throw std::runtime_error("electron count exceeds the supported minimal basis dimension");
    }

    RestrictedHartreeFockState state{};
    state.molecule = molecule;
    state.options = options;
    state.basis = integrals.basis;
    state.occupiedOrbitals = occupiedOrbitals;
    state.nuclearRepulsionEnergy = integrals.nuclearRepulsion;
    state.overlapMatrix = integrals.overlap;
    state.coreHamiltonianMatrix = integrals.coreHamiltonian;
    state.electronRepulsionIntegrals = integrals.electronRepulsion;
    state.orthogonalizerMatrix = symmetricInverseSquareRoot(state.overlapMatrix, dimension);
    state.densityMatrix = zeroMatrix(dimension);
    state.coefficientMatrix = zeroMatrix(dimension);
    state.orbitalEnergies.assign(dimension, 0.0);
    state.totalEnergy = state.nuclearRepulsionEnergy;
    return state;
}

bool stepRestrictedHartreeFock(RestrictedHartreeFockState& state) {
    if (state.converged || state.iterations >= state.options.maxIterations) {
        return state.converged;
    }

    const std::size_t dimension = state.basis.size();
    validateSquareMatrixSize(state.overlapMatrix, dimension, "overlap matrix");
    validateSquareMatrixSize(state.coreHamiltonianMatrix, dimension, "core Hamiltonian matrix");
    validateSquareMatrixSize(state.orthogonalizerMatrix, dimension, "orthogonalizer matrix");
    validateSquareMatrixSize(state.densityMatrix, dimension, "density matrix");
    validateSquareMatrixSize(state.coefficientMatrix, dimension, "coefficient matrix");
    if (state.orbitalEnergies.size() != dimension) {
        throw std::runtime_error("orbital energy count does not match the basis dimension");
    }
    if (state.electronRepulsionIntegrals.size() != (dimension * dimension * dimension * dimension)) {
        throw std::runtime_error("electron repulsion integral count does not match the basis dimension");
    }

    const auto fock = buildRestrictedFockMatrix(
        state.coreHamiltonianMatrix,
        state.electronRepulsionIntegrals,
        state.densityMatrix,
        dimension);
    const auto transformedFock = multiply(
        multiply(state.orthogonalizerMatrix, fock, dimension),
        state.orthogonalizerMatrix,
        dimension);
    const auto decomposition = diagonalizeSymmetric(transformedFock, dimension);
    state.orbitalEnergies = decomposition.eigenvalues;
    state.coefficientMatrix = multiply(state.orthogonalizerMatrix, decomposition.eigenvectors, dimension);

    auto nextDensity = zeroMatrix(dimension);
    for (std::size_t mu = 0; mu < dimension; ++mu) {
        for (std::size_t nu = 0; nu < dimension; ++nu) {
            double value = 0.0;
            for (int orbital = 0; orbital < state.occupiedOrbitals; ++orbital) {
                value += 2.0 *
                         matrixEntry(state.coefficientMatrix, dimension, mu, static_cast<std::size_t>(orbital)) *
                         matrixEntry(state.coefficientMatrix, dimension, nu, static_cast<std::size_t>(orbital));
            }
            matrixEntry(nextDensity, dimension, mu, nu) = value;
        }
    }

    double electronicEnergy = 0.0;
    for (std::size_t mu = 0; mu < dimension; ++mu) {
        for (std::size_t nu = 0; nu < dimension; ++nu) {
            electronicEnergy += 0.5 * matrixEntry(nextDensity, dimension, mu, nu) *
                                (matrixEntry(state.coreHamiltonianMatrix, dimension, mu, nu) +
                                 matrixEntry(fock, dimension, mu, nu));
        }
    }

    state.energyDelta = std::abs(electronicEnergy - state.previousElectronicEnergy);
    state.densityDelta = maxAbsoluteDifference(nextDensity, state.densityMatrix);
    state.densityMatrix = std::move(nextDensity);
    state.previousElectronicEnergy = electronicEnergy;
    state.electronicEnergy = electronicEnergy;
    state.totalEnergy = electronicEnergy + state.nuclearRepulsionEnergy;
    state.iterations += 1;
    if (state.iterations > 1 &&
        state.energyDelta <= state.options.energyTolerance &&
        state.densityDelta <= state.options.densityTolerance) {
        state.converged = true;
    }

    return state.converged;
}

RestrictedHartreeFockResult solveRestrictedHartreeFock(
    const Molecule& molecule,
    const RestrictedHartreeFockOptions& options) {
    auto state = initializeRestrictedHartreeFockState(molecule, options);
    while (!state.converged && state.iterations < state.options.maxIterations) {
        stepRestrictedHartreeFock(state);
    }

    return makeRestrictedHartreeFockResult(state);
}

UnrestrictedHartreeFockResult solveUnrestrictedHartreeFock(
    const Molecule& molecule,
    const UnrestrictedHartreeFockOptions& options) {
    if (molecule.electronCount < 0) {
        throw std::runtime_error("unrestricted Hartree-Fock requires a nonnegative electron count");
    }

    const auto integrals = buildIntegralSet(molecule);
    const std::size_t dimension = integrals.basis.size();
    const auto [alphaElectrons, betaElectrons] = determineSpinPopulation(
        molecule.electronCount,
        options.alphaElectronCount,
        options.betaElectronCount);
    if (alphaElectrons > static_cast<int>(dimension) || betaElectrons > static_cast<int>(dimension)) {
        throw std::runtime_error("spin populations exceed the supported minimal basis dimension");
    }

    const auto orthogonalizer = symmetricInverseSquareRoot(integrals.overlap, dimension);
    auto alphaDensity = zeroMatrix(dimension);
    auto betaDensity = zeroMatrix(dimension);
    auto alphaCoefficients = zeroMatrix(dimension);
    auto betaCoefficients = zeroMatrix(dimension);
    std::vector<double> alphaOrbitalEnergies(dimension, 0.0);
    std::vector<double> betaOrbitalEnergies(dimension, 0.0);
    double previousElectronicEnergy = 0.0;

    UnrestrictedHartreeFockResult result{};
    result.basisCount = dimension;
    result.alphaElectronCount = alphaElectrons;
    result.betaElectronCount = betaElectrons;
    result.nuclearRepulsionEnergy = integrals.nuclearRepulsion;

    for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
        auto alphaFock = integrals.coreHamiltonian;
        auto betaFock = integrals.coreHamiltonian;
        for (std::size_t mu = 0; mu < dimension; ++mu) {
            for (std::size_t nu = 0; nu < dimension; ++nu) {
                double alphaInteraction = 0.0;
                double betaInteraction = 0.0;
                for (std::size_t lambda = 0; lambda < dimension; ++lambda) {
                    for (std::size_t sigma = 0; sigma < dimension; ++sigma) {
                        const double totalDensity =
                            matrixEntry(alphaDensity, dimension, lambda, sigma) +
                            matrixEntry(betaDensity, dimension, lambda, sigma);
                        alphaInteraction +=
                            totalDensity * integrals.electronRepulsion[eriIndex(dimension, mu, nu, lambda, sigma)] -
                            matrixEntry(alphaDensity, dimension, lambda, sigma) * integrals.electronRepulsion[eriIndex(dimension, mu, lambda, nu, sigma)];
                        betaInteraction +=
                            totalDensity * integrals.electronRepulsion[eriIndex(dimension, mu, nu, lambda, sigma)] -
                            matrixEntry(betaDensity, dimension, lambda, sigma) * integrals.electronRepulsion[eriIndex(dimension, mu, lambda, nu, sigma)];
                    }
                }
                matrixEntry(alphaFock, dimension, mu, nu) += alphaInteraction;
                matrixEntry(betaFock, dimension, mu, nu) += betaInteraction;
            }
        }

        const auto transformedAlphaFock = multiply(multiply(orthogonalizer, alphaFock, dimension), orthogonalizer, dimension);
        const auto transformedBetaFock = multiply(multiply(orthogonalizer, betaFock, dimension), orthogonalizer, dimension);
        const auto alphaDecomposition = diagonalizeSymmetric(transformedAlphaFock, dimension);
        const auto betaDecomposition = diagonalizeSymmetric(transformedBetaFock, dimension);
        alphaOrbitalEnergies = alphaDecomposition.eigenvalues;
        betaOrbitalEnergies = betaDecomposition.eigenvalues;
        alphaCoefficients = multiply(orthogonalizer, alphaDecomposition.eigenvectors, dimension);
        betaCoefficients = multiply(orthogonalizer, betaDecomposition.eigenvectors, dimension);

        auto nextAlphaDensity = zeroMatrix(dimension);
        auto nextBetaDensity = zeroMatrix(dimension);
        for (std::size_t mu = 0; mu < dimension; ++mu) {
            for (std::size_t nu = 0; nu < dimension; ++nu) {
                double alphaValue = 0.0;
                double betaValue = 0.0;
                for (int orbital = 0; orbital < alphaElectrons; ++orbital) {
                    alphaValue += matrixEntry(alphaCoefficients, dimension, mu, static_cast<std::size_t>(orbital)) *
                                  matrixEntry(alphaCoefficients, dimension, nu, static_cast<std::size_t>(orbital));
                }
                for (int orbital = 0; orbital < betaElectrons; ++orbital) {
                    betaValue += matrixEntry(betaCoefficients, dimension, mu, static_cast<std::size_t>(orbital)) *
                                 matrixEntry(betaCoefficients, dimension, nu, static_cast<std::size_t>(orbital));
                }
                matrixEntry(nextAlphaDensity, dimension, mu, nu) = alphaValue;
                matrixEntry(nextBetaDensity, dimension, mu, nu) = betaValue;
            }
        }

        double electronicEnergy = 0.0;
        for (std::size_t mu = 0; mu < dimension; ++mu) {
            for (std::size_t nu = 0; nu < dimension; ++nu) {
                electronicEnergy += 0.5 * (
                    matrixEntry(nextAlphaDensity, dimension, mu, nu) *
                        (matrixEntry(integrals.coreHamiltonian, dimension, mu, nu) + matrixEntry(alphaFock, dimension, mu, nu)) +
                    matrixEntry(nextBetaDensity, dimension, mu, nu) *
                        (matrixEntry(integrals.coreHamiltonian, dimension, mu, nu) + matrixEntry(betaFock, dimension, mu, nu)));
            }
        }

        const double energyDelta = std::abs(electronicEnergy - previousElectronicEnergy);
        const double densityDelta = std::max(
            maxAbsoluteDifference(nextAlphaDensity, alphaDensity),
            maxAbsoluteDifference(nextBetaDensity, betaDensity));
        alphaDensity = std::move(nextAlphaDensity);
        betaDensity = std::move(nextBetaDensity);
        previousElectronicEnergy = electronicEnergy;

        result.iterations = iteration + 1;
        result.electronicEnergy = electronicEnergy;
        result.totalEnergy = electronicEnergy + result.nuclearRepulsionEnergy;

        if ((iteration > 0) && energyDelta <= options.energyTolerance && densityDelta <= options.densityTolerance) {
            result.converged = true;
            break;
        }
    }

    result.alphaOrbitalEnergies = std::move(alphaOrbitalEnergies);
    result.betaOrbitalEnergies = std::move(betaOrbitalEnergies);
    result.alphaDensityMatrix = std::move(alphaDensity);
    result.betaDensityMatrix = std::move(betaDensity);
    result.alphaCoefficientMatrix = std::move(alphaCoefficients);
    result.betaCoefficientMatrix = std::move(betaCoefficients);
    return result;
}

RestrictedMollerPlesset2Result solveRestrictedMollerPlesset2(
    const Molecule& molecule,
    const RestrictedMollerPlesset2Options& options) {
    if (options.denominatorTolerance <= 0.0) {
        throw std::runtime_error("MP2 denominator tolerance must be positive");
    }

    const auto hartreeFock = solveRestrictedHartreeFock(molecule, options.hartreeFock);
    RestrictedMollerPlesset2Result result{};
    result.scfConverged = hartreeFock.converged;
    result.scfIterations = hartreeFock.iterations;
    result.basisCount = hartreeFock.basisCount;
    result.occupiedOrbitals = hartreeFock.occupiedOrbitals;
    result.hartreeFockElectronicEnergy = hartreeFock.electronicEnergy;
    result.nuclearRepulsionEnergy = hartreeFock.nuclearRepulsionEnergy;
    result.hartreeFockTotalEnergy = hartreeFock.totalEnergy;
    result.totalEnergy = hartreeFock.totalEnergy;
    result.orbitalEnergies = hartreeFock.orbitalEnergies;

    if (!hartreeFock.converged || hartreeFock.basisCount <= static_cast<std::size_t>(hartreeFock.occupiedOrbitals)) {
        return result;
    }

    const auto integrals = buildIntegralSet(molecule);
    const std::size_t dimension = hartreeFock.basisCount;
    const std::size_t occupiedCount = static_cast<std::size_t>(hartreeFock.occupiedOrbitals);
    double correlationEnergy = 0.0;

    for (std::size_t occupiedI = 0; occupiedI < occupiedCount; ++occupiedI) {
        for (std::size_t occupiedJ = 0; occupiedJ < occupiedCount; ++occupiedJ) {
            for (std::size_t virtualA = occupiedCount; virtualA < dimension; ++virtualA) {
                for (std::size_t virtualB = occupiedCount; virtualB < dimension; ++virtualB) {
                    const double denominator =
                        hartreeFock.orbitalEnergies[occupiedI] +
                        hartreeFock.orbitalEnergies[occupiedJ] -
                        hartreeFock.orbitalEnergies[virtualA] -
                        hartreeFock.orbitalEnergies[virtualB];
                    if (std::abs(denominator) <= options.denominatorTolerance) {
                        continue;
                    }

                    const double ijab = molecularOrbitalIntegral(
                        integrals,
                        hartreeFock.coefficientMatrix,
                        occupiedI,
                        occupiedJ,
                        virtualA,
                        virtualB);
                    const double ijba = molecularOrbitalIntegral(
                        integrals,
                        hartreeFock.coefficientMatrix,
                        occupiedI,
                        occupiedJ,
                        virtualB,
                        virtualA);
                    correlationEnergy += (ijab * ((2.0 * ijab) - ijba)) / denominator;
                }
            }
        }
    }

    result.correlationEnergy = correlationEnergy;
    result.totalEnergy = result.hartreeFockTotalEnergy + correlationEnergy;
    return result;
}

}  // namespace physicsmade::quantum_chemistry
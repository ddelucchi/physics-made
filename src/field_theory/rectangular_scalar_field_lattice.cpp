#include "physicsmade/field_theory/rectangular_scalar_field_lattice.hpp"

#include <algorithm>
#include <cmath>

namespace {

std::size_t wrapIndex(std::size_t index, std::size_t extent, int delta) noexcept {
    const auto signedExtent = static_cast<long long>(extent);
    const auto signedIndex = static_cast<long long>(index);
    return static_cast<std::size_t>((signedIndex + delta + signedExtent) % signedExtent);
}

}  // namespace

namespace physicsmade::field_theory {

RectangularScalarFieldLattice::RectangularScalarFieldLattice(RectangularLatticeSpec spec)
    : spec_(spec), phi_(spec.sizeX * spec.sizeY * spec.sizeZ, 0.0), pi_(phi_.size(), 0.0) {}

void RectangularScalarFieldLattice::seedGaussianPacket(
    double amplitude,
    const math::Vector3& center,
    double width,
    const math::Vector3& waveVector) {
    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const math::Vector3 position{
                    spec_.spacing * static_cast<double>(x),
                    spec_.spacing * static_cast<double>(y),
                    spec_.spacing * static_cast<double>(z),
                };
                const auto delta = position - center;
                const double radiusSquared = delta.normSquared();
                const double envelope = amplitude * std::exp(-(radiusSquared / (2.0 * width * width)));
                const double phase = waveVector.dot(position);
                const auto index = flatIndex(x, y, z);
                phi_[index] = envelope * std::cos(phase);
                pi_[index] = -envelope * waveVector.norm() * std::sin(phase);
            }
        }
    }
}

void RectangularScalarFieldLattice::step(double dtSeconds, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<double> nextPhi = phi_;
        std::vector<double> nextPi = pi_;

        for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
            for (std::size_t y = 0; y < spec_.sizeY; ++y) {
                for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                    const auto index = flatIndex(x, y, z);
                    nextPhi[index] = phi_[index] + (dtSeconds * pi_[index]);

                    const double nonlinearTerm = spec_.selfCoupling * phi_[index] * phi_[index] * phi_[index];
                    nextPi[index] = pi_[index] + (dtSeconds * (laplacianAt(x, y, z) - (spec_.mass * spec_.mass * phi_[index]) - nonlinearTerm));
                }
            }
        }

        phi_ = std::move(nextPhi);
        pi_ = std::move(nextPi);
    }
}

RectangularFieldObservables RectangularScalarFieldLattice::observables() const {
    RectangularFieldObservables observables{};
    const double cellVolume = spec_.spacing * spec_.spacing * spec_.spacing;

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const auto rightX = flatIndex(wrapIndex(x, spec_.sizeX, 1), y, z);
                const auto rightY = flatIndex(x, wrapIndex(y, spec_.sizeY, 1), z);
                const auto rightZ = flatIndex(x, y, wrapIndex(z, spec_.sizeZ, 1));

                const double gradientX = (phi_[rightX] - phi_[index]) / spec_.spacing;
                const double gradientY = (phi_[rightY] - phi_[index]) / spec_.spacing;
                const double gradientZ = (phi_[rightZ] - phi_[index]) / spec_.spacing;
                const double gradientDensity = 0.5 * ((gradientX * gradientX) + (gradientY * gradientY) + (gradientZ * gradientZ));
                const double potentialDensityValue = potentialDensity(phi_[index]);
                const double density = 0.5 * pi_[index] * pi_[index] + gradientDensity + potentialDensityValue;

                observables.totalEnergy += density * cellVolume;
                observables.gradientEnergy += gradientDensity * cellVolume;
                observables.potentialEnergy += potentialDensityValue * cellVolume;
                observables.meanField += phi_[index];
                observables.maxAmplitude = std::max(observables.maxAmplitude, std::abs(phi_[index]));
            }
        }
    }

    if (!phi_.empty()) {
        observables.meanField /= static_cast<double>(phi_.size());
    }

    return observables;
}

std::size_t RectangularScalarFieldLattice::flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return x + (spec_.sizeX * (y + (spec_.sizeY * z)));
}

double RectangularScalarFieldLattice::laplacianAt(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto center = phi_[flatIndex(x, y, z)];
    const auto xMinus = phi_[flatIndex(wrapIndex(x, spec_.sizeX, -1), y, z)];
    const auto xPlus = phi_[flatIndex(wrapIndex(x, spec_.sizeX, 1), y, z)];
    const auto yMinus = phi_[flatIndex(x, wrapIndex(y, spec_.sizeY, -1), z)];
    const auto yPlus = phi_[flatIndex(x, wrapIndex(y, spec_.sizeY, 1), z)];
    const auto zMinus = phi_[flatIndex(x, y, wrapIndex(z, spec_.sizeZ, -1))];
    const auto zPlus = phi_[flatIndex(x, y, wrapIndex(z, spec_.sizeZ, 1))];

    return (xMinus + xPlus + yMinus + yPlus + zMinus + zPlus - (6.0 * center)) / (spec_.spacing * spec_.spacing);
}

double RectangularScalarFieldLattice::potentialDensity(double value) const noexcept {
    const double massTerm = 0.5 * spec_.mass * spec_.mass * value * value;
    const double selfInteraction = 0.25 * spec_.selfCoupling * value * value * value * value;
    return massTerm + selfInteraction;
}

}  // namespace physicsmade::field_theory
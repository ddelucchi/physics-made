#include "physicsmade/field_theory/scalar_field_lattice.hpp"

#include <algorithm>
#include <cmath>

namespace physicsmade::field_theory {

ScalarFieldLattice::ScalarFieldLattice(LatticeSpec spec)
    : spec_(spec), phi_(spec.siteCount, 0.0), pi_(spec.siteCount, 0.0) {}

void ScalarFieldLattice::seedGaussianPacket(double amplitude, double center, double width, double waveNumber) {
    for (std::size_t index = 0; index < spec_.siteCount; ++index) {
        const double x = spec_.spacing * static_cast<double>(index);
        const double envelope = amplitude * std::exp(-((x - center) * (x - center)) / (2.0 * width * width));
        phi_[index] = envelope * std::cos(waveNumber * x);
        pi_[index] = -waveNumber * envelope * std::sin(waveNumber * x);
    }
}

void ScalarFieldLattice::step(double dtSeconds, int iterations) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<double> nextPhi = phi_;
        std::vector<double> nextPi = pi_;

        for (std::size_t index = 0; index < spec_.siteCount; ++index) {
            nextPhi[index] = phi_[index] + (dtSeconds * pi_[index]);

            const double nonlinearTerm = spec_.selfCoupling * phi_[index] * phi_[index] * phi_[index];
            nextPi[index] = pi_[index] + (dtSeconds * (laplacianAt(index) - (spec_.mass * spec_.mass * phi_[index]) - nonlinearTerm));
        }

        phi_ = std::move(nextPhi);
        pi_ = std::move(nextPi);
    }
}

FieldObservables ScalarFieldLattice::observables() const {
    FieldObservables observables{};
    for (std::size_t index = 0; index < spec_.siteCount; ++index) {
        const std::size_t nextIndex = (index + 1) % spec_.siteCount;
        const double gradient = (phi_[nextIndex] - phi_[index]) / spec_.spacing;
        const double density = 0.5 * pi_[index] * pi_[index] + 0.5 * gradient * gradient + potentialDensity(phi_[index]);
        observables.totalEnergy += density * spec_.spacing;
        observables.meanField += phi_[index];
        observables.maxAmplitude = std::max(observables.maxAmplitude, std::abs(phi_[index]));
        observables.nearestNeighborCorrelation += phi_[index] * phi_[nextIndex];
    }

    if (spec_.siteCount > 0) {
        observables.meanField /= static_cast<double>(spec_.siteCount);
        observables.nearestNeighborCorrelation /= static_cast<double>(spec_.siteCount);
    }

    return observables;
}

double ScalarFieldLattice::laplacianAt(std::size_t index) const noexcept {
    const std::size_t leftIndex = (index + spec_.siteCount - 1) % spec_.siteCount;
    const std::size_t rightIndex = (index + 1) % spec_.siteCount;
    return (phi_[leftIndex] - (2.0 * phi_[index]) + phi_[rightIndex]) / (spec_.spacing * spec_.spacing);
}

double ScalarFieldLattice::potentialDensity(double value) const noexcept {
    const double massTerm = 0.5 * spec_.mass * spec_.mass * value * value;
    const double selfInteraction = 0.25 * spec_.selfCoupling * value * value * value * value;
    return massTerm + selfInteraction;
}

}  // namespace physicsmade::field_theory

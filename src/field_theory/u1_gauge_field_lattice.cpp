#include "physicsmade/field_theory/u1_gauge_field_lattice.hpp"

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

U1GaugeFieldLattice::U1GaugeFieldLattice(U1GaugeLatticeSpec spec)
    : spec_(spec),
      linkX_(spec.sizeX * spec.sizeY * spec.sizeZ, 0.0),
      linkY_(linkX_.size(), 0.0),
      linkZ_(linkX_.size(), 0.0),
      electricX_(linkX_.size(), 0.0),
      electricY_(linkX_.size(), 0.0),
      electricZ_(linkX_.size(), 0.0) {}

void U1GaugeFieldLattice::seedStandingWave(double amplitude, const math::Vector3& waveVector) {
    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const math::Vector3 position{
                    spec_.spacing * static_cast<double>(x),
                    spec_.spacing * static_cast<double>(y),
                    spec_.spacing * static_cast<double>(z),
                };
                const double phase = waveVector.dot(position);
                linkX_[index] = amplitude * std::cos(phase);
                linkY_[index] = amplitude * std::sin(phase);
                linkZ_[index] = 0.5 * amplitude * std::cos(0.5 * phase);
                electricX_[index] = 0.5 * amplitude * std::sin(phase);
                electricY_[index] = -0.5 * amplitude * std::cos(phase);
                electricZ_[index] = 0.25 * amplitude * std::sin(0.5 * phase);
            }
        }
    }
}

void U1GaugeFieldLattice::seedMagneticFluxSheet(double amplitude) {
    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                linkY_[index] += amplitude * static_cast<double>(x) / static_cast<double>(std::max<std::size_t>(1, spec_.sizeX));
            }
        }
    }
}

void U1GaugeFieldLattice::step(double dtSeconds, int iterations) {
    const double couplingScale = 1.0 / std::max(spec_.coupling * spec_.coupling, 1.0e-12);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto nextLinkX = linkX_;
        auto nextLinkY = linkY_;
        auto nextLinkZ = linkZ_;

        for (std::size_t index = 0; index < linkX_.size(); ++index) {
            nextLinkX[index] += dtSeconds * electricX_[index];
            nextLinkY[index] += dtSeconds * electricY_[index];
            nextLinkZ[index] += dtSeconds * electricZ_[index];
        }

        auto nextElectricX = electricX_;
        auto nextElectricY = electricY_;
        auto nextElectricZ = electricZ_;

        for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
            for (std::size_t y = 0; y < spec_.sizeY; ++y) {
                for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                    const auto index = flatIndex(x, y, z);
                    const auto prevY = wrapIndex(y, spec_.sizeY, -1);
                    const auto prevZ = wrapIndex(z, spec_.sizeZ, -1);
                    const auto prevX = wrapIndex(x, spec_.sizeX, -1);

                    const double residualX =
                        std::sin(plaquettePhaseXY(x, y, z)) - std::sin(plaquettePhaseXY(x, prevY, z)) +
                        std::sin(plaquettePhaseXZ(x, y, z)) - std::sin(plaquettePhaseXZ(x, y, prevZ));
                    const double residualY =
                        -std::sin(plaquettePhaseXY(x, y, z)) + std::sin(plaquettePhaseXY(prevX, y, z)) +
                        std::sin(plaquettePhaseYZ(x, y, z)) - std::sin(plaquettePhaseYZ(x, y, prevZ));
                    const double residualZ =
                        -std::sin(plaquettePhaseXZ(x, y, z)) + std::sin(plaquettePhaseXZ(prevX, y, z)) -
                        std::sin(plaquettePhaseYZ(x, y, z)) + std::sin(plaquettePhaseYZ(x, prevY, z));

                    nextElectricX[index] -= dtSeconds * couplingScale * residualX;
                    nextElectricY[index] -= dtSeconds * couplingScale * residualY;
                    nextElectricZ[index] -= dtSeconds * couplingScale * residualZ;
                }
            }
        }

        linkX_ = std::move(nextLinkX);
        linkY_ = std::move(nextLinkY);
        linkZ_ = std::move(nextLinkZ);
        electricX_ = std::move(nextElectricX);
        electricY_ = std::move(nextElectricY);
        electricZ_ = std::move(nextElectricZ);
    }
}

U1GaugeObservables U1GaugeFieldLattice::observables() const {
    U1GaugeObservables observables{};
    const double cellVolume = spec_.spacing * spec_.spacing * spec_.spacing;
    const double couplingScale = 1.0 / std::max(spec_.coupling * spec_.coupling, 1.0e-12);

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const double electricMagnitudeSquared =
                    (electricX_[index] * electricX_[index]) +
                    (electricY_[index] * electricY_[index]) +
                    (electricZ_[index] * electricZ_[index]);
                observables.electricEnergy += 0.5 * electricMagnitudeSquared * cellVolume;
                observables.maxElectricField = std::max(
                    observables.maxElectricField,
                    std::sqrt(electricMagnitudeSquared));

                const double plaquetteXY = plaquettePhaseXY(x, y, z);
                const double plaquetteXZ = plaquettePhaseXZ(x, y, z);
                const double plaquetteYZ = plaquettePhaseYZ(x, y, z);
                observables.magneticEnergy +=
                    couplingScale * ((1.0 - std::cos(plaquetteXY)) + (1.0 - std::cos(plaquetteXZ)) + (1.0 - std::cos(plaquetteYZ))) * cellVolume;
                observables.meanPlaquette +=
                    (std::cos(plaquetteXY) + std::cos(plaquetteXZ) + std::cos(plaquetteYZ)) / 3.0;
            }
        }
    }

    if (!linkX_.empty()) {
        observables.meanPlaquette /= static_cast<double>(linkX_.size());
    }

    observables.totalEnergy = observables.electricEnergy + observables.magneticEnergy;
    return observables;
}

std::size_t U1GaugeFieldLattice::flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return x + (spec_.sizeX * (y + (spec_.sizeY * z)));
}

double U1GaugeFieldLattice::plaquettePhaseXY(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    return linkX_[flatIndex(x, y, z)] + linkY_[flatIndex(xPlus, y, z)] - linkX_[flatIndex(x, yPlus, z)] - linkY_[flatIndex(x, y, z)];
}

double U1GaugeFieldLattice::plaquettePhaseXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return linkX_[flatIndex(x, y, z)] + linkZ_[flatIndex(xPlus, y, z)] - linkX_[flatIndex(x, y, zPlus)] - linkZ_[flatIndex(x, y, z)];
}

double U1GaugeFieldLattice::plaquettePhaseYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return linkY_[flatIndex(x, y, z)] + linkZ_[flatIndex(x, yPlus, z)] - linkY_[flatIndex(x, y, zPlus)] - linkZ_[flatIndex(x, y, z)];
}

}  // namespace physicsmade::field_theory
#include "physicsmade/field_theory/su2_gauge_field_lattice.hpp"

#include <algorithm>
#include <cmath>

namespace {

std::size_t wrapIndex(std::size_t index, std::size_t extent, int delta) noexcept {
    const auto signedExtent = static_cast<long long>(extent);
    const auto signedIndex = static_cast<long long>(index);
    return static_cast<std::size_t>((signedIndex + delta + signedExtent) % signedExtent);
}

physicsmade::math::Vector3 normalizedOrFallback(const physicsmade::math::Vector3& vector) noexcept {
    if (vector.normSquared() <= physicsmade::common::kEpsilon) {
        return {0.0, 0.0, 1.0};
    }

    return vector.normalized();
}

physicsmade::math::Quaternion linkFromAlgebra(const physicsmade::math::Vector3& algebra) noexcept {
    const double magnitude = algebra.norm();
    if (magnitude <= physicsmade::common::kEpsilon) {
        return physicsmade::math::Quaternion::identity();
    }

    return physicsmade::math::Quaternion::fromAxisAngle(algebra / magnitude, magnitude).normalized();
}

physicsmade::math::Vector3 quaternionVectorPart(const physicsmade::math::Quaternion& quaternion) noexcept {
    return {quaternion.x, quaternion.y, quaternion.z};
}

double normalizedTraceHalf(const physicsmade::math::Quaternion& quaternion) noexcept {
    return std::clamp(quaternion.normalized().w, -1.0, 1.0);
}

}  // namespace

namespace physicsmade::field_theory {

SU2GaugeFieldLattice::SU2GaugeFieldLattice(SU2GaugeLatticeSpec spec)
    : spec_(spec),
      linkX_(spec.sizeX * spec.sizeY * spec.sizeZ, math::Quaternion::identity()),
      linkY_(linkX_.size(), math::Quaternion::identity()),
      linkZ_(linkX_.size(), math::Quaternion::identity()),
      electricX_(linkX_.size(), {}),
      electricY_(linkX_.size(), {}),
      electricZ_(linkX_.size(), {}) {}

void SU2GaugeFieldLattice::seedStandingWave(
    double amplitude,
    const math::Vector3& waveVector,
    const math::Vector3& colorDirection) {
    const auto color = normalizedOrFallback(colorDirection);

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
                const auto algebraX = color * (amplitude * std::cos(phase));
                const auto algebraY = color * (amplitude * std::sin(phase));
                const auto algebraZ = color * (0.5 * amplitude * std::cos(0.5 * phase));

                linkX_[index] = linkFromAlgebra(algebraX);
                linkY_[index] = linkFromAlgebra(algebraY);
                linkZ_[index] = linkFromAlgebra(algebraZ);
                electricX_[index] = color * (0.5 * amplitude * std::sin(phase));
                electricY_[index] = color * (-0.5 * amplitude * std::cos(phase));
                electricZ_[index] = color * (0.25 * amplitude * std::sin(0.5 * phase));
            }
        }
    }
}

void SU2GaugeFieldLattice::seedColorMagneticFlux(
    double amplitude,
    const math::Vector3& colorDirection) {
    const auto color = normalizedOrFallback(colorDirection);
    const double extent = static_cast<double>(std::max<std::size_t>(1, spec_.sizeX));

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                linkY_[index] = (linkFromAlgebra(color * (amplitude * static_cast<double>(x) / extent)) * linkY_[index]).normalized();
            }
        }
    }
}

void SU2GaugeFieldLattice::step(double dtSeconds, int iterations) {
    const double couplingScale = 1.0 / std::max(spec_.coupling * spec_.coupling, 1.0e-12);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        auto nextLinkX = linkX_;
        auto nextLinkY = linkY_;
        auto nextLinkZ = linkZ_;

        for (std::size_t index = 0; index < linkX_.size(); ++index) {
            nextLinkX[index] = (linkFromAlgebra(dtSeconds * electricX_[index]) * linkX_[index]).normalized();
            nextLinkY[index] = (linkFromAlgebra(dtSeconds * electricY_[index]) * linkY_[index]).normalized();
            nextLinkZ[index] = (linkFromAlgebra(dtSeconds * electricZ_[index]) * linkZ_[index]).normalized();
        }

        auto nextElectricX = electricX_;
        auto nextElectricY = electricY_;
        auto nextElectricZ = electricZ_;

        for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
            for (std::size_t y = 0; y < spec_.sizeY; ++y) {
                for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                    const auto index = flatIndex(x, y, z);
                    const auto prevX = wrapIndex(x, spec_.sizeX, -1);
                    const auto prevY = wrapIndex(y, spec_.sizeY, -1);
                    const auto prevZ = wrapIndex(z, spec_.sizeZ, -1);

                    const auto residualX =
                        quaternionVectorPart(plaquetteXY(x, y, z)) - quaternionVectorPart(plaquetteXY(x, prevY, z)) +
                        quaternionVectorPart(plaquetteXZ(x, y, z)) - quaternionVectorPart(plaquetteXZ(x, y, prevZ));
                    const auto residualY =
                        -quaternionVectorPart(plaquetteXY(x, y, z)) + quaternionVectorPart(plaquetteXY(prevX, y, z)) +
                        quaternionVectorPart(plaquetteYZ(x, y, z)) - quaternionVectorPart(plaquetteYZ(x, y, prevZ));
                    const auto residualZ =
                        -quaternionVectorPart(plaquetteXZ(x, y, z)) + quaternionVectorPart(plaquetteXZ(prevX, y, z)) -
                        quaternionVectorPart(plaquetteYZ(x, y, z)) + quaternionVectorPart(plaquetteYZ(x, prevY, z));

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

SU2GaugeObservables SU2GaugeFieldLattice::observables() const {
    SU2GaugeObservables observables{};
    const double cellVolume = spec_.spacing * spec_.spacing * spec_.spacing;
    const double couplingScale = 1.0 / std::max(spec_.coupling * spec_.coupling, 1.0e-12);

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const double electricMagnitudeSquared =
                    electricX_[index].normSquared() +
                    electricY_[index].normSquared() +
                    electricZ_[index].normSquared();
                observables.electricEnergy += 0.5 * electricMagnitudeSquared * cellVolume;
                observables.maxElectricField = std::max(observables.maxElectricField, std::sqrt(electricMagnitudeSquared));
                observables.meanColorField += electricX_[index] + electricY_[index] + electricZ_[index];

                const auto plaquetteXYValue = plaquetteXY(x, y, z);
                const auto plaquetteXZValue = plaquetteXZ(x, y, z);
                const auto plaquetteYZValue = plaquetteYZ(x, y, z);
                const double traceXY = normalizedTraceHalf(plaquetteXYValue);
                const double traceXZ = normalizedTraceHalf(plaquetteXZValue);
                const double traceYZ = normalizedTraceHalf(plaquetteYZValue);
                observables.magneticEnergy += couplingScale * ((1.0 - traceXY) + (1.0 - traceXZ) + (1.0 - traceYZ)) * cellVolume;
                observables.meanPlaquetteTrace += (traceXY + traceXZ + traceYZ) / 3.0;
            }
        }
    }

    if (!linkX_.empty()) {
        observables.meanPlaquetteTrace /= static_cast<double>(linkX_.size());
        observables.meanColorField *= 1.0 / (3.0 * static_cast<double>(linkX_.size()));
    }

    observables.totalEnergy = observables.electricEnergy + observables.magneticEnergy;
    return observables;
}

std::size_t SU2GaugeFieldLattice::flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return x + (spec_.sizeX * (y + (spec_.sizeY * z)));
}

math::Quaternion SU2GaugeFieldLattice::plaquetteXY(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    return (
        linkX_[flatIndex(x, y, z)] *
        linkY_[flatIndex(xPlus, y, z)] *
        linkX_[flatIndex(x, yPlus, z)].conjugate() *
        linkY_[flatIndex(x, y, z)].conjugate()).normalized();
}

math::Quaternion SU2GaugeFieldLattice::plaquetteXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return (
        linkX_[flatIndex(x, y, z)] *
        linkZ_[flatIndex(xPlus, y, z)] *
        linkX_[flatIndex(x, y, zPlus)].conjugate() *
        linkZ_[flatIndex(x, y, z)].conjugate()).normalized();
}

math::Quaternion SU2GaugeFieldLattice::plaquetteYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return (
        linkY_[flatIndex(x, y, z)] *
        linkZ_[flatIndex(x, yPlus, z)] *
        linkY_[flatIndex(x, y, zPlus)].conjugate() *
        linkZ_[flatIndex(x, y, z)].conjugate()).normalized();
}

}  // namespace physicsmade::field_theory
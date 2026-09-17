#include "physicsmade/field_theory/coupled_su2_fermion_lattice.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

#include "physicsmade/common/config.hpp"

namespace {

using Complex = std::complex<double>;
using Spinor = physicsmade::field_theory::FermionColorSpinor;

struct FermionCurrentSet {
    std::vector<physicsmade::math::Vector3> x{};
    std::vector<physicsmade::math::Vector3> y{};
    std::vector<physicsmade::math::Vector3> z{};
    physicsmade::math::Vector3 meanCurrent{};
    double interactionActivity{0.0};
};

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

Spinor zeroSpinor() noexcept {
    return {Complex{0.0, 0.0}, Complex{0.0, 0.0}};
}

Spinor operator+(const Spinor& left, const Spinor& right) noexcept {
    return {left[0] + right[0], left[1] + right[1]};
}

Spinor operator-(const Spinor& left, const Spinor& right) noexcept {
    return {left[0] - right[0], left[1] - right[1]};
}

Spinor operator*(const Spinor& spinor, const Complex& scalar) noexcept {
    return {spinor[0] * scalar, spinor[1] * scalar};
}

Spinor operator*(const Complex& scalar, const Spinor& spinor) noexcept {
    return spinor * scalar;
}

Spinor& operator+=(Spinor& left, const Spinor& right) noexcept {
    left[0] += right[0];
    left[1] += right[1];
    return left;
}

Spinor coherentColorSpinor(const physicsmade::math::Vector3& colorDirection) noexcept {
    const auto direction = normalizedOrFallback(colorDirection);
    const double denominator = std::sqrt(std::max(1.0 + direction.z, 1.0e-12));
    if ((1.0 + direction.z) <= 1.0e-12) {
        return {Complex{0.0, 0.0}, Complex{1.0, 0.0}};
    }

    return {
        Complex{std::sqrt(0.5 * (1.0 + direction.z)), 0.0},
        Complex{direction.x / (std::sqrt(2.0) * denominator), direction.y / (std::sqrt(2.0) * denominator)},
    };
}

Spinor applyLink(const physicsmade::math::Quaternion& link, const Spinor& spinor) noexcept {
    const auto normalizedLink = link.normalized();
    const Complex a{normalizedLink.w, normalizedLink.z};
    const Complex b{normalizedLink.y, normalizedLink.x};
    return {
        (a * spinor[0]) + (b * spinor[1]),
        (-std::conj(b) * spinor[0]) + (std::conj(a) * spinor[1]),
    };
}

double siteDensity(const Spinor& spinor) noexcept {
    return std::norm(spinor[0]) + std::norm(spinor[1]);
}

Complex overlap(const Spinor& left, const Spinor& right) noexcept {
    return (std::conj(left[0]) * right[0]) + (std::conj(left[1]) * right[1]);
}

physicsmade::math::Vector3 colorCurrentBilinear(const Spinor& left, const Spinor& right) noexcept {
    const Complex sigma1 = (std::conj(left[0]) * right[1]) + (std::conj(left[1]) * right[0]);
    const Complex sigma2 = (Complex{0.0, -1.0} * std::conj(left[0]) * right[1]) +
                           (Complex{0.0, 1.0} * std::conj(left[1]) * right[0]);
    const Complex sigma3 = (std::conj(left[0]) * right[0]) - (std::conj(left[1]) * right[1]);
    return {std::imag(sigma1), std::imag(sigma2), std::imag(sigma3)};
}

double staggeredEta(int direction, std::size_t x, std::size_t y, std::size_t z) noexcept {
    (void)z;
    switch (direction) {
        case 0:
            return 1.0;
        case 1:
            return ((x % 2U) == 0U) ? 1.0 : -1.0;
        case 2:
            return (((x + y) % 2U) == 0U) ? 1.0 : -1.0;
        default:
            return 1.0;
    }
}

double staggeredParity(std::size_t x, std::size_t y, std::size_t z) noexcept {
    return (((x + y + z) % 2U) == 0U) ? 1.0 : -1.0;
}

physicsmade::math::Vector3 quaternionVectorPart(const physicsmade::math::Quaternion& quaternion) noexcept {
    return {quaternion.x, quaternion.y, quaternion.z};
}

double normalizedTraceHalf(const physicsmade::math::Quaternion& quaternion) noexcept {
    return std::clamp(quaternion.normalized().w, -1.0, 1.0);
}

FermionCurrentSet computeFermionCurrents(
    const physicsmade::field_theory::CoupledSu2FermionLatticeSpec& spec,
    const std::vector<Spinor>& psi,
    const std::vector<physicsmade::math::Quaternion>& linkX,
    const std::vector<physicsmade::math::Quaternion>& linkY,
    const std::vector<physicsmade::math::Quaternion>& linkZ) {
    FermionCurrentSet currents{};
    currents.x.resize(psi.size(), {});
    currents.y.resize(psi.size(), {});
    currents.z.resize(psi.size(), {});

    const double inverseSpacing = 1.0 / std::max(spec.spacing, 1.0e-12);
    const double cellVolume = spec.spacing * spec.spacing * spec.spacing;
    const auto flatIndex = [&spec](std::size_t x, std::size_t y, std::size_t z) {
        return x + (spec.sizeX * (y + (spec.sizeY * z)));
    };

    for (std::size_t z = 0; z < spec.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec.sizeY; ++y) {
            for (std::size_t x = 0; x < spec.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const auto xPlus = wrapIndex(x, spec.sizeX, 1);
                const auto yPlus = wrapIndex(y, spec.sizeY, 1);
                const auto zPlus = wrapIndex(z, spec.sizeZ, 1);

                const auto forwardX = applyLink(linkX[index], psi[flatIndex(xPlus, y, z)]);
                const auto forwardY = applyLink(linkY[index], psi[flatIndex(x, yPlus, z)]);
                const auto forwardZ = applyLink(linkZ[index], psi[flatIndex(x, y, zPlus)]);

                const double etaX = staggeredEta(0, x, y, z) * inverseSpacing;
                const double etaY = staggeredEta(1, x, y, z) * inverseSpacing;
                const double etaZ = staggeredEta(2, x, y, z) * inverseSpacing;

                const Complex overlapX = overlap(psi[index], forwardX);
                const Complex overlapY = overlap(psi[index], forwardY);
                const Complex overlapZ = overlap(psi[index], forwardZ);

                currents.x[index] = etaX * colorCurrentBilinear(psi[index], forwardX);
                currents.y[index] = etaY * colorCurrentBilinear(psi[index], forwardY);
                currents.z[index] = etaZ * colorCurrentBilinear(psi[index], forwardZ);
                currents.meanCurrent.x += etaX * std::imag(overlapX);
                currents.meanCurrent.y += etaY * std::imag(overlapY);
                currents.meanCurrent.z += etaZ * std::imag(overlapZ);
                currents.interactionActivity +=
                    (currents.x[index].normSquared() + currents.y[index].normSquared() + currents.z[index].normSquared()) * cellVolume;
            }
        }
    }

    if (!psi.empty()) {
        currents.meanCurrent *= 1.0 / static_cast<double>(psi.size());
    }

    return currents;
}

}  // namespace

namespace physicsmade::field_theory {

CoupledSu2FermionLattice::CoupledSu2FermionLattice(CoupledSu2FermionLatticeSpec spec)
    : spec_(spec),
      psi_(spec.sizeX * spec.sizeY * spec.sizeZ, zeroSpinor()),
      linkX_(psi_.size(), math::Quaternion::identity()),
      linkY_(psi_.size(), math::Quaternion::identity()),
      linkZ_(psi_.size(), math::Quaternion::identity()),
      electricX_(psi_.size(), {}),
      electricY_(psi_.size(), {}),
      electricZ_(psi_.size(), {}) {}

void CoupledSu2FermionLattice::seedFermionGaussianPacket(
    double amplitude,
    const math::Vector3& center,
    double width,
    const math::Vector3& waveVector,
    const math::Vector3& colorDirection) {
    const auto colorSpinor = coherentColorSpinor(colorDirection);
    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const math::Vector3 position{
                    spec_.spacing * static_cast<double>(x),
                    spec_.spacing * static_cast<double>(y),
                    spec_.spacing * static_cast<double>(z),
                };
                const auto delta = position - center;
                const double radiusSquared = delta.normSquared();
                const double envelope = amplitude * std::exp(-(radiusSquared / (2.0 * width * width)));
                const Complex phase = std::exp(Complex{0.0, waveVector.dot(position)});
                psi_[index][0] = envelope * phase * colorSpinor[0];
                psi_[index][1] = envelope * phase * colorSpinor[1];
            }
        }
    }
}

void CoupledSu2FermionLattice::seedGaugeStandingWave(
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
                linkX_[index] = linkFromAlgebra(color * (amplitude * std::cos(phase)));
                linkY_[index] = linkFromAlgebra(color * (amplitude * std::sin(phase)));
                linkZ_[index] = linkFromAlgebra(color * (0.5 * amplitude * std::cos(0.5 * phase)));
                electricX_[index] = color * (0.4 * amplitude * std::sin(phase));
                electricY_[index] = color * (-0.4 * amplitude * std::cos(phase));
                electricZ_[index] = color * (0.2 * amplitude * std::sin(0.5 * phase));
            }
        }
    }
}

void CoupledSu2FermionLattice::seedColorFlux(
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

void CoupledSu2FermionLattice::step(double dtSeconds, int iterations) {
    const double couplingScale = 1.0 / std::max(spec_.gaugeCoupling * spec_.gaugeCoupling, 1.0e-12);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto k1 = derivative(psi_);
        auto state2 = psi_;
        for (std::size_t index = 0; index < state2.size(); ++index) {
            state2[index] += k1[index] * Complex{0.5 * dtSeconds, 0.0};
        }
        const auto k2 = derivative(state2);
        auto state3 = psi_;
        for (std::size_t index = 0; index < state3.size(); ++index) {
            state3[index] += k2[index] * Complex{0.5 * dtSeconds, 0.0};
        }
        const auto k3 = derivative(state3);
        auto state4 = psi_;
        for (std::size_t index = 0; index < state4.size(); ++index) {
            state4[index] += k3[index] * Complex{dtSeconds, 0.0};
        }
        const auto k4 = derivative(state4);
        for (std::size_t index = 0; index < psi_.size(); ++index) {
            psi_[index] += (k1[index] + (Complex{2.0, 0.0} * k2[index]) + (Complex{2.0, 0.0} * k3[index]) + k4[index]) * Complex{dtSeconds / 6.0, 0.0};
        }

        const auto currents = computeFermionCurrents(spec_, psi_, linkX_, linkY_, linkZ_);

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

                    nextElectricX[index] -= dtSeconds * (couplingScale * residualX + (spec_.fermionBackreaction * currents.x[index]));
                    nextElectricY[index] -= dtSeconds * (couplingScale * residualY + (spec_.fermionBackreaction * currents.y[index]));
                    nextElectricZ[index] -= dtSeconds * (couplingScale * residualZ + (spec_.fermionBackreaction * currents.z[index]));
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

CoupledSu2FermionObservables CoupledSu2FermionLattice::observables() const {
    CoupledSu2FermionObservables observables{};
    const double cellVolume = spec_.spacing * spec_.spacing * spec_.spacing;
    const double inverseSpacing = 1.0 / std::max(spec_.spacing, 1.0e-12);
    const double couplingScale = 1.0 / std::max(spec_.gaugeCoupling * spec_.gaugeCoupling, 1.0e-12);
    const auto currents = computeFermionCurrents(spec_, psi_, linkX_, linkY_, linkZ_);

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const double density = siteDensity(psi_[index]);
                observables.totalProbability += density * cellVolume;
                observables.fermionMassActivity += std::abs(spec_.mass) * density * cellVolume;
                observables.maxSiteDensity = std::max(observables.maxSiteDensity, density);

                const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
                const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
                const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
                const auto xMinus = wrapIndex(x, spec_.sizeX, -1);
                const auto yMinus = wrapIndex(y, spec_.sizeY, -1);
                const auto zMinus = wrapIndex(z, spec_.sizeZ, -1);

                const auto forwardX = applyLink(linkX_[index], psi_[flatIndex(xPlus, y, z)]);
                const auto forwardY = applyLink(linkY_[index], psi_[flatIndex(x, yPlus, z)]);
                const auto forwardZ = applyLink(linkZ_[index], psi_[flatIndex(x, y, zPlus)]);
                const auto backwardX = applyLink(linkX_[flatIndex(xMinus, y, z)].conjugate(), psi_[flatIndex(xMinus, y, z)]);
                const auto backwardY = applyLink(linkY_[flatIndex(x, yMinus, z)].conjugate(), psi_[flatIndex(x, yMinus, z)]);
                const auto backwardZ = applyLink(linkZ_[flatIndex(x, y, zMinus)].conjugate(), psi_[flatIndex(x, y, zMinus)]);

                const auto hoppingX = (forwardX - backwardX) * Complex{0.5 * staggeredEta(0, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                const auto hoppingY = (forwardY - backwardY) * Complex{0.5 * staggeredEta(1, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                const auto hoppingZ = (forwardZ - backwardZ) * Complex{0.5 * staggeredEta(2, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                observables.fermionKineticActivity +=
                    (siteDensity(hoppingX) + siteDensity(hoppingY) + siteDensity(hoppingZ)) * cellVolume;

                const double electricMagnitudeSquared =
                    electricX_[index].normSquared() +
                    electricY_[index].normSquared() +
                    electricZ_[index].normSquared();
                observables.gaugeElectricEnergy += 0.5 * electricMagnitudeSquared * cellVolume;
                observables.meanColorField += electricX_[index] + electricY_[index] + electricZ_[index];

                const auto plaquetteXYValue = plaquetteXY(x, y, z);
                const auto plaquetteXZValue = plaquetteXZ(x, y, z);
                const auto plaquetteYZValue = plaquetteYZ(x, y, z);
                const double traceXY = normalizedTraceHalf(plaquetteXYValue);
                const double traceXZ = normalizedTraceHalf(plaquetteXZValue);
                const double traceYZ = normalizedTraceHalf(plaquetteYZValue);
                observables.gaugeMagneticEnergy += couplingScale * ((1.0 - traceXY) + (1.0 - traceXZ) + (1.0 - traceYZ)) * cellVolume;
                observables.meanPlaquetteTrace += (traceXY + traceXZ + traceYZ) / 3.0;
            }
        }
    }

    observables.meanCurrent = currents.meanCurrent;
    observables.interactionActivity = currents.interactionActivity;
    if (!psi_.empty()) {
        observables.meanPlaquetteTrace /= static_cast<double>(psi_.size());
        observables.meanColorField *= 1.0 / (3.0 * static_cast<double>(psi_.size()));
    }
    observables.totalEnergy =
        observables.fermionKineticActivity +
        observables.fermionMassActivity +
        observables.gaugeElectricEnergy +
        observables.gaugeMagneticEnergy +
        observables.interactionActivity;
    return observables;
}

std::vector<CoupledSu2FermionSiteSample> CoupledSu2FermionLattice::sampleSites() const {
    std::vector<CoupledSu2FermionSiteSample> samples;
    samples.reserve(psi_.size());
    const auto currents = computeFermionCurrents(spec_, psi_, linkX_, linkY_, linkZ_);
    const double inverseSpacing = 1.0 / std::max(spec_.spacing, 1.0e-12);

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
                const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
                const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);

                const auto forwardX = applyLink(linkX_[index], psi_[flatIndex(xPlus, y, z)]);
                const auto forwardY = applyLink(linkY_[index], psi_[flatIndex(x, yPlus, z)]);
                const auto forwardZ = applyLink(linkZ_[index], psi_[flatIndex(x, y, zPlus)]);
                const double currentX = staggeredEta(0, x, y, z) * inverseSpacing * std::imag(overlap(psi_[index], forwardX));
                const double currentY = staggeredEta(1, x, y, z) * inverseSpacing * std::imag(overlap(psi_[index], forwardY));
                const double currentZ = staggeredEta(2, x, y, z) * inverseSpacing * std::imag(overlap(psi_[index], forwardZ));

                const auto plaquetteXYValue = plaquetteXY(x, y, z);
                const auto plaquetteXZValue = plaquetteXZ(x, y, z);
                const auto plaquetteYZValue = plaquetteYZ(x, y, z);
                const double plaquetteTrace =
                    (normalizedTraceHalf(plaquetteXYValue) + normalizedTraceHalf(plaquetteXZValue) + normalizedTraceHalf(plaquetteYZValue)) / 3.0;

                samples.push_back({
                    {
                        spec_.spacing * static_cast<double>(x),
                        spec_.spacing * static_cast<double>(y),
                        spec_.spacing * static_cast<double>(z),
                    },
                    siteDensity(psi_[index]),
                    {currentX, currentY, currentZ},
                    electricX_[index] + electricY_[index] + electricZ_[index],
                    plaquetteTrace,
                });
            }
        }
    }

    return samples;
}

std::size_t CoupledSu2FermionLattice::flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return x + (spec_.sizeX * (y + (spec_.sizeY * z)));
}

std::vector<FermionColorSpinor> CoupledSu2FermionLattice::derivative(const std::vector<FermionColorSpinor>& state) const {
    std::vector<FermionColorSpinor> dPsi(state.size(), zeroSpinor());
    const double inverseSpacing = 1.0 / std::max(spec_.spacing, 1.0e-12);
    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                Spinor hopping = zeroSpinor();

                const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
                const auto xMinus = wrapIndex(x, spec_.sizeX, -1);
                const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
                const auto yMinus = wrapIndex(y, spec_.sizeY, -1);
                const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
                const auto zMinus = wrapIndex(z, spec_.sizeZ, -1);

                hopping += (applyLink(linkX_[index], state[flatIndex(xPlus, y, z)]) -
                            applyLink(linkX_[flatIndex(xMinus, y, z)].conjugate(), state[flatIndex(xMinus, y, z)])) *
                           Complex{0.5 * staggeredEta(0, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                hopping += (applyLink(linkY_[index], state[flatIndex(x, yPlus, z)]) -
                            applyLink(linkY_[flatIndex(x, yMinus, z)].conjugate(), state[flatIndex(x, yMinus, z)])) *
                           Complex{0.5 * staggeredEta(1, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                hopping += (applyLink(linkZ_[index], state[flatIndex(x, y, zPlus)]) -
                            applyLink(linkZ_[flatIndex(x, y, zMinus)].conjugate(), state[flatIndex(x, y, zMinus)])) *
                           Complex{0.5 * staggeredEta(2, x, y, z) * spec_.hopping * inverseSpacing, 0.0};

                const Complex massFactor{0.0, -spec_.mass * staggeredParity(x, y, z)};
                dPsi[index] = hopping + (state[index] * massFactor);
            }
        }
    }
    return dPsi;
}

math::Quaternion CoupledSu2FermionLattice::plaquetteXY(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    return (
        linkX_[flatIndex(x, y, z)] *
        linkY_[flatIndex(xPlus, y, z)] *
        linkX_[flatIndex(x, yPlus, z)].conjugate() *
        linkY_[flatIndex(x, y, z)].conjugate()).normalized();
}

math::Quaternion CoupledSu2FermionLattice::plaquetteXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return (
        linkX_[flatIndex(x, y, z)] *
        linkZ_[flatIndex(xPlus, y, z)] *
        linkX_[flatIndex(x, y, zPlus)].conjugate() *
        linkZ_[flatIndex(x, y, z)].conjugate()).normalized();
}

math::Quaternion CoupledSu2FermionLattice::plaquetteYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
    const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
    return (
        linkY_[flatIndex(x, y, z)] *
        linkZ_[flatIndex(x, yPlus, z)] *
        linkY_[flatIndex(x, y, zPlus)].conjugate() *
        linkZ_[flatIndex(x, y, z)].conjugate()).normalized();
}

}  // namespace physicsmade::field_theory
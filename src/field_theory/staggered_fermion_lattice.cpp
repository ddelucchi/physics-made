#include "physicsmade/field_theory/staggered_fermion_lattice.hpp"

#include <algorithm>
#include <cmath>

#include "physicsmade/common/config.hpp"

namespace {

using Complex = std::complex<double>;
using Spinor = physicsmade::field_theory::FermionColorSpinor;

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
        Complex{direction.x / (std::sqrt(2.0) * denominator), direction.y / (std::sqrt(2.0) * denominator)}
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

}  // namespace

namespace physicsmade::field_theory {

StaggeredFermionLattice::StaggeredFermionLattice(StaggeredFermionLatticeSpec spec)
    : spec_(spec),
      psi_(spec.sizeX * spec.sizeY * spec.sizeZ, zeroSpinor()),
      linkX_(psi_.size(), math::Quaternion::identity()),
      linkY_(psi_.size(), math::Quaternion::identity()),
      linkZ_(psi_.size(), math::Quaternion::identity()) {}

void StaggeredFermionLattice::seedGaussianPacket(
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

void StaggeredFermionLattice::seedColorFlux(
    double amplitude,
    const math::Vector3& colorDirection) {
    const auto color = normalizedOrFallback(colorDirection);
    const double extent = static_cast<double>(std::max<std::size_t>(1, spec_.sizeX));

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const auto algebra = color * (spec_.gaugeCoupling * amplitude * static_cast<double>(x) / extent);
                linkY_[index] = (linkFromAlgebra(algebra) * linkY_[index]).normalized();
            }
        }
    }
}

void StaggeredFermionLattice::step(double dtSeconds, int iterations) {
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
    }
}

StaggeredFermionObservables StaggeredFermionLattice::observables() const {
    StaggeredFermionObservables observables{};
    const double cellVolume = spec_.spacing * spec_.spacing * spec_.spacing;
    const double inverseSpacing = 1.0 / std::max(spec_.spacing, 1.0e-12);

    for (std::size_t z = 0; z < spec_.sizeZ; ++z) {
        for (std::size_t y = 0; y < spec_.sizeY; ++y) {
            for (std::size_t x = 0; x < spec_.sizeX; ++x) {
                const auto index = flatIndex(x, y, z);
                const double density = siteDensity(psi_[index]);
                const double parity = staggeredParity(x, y, z);
                observables.totalProbability += density * cellVolume;
                observables.massActivity += std::abs(spec_.mass) * density * cellVolume;
                observables.maxSiteDensity = std::max(observables.maxSiteDensity, density);
                observables.staggeredCharge += parity * density * cellVolume;

                const auto xPlus = wrapIndex(x, spec_.sizeX, 1);
                const auto yPlus = wrapIndex(y, spec_.sizeY, 1);
                const auto zPlus = wrapIndex(z, spec_.sizeZ, 1);
                const auto forwardX = applyLink(linkX_[index], psi_[flatIndex(xPlus, y, z)]);
                const auto forwardY = applyLink(linkY_[index], psi_[flatIndex(x, yPlus, z)]);
                const auto forwardZ = applyLink(linkZ_[index], psi_[flatIndex(x, y, zPlus)]);
                const Complex currentX = staggeredEta(0, x, y, z) * overlap(psi_[index], forwardX);
                const Complex currentY = staggeredEta(1, x, y, z) * overlap(psi_[index], forwardY);
                const Complex currentZ = staggeredEta(2, x, y, z) * overlap(psi_[index], forwardZ);
                observables.meanCurrent.x += inverseSpacing * std::imag(currentX);
                observables.meanCurrent.y += inverseSpacing * std::imag(currentY);
                observables.meanCurrent.z += inverseSpacing * std::imag(currentZ);

                const auto xMinus = wrapIndex(x, spec_.sizeX, -1);
                const auto yMinus = wrapIndex(y, spec_.sizeY, -1);
                const auto zMinus = wrapIndex(z, spec_.sizeZ, -1);
                const auto backwardX = applyLink(linkX_[flatIndex(xMinus, y, z)].conjugate(), psi_[flatIndex(xMinus, y, z)]);
                const auto backwardY = applyLink(linkY_[flatIndex(x, yMinus, z)].conjugate(), psi_[flatIndex(x, yMinus, z)]);
                const auto backwardZ = applyLink(linkZ_[flatIndex(x, y, zMinus)].conjugate(), psi_[flatIndex(x, y, zMinus)]);

                const auto hoppingX = (forwardX - backwardX) * Complex{0.5 * staggeredEta(0, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                const auto hoppingY = (forwardY - backwardY) * Complex{0.5 * staggeredEta(1, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                const auto hoppingZ = (forwardZ - backwardZ) * Complex{0.5 * staggeredEta(2, x, y, z) * spec_.hopping * inverseSpacing, 0.0};
                observables.kineticActivity +=
                    (siteDensity(hoppingX) + siteDensity(hoppingY) + siteDensity(hoppingZ)) * cellVolume;
            }
        }
    }

    if (!psi_.empty()) {
        observables.meanCurrent *= 1.0 / static_cast<double>(psi_.size());
    }

    observables.totalActivity = observables.kineticActivity + observables.massActivity;
    return observables;
}

std::size_t StaggeredFermionLattice::flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept {
    return x + (spec_.sizeX * (y + (spec_.sizeY * z)));
}

std::vector<FermionColorSpinor> StaggeredFermionLattice::derivative(const std::vector<FermionColorSpinor>& state) const {
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

}  // namespace physicsmade::field_theory
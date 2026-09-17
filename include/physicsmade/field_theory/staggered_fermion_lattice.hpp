#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::field_theory {

using FermionColorSpinor = std::array<std::complex<double>, 2>;

struct StaggeredFermionLatticeSpec {
    std::size_t sizeX{16};
    std::size_t sizeY{16};
    std::size_t sizeZ{8};
    double spacing{0.25};
    double mass{0.6};
    double hopping{1.0};
    double gaugeCoupling{1.0};
};

struct StaggeredFermionObservables {
    double totalProbability{0.0};
    double kineticActivity{0.0};
    double massActivity{0.0};
    double totalActivity{0.0};
    double maxSiteDensity{0.0};
    double staggeredCharge{0.0};
    math::Vector3 meanCurrent{};
};

class StaggeredFermionLattice {
  public:
    explicit StaggeredFermionLattice(StaggeredFermionLatticeSpec spec = {});

    const StaggeredFermionLatticeSpec& spec() const noexcept {
        return spec_;
    }

    const std::vector<FermionColorSpinor>& psi() const noexcept {
        return psi_;
    }

    void seedGaussianPacket(
        double amplitude,
        const math::Vector3& center,
        double width,
        const math::Vector3& waveVector,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void seedColorFlux(
        double amplitude,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void step(double dtSeconds, int iterations = 1);
    StaggeredFermionObservables observables() const;

  private:
    std::size_t flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    std::vector<FermionColorSpinor> derivative(const std::vector<FermionColorSpinor>& state) const;

    StaggeredFermionLatticeSpec spec_{};
    std::vector<FermionColorSpinor> psi_{};
    std::vector<math::Quaternion> linkX_{};
    std::vector<math::Quaternion> linkY_{};
    std::vector<math::Quaternion> linkZ_{};
};

}  // namespace physicsmade::field_theory
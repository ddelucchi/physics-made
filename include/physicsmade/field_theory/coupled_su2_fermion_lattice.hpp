#pragma once

#include <cstddef>
#include <vector>

#include "physicsmade/field_theory/staggered_fermion_lattice.hpp"
#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::field_theory {

struct CoupledSu2FermionLatticeSpec {
    std::size_t sizeX{16};
    std::size_t sizeY{16};
    std::size_t sizeZ{8};
    double spacing{0.25};
    double mass{0.55};
    double hopping{1.0};
    double gaugeCoupling{0.85};
    double fermionBackreaction{0.35};
};

struct CoupledSu2FermionObservables {
    double totalProbability{0.0};
    double fermionKineticActivity{0.0};
    double fermionMassActivity{0.0};
    double gaugeElectricEnergy{0.0};
    double gaugeMagneticEnergy{0.0};
    double interactionActivity{0.0};
    double totalEnergy{0.0};
    double maxSiteDensity{0.0};
    double meanPlaquetteTrace{0.0};
    math::Vector3 meanCurrent{};
    math::Vector3 meanColorField{};
};

struct CoupledSu2FermionSiteSample {
    math::Vector3 position{};
    double density{0.0};
    math::Vector3 current{};
    math::Vector3 electricField{};
    double plaquetteTrace{1.0};
};

class CoupledSu2FermionLattice {
  public:
    explicit CoupledSu2FermionLattice(CoupledSu2FermionLatticeSpec spec = {});

    const CoupledSu2FermionLatticeSpec& spec() const noexcept {
        return spec_;
    }

    void seedFermionGaussianPacket(
        double amplitude,
        const math::Vector3& center,
        double width,
        const math::Vector3& waveVector,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void seedGaugeStandingWave(
        double amplitude,
        const math::Vector3& waveVector,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void seedColorFlux(
        double amplitude,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void step(double dtSeconds, int iterations = 1);
    CoupledSu2FermionObservables observables() const;
        std::vector<CoupledSu2FermionSiteSample> sampleSites() const;

  private:
    std::size_t flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    std::vector<FermionColorSpinor> derivative(const std::vector<FermionColorSpinor>& state) const;
    math::Quaternion plaquetteXY(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    math::Quaternion plaquetteXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    math::Quaternion plaquetteYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;

    CoupledSu2FermionLatticeSpec spec_{};
    std::vector<FermionColorSpinor> psi_{};
    std::vector<math::Quaternion> linkX_{};
    std::vector<math::Quaternion> linkY_{};
    std::vector<math::Quaternion> linkZ_{};
    std::vector<math::Vector3> electricX_{};
    std::vector<math::Vector3> electricY_{};
    std::vector<math::Vector3> electricZ_{};
};

}  // namespace physicsmade::field_theory
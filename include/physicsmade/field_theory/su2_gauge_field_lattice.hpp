#pragma once

#include <cstddef>
#include <vector>

#include "physicsmade/math/quaternion.hpp"
#include "physicsmade/math/vector3.hpp"

namespace physicsmade::field_theory {

struct SU2GaugeLatticeSpec {
    std::size_t sizeX{16};
    std::size_t sizeY{16};
    std::size_t sizeZ{8};
    double spacing{0.25};
    double coupling{1.0};
};

struct SU2GaugeObservables {
    double electricEnergy{0.0};
    double magneticEnergy{0.0};
    double totalEnergy{0.0};
    double meanPlaquetteTrace{0.0};
    double maxElectricField{0.0};
    math::Vector3 meanColorField{};
};

class SU2GaugeFieldLattice {
  public:
    explicit SU2GaugeFieldLattice(SU2GaugeLatticeSpec spec = {});

    const SU2GaugeLatticeSpec& spec() const noexcept {
        return spec_;
    }

    void seedStandingWave(
        double amplitude,
        const math::Vector3& waveVector,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void seedColorMagneticFlux(
        double amplitude,
        const math::Vector3& colorDirection = {0.0, 0.0, 1.0});
    void step(double dtSeconds, int iterations = 1);
    SU2GaugeObservables observables() const;

  private:
    std::size_t flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    math::Quaternion plaquetteXY(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    math::Quaternion plaquetteXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    math::Quaternion plaquetteYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;

    SU2GaugeLatticeSpec spec_{};
    std::vector<math::Quaternion> linkX_{};
    std::vector<math::Quaternion> linkY_{};
    std::vector<math::Quaternion> linkZ_{};
    std::vector<math::Vector3> electricX_{};
    std::vector<math::Vector3> electricY_{};
    std::vector<math::Vector3> electricZ_{};
};

}  // namespace physicsmade::field_theory
#pragma once

#include <cstddef>
#include <vector>

#include "physicsmade/math/vector3.hpp"

namespace physicsmade::field_theory {

struct U1GaugeLatticeSpec {
    std::size_t sizeX{20};
    std::size_t sizeY{20};
    std::size_t sizeZ{20};
    double spacing{0.2};
    double coupling{1.0};
};

struct U1GaugeObservables {
    double electricEnergy{0.0};
    double magneticEnergy{0.0};
    double totalEnergy{0.0};
  double meanPlaquette{0.0};
    double maxElectricField{0.0};
};

class U1GaugeFieldLattice {
  public:
    explicit U1GaugeFieldLattice(U1GaugeLatticeSpec spec = {});

    const U1GaugeLatticeSpec& spec() const noexcept {
        return spec_;
    }

    void seedStandingWave(double amplitude, const math::Vector3& waveVector);
    void seedMagneticFluxSheet(double amplitude);
    void step(double dtSeconds, int iterations = 1);
    U1GaugeObservables observables() const;

  private:
    std::size_t flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    double plaquettePhaseXY(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    double plaquettePhaseXZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    double plaquettePhaseYZ(std::size_t x, std::size_t y, std::size_t z) const noexcept;

    U1GaugeLatticeSpec spec_{};
    std::vector<double> linkX_{};
    std::vector<double> linkY_{};
    std::vector<double> linkZ_{};
    std::vector<double> electricX_{};
    std::vector<double> electricY_{};
    std::vector<double> electricZ_{};
};

}  // namespace physicsmade::field_theory
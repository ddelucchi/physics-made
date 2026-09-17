#pragma once

#include <cstddef>
#include <vector>

#include "physicsmade/math/vector3.hpp"

namespace physicsmade::field_theory {

struct RectangularLatticeSpec {
    std::size_t sizeX{24};
    std::size_t sizeY{24};
    std::size_t sizeZ{24};
    double spacing{0.1};
    double mass{1.0};
    double selfCoupling{0.0};
};

struct RectangularFieldObservables {
    double totalEnergy{0.0};
    double gradientEnergy{0.0};
    double potentialEnergy{0.0};
    double meanField{0.0};
    double maxAmplitude{0.0};
};

class RectangularScalarFieldLattice {
  public:
    explicit RectangularScalarFieldLattice(RectangularLatticeSpec spec = {});

    const RectangularLatticeSpec& spec() const noexcept {
        return spec_;
    }

    const std::vector<double>& phi() const noexcept {
        return phi_;
    }

    const std::vector<double>& pi() const noexcept {
        return pi_;
    }

    void seedGaussianPacket(
        double amplitude,
        const math::Vector3& center,
        double width,
        const math::Vector3& waveVector);
    void step(double dtSeconds, int iterations = 1);
    RectangularFieldObservables observables() const;

  private:
    std::size_t flatIndex(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    double laplacianAt(std::size_t x, std::size_t y, std::size_t z) const noexcept;
    double potentialDensity(double value) const noexcept;

    RectangularLatticeSpec spec_{};
    std::vector<double> phi_{};
    std::vector<double> pi_{};
};

}  // namespace physicsmade::field_theory
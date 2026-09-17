#pragma once

#include <cstddef>
#include <vector>

namespace physicsmade::field_theory {

struct LatticeSpec {
    std::size_t siteCount{64};
    double spacing{0.1};
    double mass{1.0};
    double selfCoupling{0.0};
};

struct FieldObservables {
    double totalEnergy{0.0};
    double meanField{0.0};
    double maxAmplitude{0.0};
    double nearestNeighborCorrelation{0.0};
};

class ScalarFieldLattice {
  public:
    explicit ScalarFieldLattice(LatticeSpec spec = {});

    const LatticeSpec& spec() const noexcept {
        return spec_;
    }

    const std::vector<double>& phi() const noexcept {
        return phi_;
    }

    const std::vector<double>& pi() const noexcept {
        return pi_;
    }

    void seedGaussianPacket(double amplitude, double center, double width, double waveNumber);
    void step(double dtSeconds, int iterations = 1);
    FieldObservables observables() const;

  private:
    double laplacianAt(std::size_t index) const noexcept;
    double potentialDensity(double value) const noexcept;

    LatticeSpec spec_{};
    std::vector<double> phi_{};
    std::vector<double> pi_{};
};

}  // namespace physicsmade::field_theory

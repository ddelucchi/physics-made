#pragma once

#include <string>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/spacetime/metric_tensor.hpp"

namespace physicsmade::spacetime {

class KerrMetric final : public MetricTensor {
  public:
    explicit KerrMetric(
        double centralMass,
        double dimensionlessSpin = 0.6,
        math::Vector3 origin = {},
        double gravitationalConstant = common::kGravitationalConstant,
        double speedOfLight = common::kSpeedOfLight);

    Tensor4 covariant(const math::FourVector& position) const override;
    std::string name() const override;

    double centralMass() const noexcept {
        return centralMass_;
    }

    double dimensionlessSpin() const noexcept {
        return dimensionlessSpin_;
    }

    double gravitationalRadius() const noexcept;
    double spinParameter() const noexcept;
    double eventHorizonRadius() const noexcept;

  private:
    double centralMass_{0.0};
    double dimensionlessSpin_{0.0};
    math::Vector3 origin_{};
    double gravitationalConstant_{common::kGravitationalConstant};
    double speedOfLight_{common::kSpeedOfLight};
};

}  // namespace physicsmade::spacetime
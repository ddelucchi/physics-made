#pragma once

#include <string>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/spacetime/metric_tensor.hpp"

namespace physicsmade::spacetime {

class WeakFieldSphericalMassMetric final : public MetricTensor {
  public:
    explicit WeakFieldSphericalMassMetric(
        double centralMass,
        math::Vector3 origin = {},
        double gravitationalConstant = common::kGravitationalConstant,
        double speedOfLight = common::kSpeedOfLight)
        : centralMass_(centralMass),
          origin_(origin),
          gravitationalConstant_(gravitationalConstant),
          speedOfLight_(speedOfLight) {}

    Tensor4 covariant(const math::FourVector& position) const override;
    std::string name() const override;

    double centralMass() const noexcept {
        return centralMass_;
    }

  private:
    double centralMass_{0.0};
    math::Vector3 origin_{};
    double gravitationalConstant_{common::kGravitationalConstant};
    double speedOfLight_{common::kSpeedOfLight};
};

}  // namespace physicsmade::spacetime

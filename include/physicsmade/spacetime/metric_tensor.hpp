#pragma once

#include <array>
#include <string>

#include "physicsmade/math/four_vector.hpp"

namespace physicsmade::spacetime {

using Tensor4 = std::array<std::array<double, 4>, 4>;

class MetricTensor {
  public:
    virtual ~MetricTensor() = default;

    virtual Tensor4 covariant(const math::FourVector& position) const = 0;
    virtual std::string name() const = 0;
};

class MinkowskiMetric final : public MetricTensor {
  public:
    Tensor4 covariant(const math::FourVector& position) const override;
    std::string name() const override;
};

}  // namespace physicsmade::spacetime

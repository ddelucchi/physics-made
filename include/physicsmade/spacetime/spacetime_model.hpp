#pragma once

#include <cstddef>
#include <memory>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/four_vector.hpp"
#include "physicsmade/spacetime/metric_tensor.hpp"

namespace physicsmade::spacetime {

class SpacetimeModel {
  public:
    explicit SpacetimeModel(
        std::shared_ptr<const MetricTensor> metric = std::make_shared<MinkowskiMetric>(),
        double speedOfLight = common::kSpeedOfLight)
        : metric_(std::move(metric)), speedOfLight_(speedOfLight) {}

    const MetricTensor& metric() const noexcept {
        return *metric_;
    }

    double speedOfLight() const noexcept {
        return speedOfLight_;
    }

    double intervalSquared(const math::FourVector& a, const math::FourVector& b) const noexcept {
        const auto delta = b - a;
        const auto tensor = metric().covariant(a);
        const double components[4] = {delta.t, delta.x, delta.y, delta.z};

        double interval = 0.0;
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                interval += tensor[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)] * components[row] * components[column];
            }
        }

        return interval;
    }

  private:
    std::shared_ptr<const MetricTensor> metric_;
    double speedOfLight_{common::kSpeedOfLight};
};

}  // namespace physicsmade::spacetime

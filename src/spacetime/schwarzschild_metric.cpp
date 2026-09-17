#include "physicsmade/spacetime/schwarzschild_metric.hpp"

#include <algorithm>
#include <cmath>

namespace physicsmade::spacetime {

Tensor4 SchwarzschildMetric::covariant(const math::FourVector& position) const {
    const math::Vector3 displacement{
        position.x - origin_.x,
        position.y - origin_.y,
        position.z - origin_.z,
    };

    const double isotropicRadius = std::max(displacement.norm(), common::kEpsilon);
    const double compactness = (gravitationalConstant_ * centralMass_) / (2.0 * speedOfLight_ * speedOfLight_ * isotropicRadius);
    const double clampedCompactness = std::min(compactness, 1.0 - 1.0e-12);
    const double temporalNumerator = 1.0 - clampedCompactness;
    const double temporalDenominator = 1.0 + clampedCompactness;
    const double temporalScale = -std::pow(temporalNumerator / temporalDenominator, 2.0);
    const double spatialScale = std::pow(temporalDenominator, 4.0);

    return {{
        {{temporalScale, 0.0, 0.0, 0.0}},
        {{0.0, spatialScale, 0.0, 0.0}},
        {{0.0, 0.0, spatialScale, 0.0}},
        {{0.0, 0.0, 0.0, spatialScale}},
    }};
}

std::string SchwarzschildMetric::name() const {
    return "schwarzschild-isotropic";
}

double SchwarzschildMetric::schwarzschildRadius() const noexcept {
    return (2.0 * gravitationalConstant_ * centralMass_) / (speedOfLight_ * speedOfLight_);
}

double SchwarzschildMetric::isotropicHorizonRadius() const noexcept {
    return 0.25 * schwarzschildRadius();
}

}  // namespace physicsmade::spacetime
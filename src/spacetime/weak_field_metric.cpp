#include "physicsmade/spacetime/weak_field_metric.hpp"

#include <algorithm>

namespace physicsmade::spacetime {

Tensor4 WeakFieldSphericalMassMetric::covariant(const math::FourVector& position) const {
    const math::Vector3 displacement{
        position.x - origin_.x,
        position.y - origin_.y,
        position.z - origin_.z,
    };

    const double radius = std::max(displacement.norm(), common::kEpsilon);
    const double potential = -(gravitationalConstant_ * centralMass_) / radius;
    const double correction = (2.0 * potential) / (speedOfLight_ * speedOfLight_);

    return {{
        {{-(1.0 + correction), 0.0, 0.0, 0.0}},
        {{0.0, 1.0 - correction, 0.0, 0.0}},
        {{0.0, 0.0, 1.0 - correction, 0.0}},
        {{0.0, 0.0, 0.0, 1.0 - correction}},
    }};
}

std::string WeakFieldSphericalMassMetric::name() const {
    return "weak-field-spherical-mass";
}

}  // namespace physicsmade::spacetime

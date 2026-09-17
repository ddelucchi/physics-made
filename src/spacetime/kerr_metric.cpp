#include "physicsmade/spacetime/kerr_metric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace physicsmade::spacetime {

KerrMetric::KerrMetric(
    double centralMass,
    double dimensionlessSpin,
    math::Vector3 origin,
    double gravitationalConstant,
    double speedOfLight)
    : centralMass_(centralMass),
      dimensionlessSpin_(dimensionlessSpin),
      origin_(origin),
      gravitationalConstant_(gravitationalConstant),
      speedOfLight_(speedOfLight) {
    if (centralMass_ <= 0.0) {
        throw std::runtime_error("Kerr metric requires a positive central mass");
    }
    if (speedOfLight_ <= 0.0 || gravitationalConstant_ <= 0.0) {
        throw std::runtime_error("Kerr metric requires positive physical constants");
    }
    if (std::abs(dimensionlessSpin_) >= 1.0) {
        throw std::runtime_error("Kerr metric dimensionless spin must stay below the extremal limit");
    }
}

Tensor4 KerrMetric::covariant(const math::FourVector& position) const {
    const auto displacement = position.spatial() - origin_;
    const double x = displacement.x;
    const double y = displacement.y;
    const double z = displacement.z;
    const double a = spinParameter();
    const double aSquared = a * a;
    const double rhoSquared = (x * x) + (y * y) + (z * z);
    const double radialTerm = rhoSquared - aSquared;
    const double radical = std::sqrt(std::max((radialTerm * radialTerm) + (4.0 * aSquared * z * z), common::kEpsilon));
    const double rSquared = std::max(0.5 * (radialTerm + radical), common::kEpsilon);
    const double r = std::sqrt(rSquared);
    const double denominator = std::max(rSquared + aSquared, common::kEpsilon);
    const double hDenominator = std::max((rSquared * rSquared) + (aSquared * z * z), common::kEpsilon);
    const double h = gravitationalRadius() * r * rSquared / hDenominator;

    const std::array<double, 4> nullVector{
        1.0,
        ((r * x) + (a * y)) / denominator,
        ((r * y) - (a * x)) / denominator,
        z / std::max(r, common::kEpsilon),
    };

    Tensor4 metric{{
        {{-1.0, 0.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0, 0.0}},
        {{0.0, 0.0, 1.0, 0.0}},
        {{0.0, 0.0, 0.0, 1.0}},
    }};

    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            metric[row][column] += 2.0 * h * nullVector[row] * nullVector[column];
        }
    }

    return metric;
}

std::string KerrMetric::name() const {
    return "kerr-schild-cartesian";
}

double KerrMetric::gravitationalRadius() const noexcept {
    return (gravitationalConstant_ * centralMass_) / (speedOfLight_ * speedOfLight_);
}

double KerrMetric::spinParameter() const noexcept {
    return dimensionlessSpin_ * gravitationalRadius();
}

double KerrMetric::eventHorizonRadius() const noexcept {
    const double radius = gravitationalRadius();
    const double spin = spinParameter();
    return radius + std::sqrt(std::max((radius * radius) - (spin * spin), 0.0));
}

}  // namespace physicsmade::spacetime
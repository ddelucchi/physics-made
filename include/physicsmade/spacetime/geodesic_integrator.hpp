#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "physicsmade/math/four_vector.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/spacetime/spacetime_model.hpp"

namespace physicsmade::spacetime {

using ChristoffelSymbols = std::array<Tensor4, 4>;

struct GeodesicState {
    math::FourVector position{};
    math::FourVector tangent{};
    double affineParameter{0.0};
};

struct GeodesicIntegratorOptions {
    double affineStep{1.0};
    double absoluteDerivativeStep{1.0};
    double relativeDerivativeStep{1.0e-6};
};

class NumericGeodesicIntegrator {
  public:
    explicit NumericGeodesicIntegrator(
        SpacetimeModel spacetime = SpacetimeModel{},
        GeodesicIntegratorOptions options = {});

    const SpacetimeModel& spacetime() const noexcept {
        return spacetime_;
    }

    const GeodesicIntegratorOptions& options() const noexcept {
        return options_;
    }

    ChristoffelSymbols christoffelSymbols(const math::FourVector& position) const;
    GeodesicState step(const GeodesicState& state) const;
    std::vector<GeodesicState> integrate(const GeodesicState& initialState, std::size_t steps) const;

  private:
    SpacetimeModel spacetime_{};
    GeodesicIntegratorOptions options_{};
};

GeodesicState makeTimelikeGeodesicState(
    const SpacetimeModel& spacetime,
    const math::FourVector& position,
    const math::Vector3& spatialVelocity,
    double affineParameter = 0.0);

double tangentNormSquared(const SpacetimeModel& spacetime, const GeodesicState& state) noexcept;
double coordinateSpeed(const GeodesicState& state) noexcept;

}  // namespace physicsmade::spacetime
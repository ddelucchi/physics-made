#include "physicsmade/spacetime/geodesic_integrator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

using StateVector = std::array<double, 8>;

double fourVectorComponent(const physicsmade::math::FourVector& vector, std::size_t index) {
    switch (index) {
        case 0:
            return vector.t;
        case 1:
            return vector.x;
        case 2:
            return vector.y;
        case 3:
            return vector.z;
        default:
            return 0.0;
    }
}

void setFourVectorComponent(physicsmade::math::FourVector& vector, std::size_t index, double value) {
    switch (index) {
        case 0:
            vector.t = value;
            break;
        case 1:
            vector.x = value;
            break;
        case 2:
            vector.y = value;
            break;
        case 3:
            vector.z = value;
            break;
        default:
            break;
    }
}

physicsmade::math::FourVector addScaled(
    const physicsmade::math::FourVector& base,
    std::size_t index,
    double delta) {
    auto result = base;
    setFourVectorComponent(result, index, fourVectorComponent(result, index) + delta);
    return result;
}

double adaptiveDerivativeStep(
    const physicsmade::spacetime::GeodesicIntegratorOptions& options,
    double coordinateValue) {
    return std::max(
        options.absoluteDerivativeStep,
        options.relativeDerivativeStep * std::max(1.0, std::abs(coordinateValue)));
}

physicsmade::spacetime::Tensor4 invertTensor(const physicsmade::spacetime::Tensor4& tensor) {
    std::array<std::array<double, 8>, 4> augmented{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            augmented[row][column] = tensor[row][column];
            augmented[row][column + 4] = (row == column) ? 1.0 : 0.0;
        }
    }

    for (std::size_t pivot = 0; pivot < 4; ++pivot) {
        std::size_t pivotRow = pivot;
        double pivotMagnitude = std::abs(augmented[pivotRow][pivot]);
        for (std::size_t row = pivot + 1; row < 4; ++row) {
            const double magnitude = std::abs(augmented[row][pivot]);
            if (magnitude > pivotMagnitude) {
                pivotMagnitude = magnitude;
                pivotRow = row;
            }
        }

        if (pivotMagnitude <= physicsmade::common::kEpsilon) {
            throw std::runtime_error("metric tensor is singular and cannot be inverted");
        }

        if (pivotRow != pivot) {
            std::swap(augmented[pivot], augmented[pivotRow]);
        }

        const double pivotValue = augmented[pivot][pivot];
        for (double& entry : augmented[pivot]) {
            entry /= pivotValue;
        }

        for (std::size_t row = 0; row < 4; ++row) {
            if (row == pivot) {
                continue;
            }

            const double factor = augmented[row][pivot];
            for (std::size_t column = 0; column < 8; ++column) {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }

    physicsmade::spacetime::Tensor4 inverse{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            inverse[row][column] = augmented[row][column + 4];
        }
    }

    return inverse;
}

double contractMetric(
    const physicsmade::spacetime::Tensor4& metric,
    const std::array<double, 4>& vector) {
    double value = 0.0;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            value += metric[row][column] * vector[row] * vector[column];
        }
    }
    return value;
}

StateVector toStateVector(const physicsmade::spacetime::GeodesicState& state) {
    return {
        state.position.t,
        state.position.x,
        state.position.y,
        state.position.z,
        state.tangent.t,
        state.tangent.x,
        state.tangent.y,
        state.tangent.z,
    };
}

physicsmade::spacetime::GeodesicState fromStateVector(
    const StateVector& stateVector,
    double affineParameter) {
    return {
        {stateVector[0], stateVector[1], stateVector[2], stateVector[3]},
        {stateVector[4], stateVector[5], stateVector[6], stateVector[7]},
        affineParameter,
    };
}

StateVector scaledAdd(const StateVector& base, const StateVector& delta, double scale) {
    StateVector result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = base[index] + (scale * delta[index]);
    }
    return result;
}

StateVector combineRk4(
    const StateVector& base,
    const StateVector& k1,
    const StateVector& k2,
    const StateVector& k3,
    const StateVector& k4,
    double step) {
    StateVector result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = base[index] + ((step / 6.0) * (k1[index] + (2.0 * k2[index]) + (2.0 * k3[index]) + k4[index]));
    }
    return result;
}

}  // namespace

namespace physicsmade::spacetime {

NumericGeodesicIntegrator::NumericGeodesicIntegrator(
    SpacetimeModel spacetime,
    GeodesicIntegratorOptions options)
    : spacetime_(std::move(spacetime)), options_(options) {
    if (options_.affineStep <= 0.0) {
        throw std::runtime_error("geodesic affine step must be positive");
    }
    if (options_.absoluteDerivativeStep <= 0.0 || options_.relativeDerivativeStep <= 0.0) {
        throw std::runtime_error("geodesic derivative steps must be positive");
    }
}

ChristoffelSymbols NumericGeodesicIntegrator::christoffelSymbols(const math::FourVector& position) const {
    const auto metric = spacetime_.metric().covariant(position);
    const auto inverseMetric = invertTensor(metric);

    std::array<Tensor4, 4> derivatives{};
    for (std::size_t coordinate = 0; coordinate < 4; ++coordinate) {
        const double step = adaptiveDerivativeStep(options_, fourVectorComponent(position, coordinate));
        const auto forward = spacetime_.metric().covariant(addScaled(position, coordinate, step));
        const auto backward = spacetime_.metric().covariant(addScaled(position, coordinate, -step));
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t column = 0; column < 4; ++column) {
                derivatives[coordinate][row][column] = (forward[row][column] - backward[row][column]) / (2.0 * step);
            }
        }
    }

    ChristoffelSymbols symbols{};
    for (std::size_t mu = 0; mu < 4; ++mu) {
        for (std::size_t alpha = 0; alpha < 4; ++alpha) {
            for (std::size_t beta = 0; beta < 4; ++beta) {
                double value = 0.0;
                for (std::size_t nu = 0; nu < 4; ++nu) {
                    value += inverseMetric[mu][nu] *
                             (derivatives[alpha][beta][nu] + derivatives[beta][alpha][nu] - derivatives[nu][alpha][beta]);
                }
                symbols[mu][alpha][beta] = 0.5 * value;
            }
        }
    }

    return symbols;
}

GeodesicState NumericGeodesicIntegrator::step(const GeodesicState& state) const {
    const auto derivative = [this](const StateVector& stateVector) {
        const auto localState = fromStateVector(stateVector, 0.0);
        const auto symbols = christoffelSymbols(localState.position);
        std::array<double, 4> tangent{
            localState.tangent.t,
            localState.tangent.x,
            localState.tangent.y,
            localState.tangent.z,
        };

        StateVector value{};
        value[0] = tangent[0];
        value[1] = tangent[1];
        value[2] = tangent[2];
        value[3] = tangent[3];
        for (std::size_t mu = 0; mu < 4; ++mu) {
            double acceleration = 0.0;
            for (std::size_t alpha = 0; alpha < 4; ++alpha) {
                for (std::size_t beta = 0; beta < 4; ++beta) {
                    acceleration -= symbols[mu][alpha][beta] * tangent[alpha] * tangent[beta];
                }
            }
            value[4 + mu] = acceleration;
        }
        return value;
    };

    const auto y0 = toStateVector(state);
    const auto k1 = derivative(y0);
    const auto k2 = derivative(scaledAdd(y0, k1, 0.5 * options_.affineStep));
    const auto k3 = derivative(scaledAdd(y0, k2, 0.5 * options_.affineStep));
    const auto k4 = derivative(scaledAdd(y0, k3, options_.affineStep));
    return fromStateVector(combineRk4(y0, k1, k2, k3, k4, options_.affineStep), state.affineParameter + options_.affineStep);
}

std::vector<GeodesicState> NumericGeodesicIntegrator::integrate(const GeodesicState& initialState, std::size_t steps) const {
    std::vector<GeodesicState> states;
    states.reserve(steps + 1);
    states.push_back(initialState);
    for (std::size_t stepIndex = 0; stepIndex < steps; ++stepIndex) {
        states.push_back(step(states.back()));
    }
    return states;
}

GeodesicState makeTimelikeGeodesicState(
    const SpacetimeModel& spacetime,
    const math::FourVector& position,
    const math::Vector3& spatialVelocity,
    double affineParameter) {
    const auto metric = spacetime.metric().covariant(position);
    const std::array<double, 4> coordinateVelocity{
        spacetime.speedOfLight(),
        spatialVelocity.x,
        spatialVelocity.y,
        spatialVelocity.z,
    };
    const double norm = contractMetric(metric, coordinateVelocity);
    if (norm >= 0.0) {
        throw std::runtime_error("timelike geodesic construction requires a subluminal coordinate velocity");
    }

    const double scale = spacetime.speedOfLight() / std::sqrt(-norm);
    return {
        position,
        {
            scale * coordinateVelocity[0],
            scale * coordinateVelocity[1],
            scale * coordinateVelocity[2],
            scale * coordinateVelocity[3],
        },
        affineParameter,
    };
}

double tangentNormSquared(const SpacetimeModel& spacetime, const GeodesicState& state) noexcept {
    const auto metric = spacetime.metric().covariant(state.position);
    const std::array<double, 4> tangent{
        state.tangent.t,
        state.tangent.x,
        state.tangent.y,
        state.tangent.z,
    };
    return contractMetric(metric, tangent);
}

double coordinateSpeed(const GeodesicState& state) noexcept {
    const double temporal = std::abs(state.tangent.t);
    if (temporal <= common::kEpsilon) {
        return 0.0;
    }
    return common::kSpeedOfLight * state.tangent.spatial().norm() / temporal;
}

}  // namespace physicsmade::spacetime
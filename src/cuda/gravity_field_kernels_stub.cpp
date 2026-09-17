#include "physicsmade/cuda/field_grid.hpp"

#include <cmath>
#include <stdexcept>

namespace {

physicsmade::cuda::FieldSample sampleGravityPoint(
    const std::vector<physicsmade::scene::ObjectState>& sources,
    const physicsmade::math::Vector3& samplePosition,
    double gravitationalConstant) {
    physicsmade::cuda::FieldSample sample{};

    for (const auto& source : sources) {
        const physicsmade::math::Vector3 delta = source.position - samplePosition;
        const double distanceSquared = delta.normSquared() + physicsmade::common::kEpsilon;
        const double distance = std::sqrt(distanceSquared);
        const double inverseDistanceCubed = 1.0 / (distanceSquared * distance);

        sample.acceleration += delta * (gravitationalConstant * source.mass * inverseDistanceCubed);
        sample.potential -= gravitationalConstant * source.mass / distance;
    }

    return sample;
}

}  // namespace

namespace physicsmade::cuda {

FieldGrid evaluateGravityField(
    const std::vector<scene::ObjectState>& sources,
    const FieldGridSpec& spec,
    double gravitationalConstant) {
    if (spec.nx <= 0 || spec.ny <= 0 || spec.nz <= 0) {
        throw std::invalid_argument("field grid dimensions must be positive");
    }

    FieldGrid grid;
    grid.spec = spec;
    grid.samples.resize(spec.sampleCount());

    for (std::size_t sampleIndex = 0; sampleIndex < grid.samples.size(); ++sampleIndex) {
        grid.samples[sampleIndex] = sampleGravityPoint(sources, spec.positionAt(sampleIndex), gravitationalConstant);
    }

    return grid;
}

bool cudaFieldSolverAvailable() noexcept {
    return false;
}

}  // namespace physicsmade::cuda

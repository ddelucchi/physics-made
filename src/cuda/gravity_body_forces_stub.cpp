#include "physicsmade/cuda/gravity_body_forces.hpp"

#include <cmath>

namespace physicsmade::cuda {

std::vector<math::Vector3> accumulateGravityForces(
    const std::vector<scene::ObjectState>& sources,
    double gravitationalConstant) {
    std::vector<math::Vector3> forces(sources.size(), {});

    for (std::size_t index = 0; index < sources.size(); ++index) {
        math::Vector3 accumulated{};
        for (std::size_t otherIndex = 0; otherIndex < sources.size(); ++otherIndex) {
            if (index == otherIndex) {
                continue;
            }

            const auto& source = sources[index];
            const auto& other = sources[otherIndex];
            const math::Vector3 delta = other.position - source.position;
            const double distanceSquared = delta.normSquared() + common::kEpsilon;
            const double distance = std::sqrt(distanceSquared);
            accumulated += delta * (gravitationalConstant * source.mass * other.mass / (distanceSquared * distance));
        }

        forces[index] = accumulated;
    }

    return forces;
}

bool cudaBodyForceSolverAvailable() noexcept {
    return false;
}

}  // namespace physicsmade::cuda

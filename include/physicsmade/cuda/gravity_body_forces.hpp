#pragma once

#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::cuda {

std::vector<math::Vector3> accumulateGravityForces(
    const std::vector<scene::ObjectState>& sources,
    double gravitationalConstant = common::kGravitationalConstant);

bool cudaBodyForceSolverAvailable() noexcept;

}  // namespace physicsmade::cuda

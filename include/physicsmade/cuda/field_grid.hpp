#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::cuda {

struct FieldSample {
    math::Vector3 acceleration{};
    double potential{0.0};
};

struct FieldGridSpec {
    int nx{1};
    int ny{1};
    int nz{1};
    math::Vector3 origin{};
    math::Vector3 spacing{1.0, 1.0, 1.0};

    std::size_t sampleCount() const noexcept {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    }

    math::Vector3 positionAt(std::size_t flatIndex) const noexcept {
        if (flatIndex >= sampleCount()) {
            return origin;
        }

        const std::size_t xyStride = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
        const std::size_t zIndex = flatIndex / xyStride;
        const std::size_t remainder = flatIndex % xyStride;
        const std::size_t yIndex = remainder / static_cast<std::size_t>(nx);
        const std::size_t xIndex = remainder % static_cast<std::size_t>(nx);

        return {
            origin.x + (spacing.x * static_cast<double>(xIndex)),
            origin.y + (spacing.y * static_cast<double>(yIndex)),
            origin.z + (spacing.z * static_cast<double>(zIndex)),
        };
    }
};

struct FieldGrid {
    FieldGridSpec spec{};
    std::vector<FieldSample> samples{};

    const FieldSample& at(int ix, int iy, int iz) const {
        const std::size_t flatIndex =
            static_cast<std::size_t>(iz * spec.nx * spec.ny) +
            static_cast<std::size_t>(iy * spec.nx) +
            static_cast<std::size_t>(ix);

        if (flatIndex >= samples.size()) {
            throw std::out_of_range("field sample index out of range");
        }

        return samples[flatIndex];
    }
};

FieldGrid evaluateGravityField(
    const std::vector<scene::ObjectState>& sources,
    const FieldGridSpec& spec,
    double gravitationalConstant = common::kGravitationalConstant);

bool cudaFieldSolverAvailable() noexcept;

}  // namespace physicsmade::cuda

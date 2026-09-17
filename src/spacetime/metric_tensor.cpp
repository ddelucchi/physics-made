#include "physicsmade/spacetime/metric_tensor.hpp"

namespace physicsmade::spacetime {

Tensor4 MinkowskiMetric::covariant(const math::FourVector& position) const {
    (void)position;
    return {{{{-1.0, 0.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0, 0.0}}, {{0.0, 0.0, 0.0, 1.0}}}};
}

std::string MinkowskiMetric::name() const {
    return "minkowski";
}

}  // namespace physicsmade::spacetime

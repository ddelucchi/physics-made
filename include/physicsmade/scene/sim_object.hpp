#pragma once

#include <string>

#include "physicsmade/scene/object_state.hpp"

namespace physicsmade::scene {

class SimObject {
  public:
    virtual ~SimObject() = default;

    virtual std::string kind() const = 0;
    virtual ObjectState initialState() const = 0;
};

}  // namespace physicsmade::scene

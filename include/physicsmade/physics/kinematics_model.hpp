#pragma once

#include <string>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"
#include "physicsmade/spacetime/spacetime_model.hpp"

namespace physicsmade::physics {

enum class KinematicRegime {
    NonRelativistic,
    SpecialRelativistic,
};

class KinematicsModel {
  public:
    virtual ~KinematicsModel() = default;

    virtual std::string name() const = 0;
    virtual KinematicRegime regime() const noexcept = 0;
    virtual double lorentzGamma(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept = 0;
    virtual double properTimeStep(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime,
        double coordinateDtSeconds) const noexcept = 0;
    virtual math::Vector3 accelerationFromForce(
        const scene::ObjectState& state,
        const math::Vector3& force,
        const spacetime::SpacetimeModel& spacetime) const noexcept = 0;
    virtual void projectVelocity(
        scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept = 0;
    virtual math::Vector3 momentum(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept = 0;
    virtual double kineticEnergy(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept = 0;
};

class NewtonianKinematics final : public KinematicsModel {
  public:
    std::string name() const override;
    KinematicRegime regime() const noexcept override;
    double lorentzGamma(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    double properTimeStep(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime,
        double coordinateDtSeconds) const noexcept override;
    math::Vector3 accelerationFromForce(
        const scene::ObjectState& state,
        const math::Vector3& force,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    void projectVelocity(
        scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    math::Vector3 momentum(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    double kineticEnergy(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
};

class SpecialRelativisticKinematics final : public KinematicsModel {
  public:
    std::string name() const override;
    KinematicRegime regime() const noexcept override;
    double lorentzGamma(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    double properTimeStep(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime,
        double coordinateDtSeconds) const noexcept override;
    math::Vector3 accelerationFromForce(
        const scene::ObjectState& state,
        const math::Vector3& force,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    void projectVelocity(
        scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    math::Vector3 momentum(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
    double kineticEnergy(
        const scene::ObjectState& state,
        const spacetime::SpacetimeModel& spacetime) const noexcept override;
};

const char* toString(KinematicRegime regime) noexcept;

}  // namespace physicsmade::physics

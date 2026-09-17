#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "physicsmade/common/config.hpp"
#include "physicsmade/math/vector3.hpp"
#include "physicsmade/scene/object_state.hpp"
#include "physicsmade/spacetime/spacetime_model.hpp"

namespace physicsmade::physics {

class Constraint {
  public:
    virtual ~Constraint() = default;

    virtual std::string name() const = 0;
    virtual void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const spacetime::SpacetimeModel& spacetime) const = 0;
};

class ConstraintSolver {
  public:
    virtual ~ConstraintSolver() = default;

    virtual std::string name() const = 0;
    virtual void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const std::vector<std::unique_ptr<Constraint>>& constraints,
        const spacetime::SpacetimeModel& spacetime) const = 0;
};

struct DistanceConstraintSpec {
    std::size_t leftIndex{0};
    std::size_t rightIndex{0};
    double targetDistance{1.0};
    double stiffness{1.0};
};

struct BallJointConstraintSpec {
    std::size_t leftIndex{0};
    std::size_t rightIndex{0};
    math::Vector3 leftLocalAnchor{};
    math::Vector3 rightLocalAnchor{};
    double stiffness{0.8};
    double damping{0.2};
};

struct HingeConstraintSpec {
    BallJointConstraintSpec anchor{};
    math::Vector3 leftLocalAxis{1.0, 0.0, 0.0};
    math::Vector3 rightLocalAxis{1.0, 0.0, 0.0};
  math::Vector3 leftLocalReference{0.0, 0.0, 1.0};
  math::Vector3 rightLocalReference{0.0, 0.0, 1.0};
    double angularStiffness{0.8};
  bool limitsEnabled{false};
  double lowerAngleLimit{-common::kPi};
  double upperAngleLimit{common::kPi};
  bool motorEnabled{false};
  double targetAngularSpeed{0.0};
  double maxMotorTorque{0.0};
};

class DistanceConstraint final : public Constraint {
  public:
    explicit DistanceConstraint(DistanceConstraintSpec spec) : spec_(spec) {}

    std::string name() const override;

    void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const spacetime::SpacetimeModel& spacetime) const override;

    const DistanceConstraintSpec& spec() const noexcept {
        return spec_;
    }

  private:
    DistanceConstraintSpec spec_{};
};

class BallJointConstraint final : public Constraint {
  public:
    explicit BallJointConstraint(BallJointConstraintSpec spec) : spec_(spec) {}

    std::string name() const override;

    void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const spacetime::SpacetimeModel& spacetime) const override;

    const BallJointConstraintSpec& spec() const noexcept {
        return spec_;
    }

  private:
    BallJointConstraintSpec spec_{};
};

class HingeConstraint final : public Constraint {
  public:
    explicit HingeConstraint(HingeConstraintSpec spec) : spec_(spec) {}

    std::string name() const override;

    void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const spacetime::SpacetimeModel& spacetime) const override;

    const HingeConstraintSpec& spec() const noexcept {
        return spec_;
    }

  private:
    HingeConstraintSpec spec_{};
};

class SequentialImpulseConstraintSolver final : public ConstraintSolver {
  public:
    explicit SequentialImpulseConstraintSolver(
        double positionalCorrection = 0.85,
        int iterations = 8,
        double frictionCoefficient = 0.6,
        double baumgarteFactor = 0.2)
        : positionalCorrection_(positionalCorrection),
          iterations_(iterations),
          frictionCoefficient_(frictionCoefficient),
          baumgarteFactor_(baumgarteFactor) {}

    std::string name() const override;

    void solve(
        std::vector<scene::ObjectState>& states,
        double dtSeconds,
        const std::vector<std::unique_ptr<Constraint>>& constraints,
        const spacetime::SpacetimeModel& spacetime) const override;

  private:
    void solveSphereContacts(std::vector<scene::ObjectState>& states, double dtSeconds) const;

    double positionalCorrection_{0.85};
    int iterations_{8};
    double frictionCoefficient_{0.6};
    double baumgarteFactor_{0.2};
};

}  // namespace physicsmade::physics

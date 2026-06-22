#pragma once

#include "go2_deploy/deploy_params.hpp"

#include <array>
#include <vector>

namespace go2_deploy {

struct RobotState {
    std::array<float, 3> gyroscope{};
    std::array<float, 4> quaternion_wxyz{};
    std::array<float, 12> joint_pos{};
    std::array<float, 12> joint_vel{};
    std::array<float, 3> command_raw{};
};

class ObservationBuilder {
public:
    explicit ObservationBuilder(const DeployParams& params);

    std::vector<float> build(const RobotState& state, const std::vector<float>& last_actions) const;

private:
    DeployParams params_;
};

std::array<float, 3> projected_gravity_from_quat_wxyz(const std::array<float, 4>& q);
std::vector<float> clip_actions(const std::vector<float>& actions, float clip_value);
std::vector<float> actions_to_joint_targets(
    const std::vector<float>& actions,
    const DeployParams& params);

}  // namespace go2_deploy

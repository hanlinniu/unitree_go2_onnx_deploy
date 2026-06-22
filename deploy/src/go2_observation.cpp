#include "go2_deploy/go2_observation.hpp"

#include <algorithm>
#include <cmath>

namespace go2_deploy {

ObservationBuilder::ObservationBuilder(const DeployParams& params) : params_(params) {}

std::array<float, 3> projected_gravity_from_quat_wxyz(const std::array<float, 4>& q) {
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];
    return {
        2.f * (-qz * qx + qw * qy),
        -2.f * (qz * qy + qw * qx),
        1.f - 2.f * (qw * qw + qz * qz),
    };
}

std::vector<float> ObservationBuilder::build(
    const RobotState& state,
    const std::vector<float>& last_actions) const {
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(params_.num_observations));

    const float ang_vel_scale = params_.obs_scales_ang_vel_dof_pos_dof_vel[0];
    for (float g : state.gyroscope) {
        obs.push_back(g * ang_vel_scale);
    }

    const auto gravity = projected_gravity_from_quat_wxyz(state.quaternion_wxyz);
    obs.insert(obs.end(), gravity.begin(), gravity.end());

    for (int i = 0; i < 3; ++i) {
        obs.push_back(state.command_raw[i] * params_.commands_scale[i]);
    }

    const float dof_pos_scale = params_.obs_scales_ang_vel_dof_pos_dof_vel[1];
    for (int i = 0; i < params_.num_actions; ++i) {
        obs.push_back((state.joint_pos[i] - params_.default_joint_pos[i]) * dof_pos_scale);
    }

    const float dof_vel_scale = params_.obs_scales_ang_vel_dof_pos_dof_vel[2];
    for (int i = 0; i < params_.num_actions; ++i) {
        obs.push_back(state.joint_vel[i] * dof_vel_scale);
    }

    obs.insert(obs.end(), last_actions.begin(), last_actions.end());

    for (float& value : obs) {
        value = std::clamp(value, -params_.clip_observations, params_.clip_observations);
    }
    return obs;
}

std::vector<float> clip_actions(const std::vector<float>& actions, float clip_value) {
    std::vector<float> clipped = actions;
    for (float& value : clipped) {
        value = std::clamp(value, -clip_value, clip_value);
    }
    return clipped;
}

std::vector<float> actions_to_joint_targets(
    const std::vector<float>& actions,
    const DeployParams& params) {
    std::vector<float> targets(params.num_actions);
    for (int i = 0; i < params.num_actions; ++i) {
        targets[i] = actions[i] * params.action_scale + params.default_joint_pos[i];
    }
    return targets;
}

}  // namespace go2_deploy

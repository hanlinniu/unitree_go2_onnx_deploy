#pragma once

#include <array>
#include <string>
#include <vector>

namespace go2_deploy {

struct DeployParams {
    int num_observations = 45;
    int num_actions = 12;
    float step_dt = 0.02f;
    float action_scale = 0.25f;
    float clip_actions = 1.2f;
    float clip_observations = 100.f;
    std::array<float, 3> obs_scales_ang_vel_dof_pos_dof_vel{0.25f, 1.f, 0.05f};
    std::array<float, 3> commands_scale{2.f, 2.f, 0.25f};
    std::array<float, 3> max_cmd{1.f, 1.f, 1.f};
    std::vector<float> default_joint_pos;
    std::vector<float> policy_stiffness;
    std::vector<float> policy_damping;
    std::vector<int> joint_ids_map;
};

DeployParams load_deploy_params(const std::string& deploy_yaml_path);

}  // namespace go2_deploy

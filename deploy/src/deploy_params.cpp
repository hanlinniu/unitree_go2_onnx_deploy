#include "go2_deploy/deploy_params.hpp"

#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace go2_deploy {

static std::vector<float> read_float_vector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("Expected YAML float sequence");
    }
    std::vector<float> out;
    out.reserve(node.size());
    for (const auto& item : node) {
        out.push_back(item.as<float>());
    }
    return out;
}

static std::vector<int> read_int_vector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("Expected YAML int sequence");
    }
    std::vector<int> out;
    out.reserve(node.size());
    for (const auto& item : node) {
        out.push_back(item.as<int>());
    }
    return out;
}

DeployParams load_deploy_params(const std::string& deploy_yaml_path) {
    YAML::Node root = YAML::LoadFile(deploy_yaml_path);
    DeployParams params;
    params.num_observations = root["num_observations"].as<int>();
    params.num_actions = root["num_actions"].as<int>();
    params.step_dt = root["step_dt"].as<float>();
    params.action_scale = root["action_scale"].as<float>();
    params.clip_actions = root["clip_actions"].as<float>();
    params.clip_observations = root["clip_observations"].as<float>();

    const auto obs_scales = root["obs_scales"];
    params.obs_scales_ang_vel_dof_pos_dof_vel = {
        obs_scales["ang_vel"].as<float>(),
        obs_scales["dof_pos"].as<float>(),
        obs_scales["dof_vel"].as<float>(),
    };

    const auto cmd_scale = read_float_vector(root["commands_scale"]);
    if (cmd_scale.size() != 3) {
        throw std::runtime_error("commands_scale must have length 3");
    }
    params.commands_scale = {cmd_scale[0], cmd_scale[1], cmd_scale[2]};

    if (root["commands"] && root["commands"]["max_cmd"]) {
        const auto max_cmd = read_float_vector(root["commands"]["max_cmd"]);
        if (max_cmd.size() == 3) {
            params.max_cmd = {max_cmd[0], max_cmd[1], max_cmd[2]};
        }
    }

    params.default_joint_pos = read_float_vector(root["default_joint_pos"]);
    params.joint_ids_map = read_int_vector(root["joint_ids_map"]);

    if (root["policy_stiffness"]) {
        params.policy_stiffness = read_float_vector(root["policy_stiffness"]);
    } else {
        params.policy_stiffness = read_float_vector(root["stiffness"]);
    }
    if (root["policy_damping"]) {
        params.policy_damping = read_float_vector(root["policy_damping"]);
    } else {
        params.policy_damping = read_float_vector(root["damping"]);
    }

    if (static_cast<int>(params.default_joint_pos.size()) != params.num_actions) {
        throw std::runtime_error("default_joint_pos size mismatch");
    }
    if (static_cast<int>(params.joint_ids_map.size()) != params.num_actions) {
        throw std::runtime_error("joint_ids_map size mismatch");
    }
    if (static_cast<int>(params.policy_stiffness.size()) != params.num_actions) {
        throw std::runtime_error("policy_stiffness size mismatch");
    }
    if (static_cast<int>(params.policy_damping.size()) != params.num_actions) {
        throw std::runtime_error("policy_damping size mismatch");
    }
    return params;
}

}  // namespace go2_deploy

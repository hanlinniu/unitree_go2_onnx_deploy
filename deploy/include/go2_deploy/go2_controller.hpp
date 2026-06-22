#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "go2_deploy/command_config.hpp"
#include "go2_deploy/deploy_params.hpp"
#include "go2_deploy/fault_history.hpp"
#include "go2_deploy/fault_scenario.hpp"
#include "go2_deploy/go2_lowlevel.hpp"
#include "go2_deploy/go2_observation.hpp"
#include "go2_deploy/onnx_fault_predictor.hpp"
#include "go2_deploy/onnx_policy.hpp"

namespace go2_deploy {

enum class ControlMode {
    Sport,
    Passive,
    FixStand,
    Velocity,
};

class Go2OnnxController {
public:
    Go2OnnxController(
        Go2LowLevel& lowlevel,
        const DeployParams& deploy_params,
        const std::string& onnx_path,
        const YAML::Node& fsm_config,
        ScenarioConfig scenario_config,
        CommandConfig command_config,
        const std::optional<std::string>& fault_predictor_path = std::nullopt);

    void start();
    void stop();

    ControlMode mode() const { return mode_.load(); }
    void request_mode(ControlMode mode) { requested_mode_.store(mode); }
    bool fixstand_ready() const { return fixstand_ready_.load(); }

private:
    void control_loop();
    void apply_passive(unitree_go::msg::dds_::LowCmd_& cmd, float kd) const;
    void apply_fixstand(unitree_go::msg::dds_::LowCmd_& cmd, double elapsed_sec);
    void apply_velocity(unitree_go::msg::dds_::LowCmd_& cmd);
    RobotState read_robot_state() const;
    void reset_fault_history();
    void update_fault_diagnosis(const std::vector<float>& proprio);

    Go2LowLevel& lowlevel_;
    DeployParams deploy_params_;
    OnnxPolicy policy_;
    std::unique_ptr<OnnxFaultPredictor> fault_predictor_;
    ProprioHistoryBuffer fault_history_;
    ObservationBuilder obs_builder_;
    YAML::Node fsm_config_;
    FaultScenario scenario_;
    CommandConfig command_config_;

    std::vector<float> last_actions_;
    std::atomic<bool> running_{false};
    std::atomic<ControlMode> mode_{ControlMode::Sport};
    std::atomic<ControlMode> requested_mode_{ControlMode::Sport};
    std::atomic<bool> fixstand_ready_{false};
    std::thread control_thread_;

    double fixstand_t0_ = 0.0;
    std::vector<float> fixstand_ts_;
    std::vector<std::vector<float>> fixstand_qs_;
    std::vector<float> fixstand_kp_;
    std::vector<float> fixstand_kd_;
    double last_fault_log_sec_{-1.0};
};

}  // namespace go2_deploy

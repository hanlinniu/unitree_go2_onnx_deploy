#include "go2_deploy/go2_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace go2_deploy {

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<float> linear_interpolate(
    double t,
    const std::vector<float>& ts,
    const std::vector<std::vector<float>>& qs) {
    if (qs.empty()) {
        return {};
    }
    if (t <= ts.front()) {
        return qs.front();
    }
    if (t >= ts.back()) {
        return qs.back();
    }
    for (size_t i = 1; i < ts.size(); ++i) {
        if (t <= ts[i]) {
            const double t0 = ts[i - 1];
            const double t1 = ts[i];
            const double alpha = (t - t0) / (t1 - t0);
            std::vector<float> out(qs[i].size());
            for (size_t j = 0; j < out.size(); ++j) {
                out[j] = static_cast<float>((1.0 - alpha) * qs[i - 1][j] + alpha * qs[i][j]);
            }
            return out;
        }
    }
    return qs.back();
}

Go2OnnxController::Go2OnnxController(
    Go2LowLevel& lowlevel,
    const DeployParams& deploy_params,
    const std::string& onnx_path,
    const YAML::Node& fsm_config,
    ScenarioConfig scenario_config,
    CommandConfig command_config,
    const std::optional<std::string>& fault_predictor_path)
    : lowlevel_(lowlevel),
      deploy_params_(deploy_params),
      policy_(onnx_path),
      fault_history_(deploy_params.fault_history_len, deploy_params.num_observations),
      obs_builder_(deploy_params),
      fsm_config_(fsm_config),
      scenario_(std::move(scenario_config)),
      command_config_(std::move(command_config)),
      last_actions_(deploy_params.num_actions, 0.f) {
    if (fault_predictor_path.has_value()) {
        fault_predictor_ = std::make_unique<OnnxFaultPredictor>(fault_predictor_path.value());
        const int expected_dim = deploy_params.fault_history_len * deploy_params.num_observations;
        if (fault_predictor_->fault_observation_dim() != expected_dim) {
            throw std::runtime_error("fault_predictor ONNX input dim mismatch with deploy.yaml");
        }
        std::cout << "Loaded fault predictor ONNX: " << fault_predictor_path.value() << std::endl;
    }
    const auto fixstand = fsm_config["FixStand"];
    fixstand_ts_ = fixstand["ts"].as<std::vector<float>>();
    fixstand_qs_ = fixstand["qs"].as<std::vector<std::vector<float>>>();
    fixstand_kp_ = fixstand["kp"].as<std::vector<float>>();
    fixstand_kd_ = fixstand["kd"].as<std::vector<float>>();
    if (policy_.num_actions() != deploy_params.num_actions) {
        throw std::runtime_error("ONNX action dimension mismatch");
    }
}

void Go2OnnxController::start() {
    if (running_.exchange(true)) {
        return;
    }
    control_thread_ = std::thread([this] { control_loop(); });
}

void Go2OnnxController::stop() {
    running_.store(false);
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
}

RobotState Go2OnnxController::read_robot_state() const {
    const auto low_state = lowlevel_.get_low_state();
    const auto wireless = lowlevel_.get_wireless();
    RobotState state{};

    state.gyroscope = {
        low_state.imu_state().gyroscope()[0],
        low_state.imu_state().gyroscope()[1],
        low_state.imu_state().gyroscope()[2],
    };
    state.quaternion_wxyz = {
        low_state.imu_state().quaternion()[0],
        low_state.imu_state().quaternion()[1],
        low_state.imu_state().quaternion()[2],
        low_state.imu_state().quaternion()[3],
    };

    for (int sim_idx = 0; sim_idx < deploy_params_.num_actions; ++sim_idx) {
        const int motor_idx = deploy_params_.joint_ids_map[sim_idx];
        state.joint_pos[sim_idx] = low_state.motor_state()[motor_idx].q();
        state.joint_vel[sim_idx] = low_state.motor_state()[motor_idx].dq();
    }

    const float ly = wireless.ly();
    const float lx = -wireless.lx();
    const float rx = -wireless.rx();

    if (command_config_.source == CommandSource::Fixed) {
        state.command_raw = {command_config_.vx, command_config_.vy, command_config_.vyaw};
        if (command_config_.fixed_command_joystick_yaw) {
            state.command_raw[2] = rx * deploy_params_.max_cmd[2];
        }
    } else {
        state.command_raw = {
            ly * deploy_params_.max_cmd[0],
            lx * deploy_params_.max_cmd[1],
            rx * deploy_params_.max_cmd[2],
        };
    }
    return state;
}

void Go2OnnxController::reset_fault_history() {
    fault_history_.reset();
    last_fault_log_sec_ = -1.0;
    last_scenario_log_sec_ = -1.0;
}

void Go2OnnxController::update_fault_diagnosis(const std::vector<float>& proprio) {
    if (!fault_predictor_) {
        return;
    }

    fault_history_.push(proprio);
    const auto diagnosis = fault_predictor_->infer(fault_history_.flatten());
    const double now = now_sec();
    if (last_fault_log_sec_ < 0.0 || now - last_fault_log_sec_ >= 1.0) {
        print_fault_diagnosis(diagnosis, deploy_params_.joint_names);
        last_fault_log_sec_ = now;
    }
}

void Go2OnnxController::apply_passive(unitree_go::msg::dds_::LowCmd_& cmd, float kd) const {
    for (int i = 0; i < 20; ++i) {
        auto& motor = cmd.motor_cmd()[i];
        motor.mode() = 0x01;
        motor.kp() = 0.f;
        motor.kd() = kd;
        motor.dq() = 0.f;
        motor.tau() = 0.f;
    }
}

void Go2OnnxController::apply_fixstand(unitree_go::msg::dds_::LowCmd_& cmd, double elapsed_sec) {
    const auto q = linear_interpolate(static_cast<float>(elapsed_sec), fixstand_ts_, fixstand_qs_);
    for (int motor_idx = 0; motor_idx < deploy_params_.num_actions; ++motor_idx) {
        auto& motor = cmd.motor_cmd()[motor_idx];
        motor.mode() = 0x01;
        motor.kp() = fixstand_kp_[motor_idx];
        motor.kd() = fixstand_kd_[motor_idx];
        motor.q() = q[motor_idx];
        motor.dq() = 0.f;
        motor.tau() = 0.f;
    }
}

void Go2OnnxController::apply_velocity(unitree_go::msg::dds_::LowCmd_& cmd) {
    const double loop_t0 = now_sec();
    scenario_.activate_if_due(loop_t0);

    const RobotState state = read_robot_state();
    const auto obs = obs_builder_.build(state, last_actions_);
    update_fault_diagnosis(obs);
    auto actions = policy_.infer(obs);
    actions = clip_actions(actions, deploy_params_.clip_actions);
    last_actions_ = actions;
    auto targets = actions_to_joint_targets(actions, deploy_params_);
    scenario_.apply_lock_target(targets);

    for (int sim_idx = 0; sim_idx < deploy_params_.num_actions; ++sim_idx) {
        const int motor_idx = deploy_params_.joint_ids_map[sim_idx];
        auto& motor = cmd.motor_cmd()[motor_idx];
        motor.mode() = 0x01;
        float kp = 0.f;
        float kd = 0.f;
        scenario_.scale_motor_gains(
            sim_idx,
            deploy_params_.policy_stiffness[sim_idx],
            deploy_params_.policy_damping[sim_idx],
            kp,
            kd);
        motor.kp() = kp;
        motor.kd() = kd;
        motor.q() = targets[sim_idx];
        motor.dq() = 0.f;
        motor.tau() = 0.f;
    }

    const double loop_elapsed = now_sec() - loop_t0;
    if (last_scenario_log_sec_ < 0.0 || loop_t0 - last_scenario_log_sec_ >= 1.0) {
        const ScenarioStatus scenario_status = scenario_.status(deploy_params_.joint_names);
        std::cout << "[run_fault_scenarios] scenario=" << scenario_type_name(scenario_.config().type)
                  << "  phase=" << scenario_status.phase << "  loop=" << std::fixed << std::setprecision(5)
                  << loop_elapsed << "s" << scenario_status.info << std::endl;
        last_scenario_log_sec_ = loop_t0;
    }
}

void Go2OnnxController::control_loop() {
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration<double>(deploy_params_.step_dt);
    auto next_tick = clock::now() + dt;

    while (running_.load()) {
        const ControlMode requested = requested_mode_.load();
        if (requested != mode_.load()) {
            mode_.store(requested);
            if (mode_.load() == ControlMode::FixStand) {
                fixstand_ready_.store(false);
                scenario_.clear();
                reset_fault_history();
                fixstand_t0_ = now_sec();
                const auto low_state = lowlevel_.get_low_state();
                fixstand_qs_.front().clear();
                fixstand_qs_.front().reserve(deploy_params_.num_actions);
                for (int motor_idx = 0; motor_idx < deploy_params_.num_actions; ++motor_idx) {
                    fixstand_qs_.front().push_back(low_state.motor_state()[motor_idx].q());
                }
            }
            if (mode_.load() == ControlMode::Velocity) {
                std::fill(last_actions_.begin(), last_actions_.end(), 0.f);
                reset_fault_history();
                for (int i = 0; i < 3; ++i) {
                    policy_.infer(std::vector<float>(deploy_params_.num_observations, 0.f));
                }
                scenario_.arm(now_sec());
            }
            if (mode_.load() == ControlMode::Sport) {
                fixstand_ready_.store(false);
                scenario_.clear();
                reset_fault_history();
            }
        }

        if (mode_.load() == ControlMode::Sport) {
            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
            continue;
        }

        unitree_go::msg::dds_::LowCmd_ cmd{};
        for (int i = 0; i < 20; ++i) {
            cmd.motor_cmd()[i].mode() = 0x01;
        }

        switch (mode_.load()) {
            case ControlMode::Passive:
                apply_passive(cmd, fsm_config_["Passive"]["kd"].as<std::vector<float>>()[0]);
                break;
            case ControlMode::FixStand: {
                const double elapsed = now_sec() - fixstand_t0_;
                apply_fixstand(cmd, elapsed);
                if (!fixstand_ts_.empty() && elapsed >= fixstand_ts_.back()) {
                    fixstand_ready_.store(true);
                }
                break;
            }
            case ControlMode::Velocity:
                apply_velocity(cmd);
                break;
        }

        lowlevel_.publish_lowcmd(cmd);
        std::this_thread::sleep_until(next_tick);
        next_tick += dt;
    }
}

}  // namespace go2_deploy

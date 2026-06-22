#include "go2_deploy/fault_scenario.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <iomanip>
#include <sstream>

namespace go2_deploy {

namespace {

constexpr std::array<int, 4> kCalfSimIndices = {2, 5, 8, 11};
constexpr std::array<const char*, 4> kLegNames = {"FL", "FR", "RL", "RR"};

int leg_name_to_calf_sim_idx(const std::string& leg) {
    for (size_t i = 0; i < kLegNames.size(); ++i) {
        if (leg == kLegNames[i]) {
            return kCalfSimIndices[i];
        }
    }
    return -1;
}

}  // namespace

FaultScenario::FaultScenario(ScenarioConfig config)
    : config_(std::move(config)),
      rng_(std::random_device{}()),
      lock_angle_rad_(static_cast<float>(config_.lock_angle_deg * M_PI / 180.0)) {
    if (config_.fault_calf != "RANDOM") {
        const int sim_idx = leg_name_to_calf_sim_idx(config_.fault_calf);
        if (sim_idx < 0) {
            throw std::runtime_error("Invalid fault calf leg: " + config_.fault_calf);
        }
        fixed_calf_sim_idx_ = sim_idx;
    }
}

void FaultScenario::arm(double now_sec) {
    armed_ = true;
    lock_active_ = false;
    fault_active_ = false;
    locked_sim_idx_.reset();
    failed_sim_idx_.reset();
    countdown_start_sec_ = now_sec;
    last_countdown_log_sec_ = -1.0;

    if (config_.type == ScenarioType::Healthy) {
        std::cout << "Scenario armed: healthy policy only" << std::endl;
        return;
    }

    const std::string target =
        fixed_calf_sim_idx_.has_value() ? calf_name_for_sim_idx(*fixed_calf_sim_idx_) + " calf" : "a random calf motor";
    if (config_.type == ScenarioType::RandomLock) {
        std::cout << "Scenario armed: healthy policy for " << config_.healthy_s << "s before locking " << target
                  << " at " << config_.lock_angle_deg << " deg" << std::endl;
    } else {
        std::cout << "Scenario armed: healthy policy for " << config_.healthy_s << "s before weakening " << target
                  << std::endl;
    }
}

void FaultScenario::clear() {
    armed_ = false;
    lock_active_ = false;
    fault_active_ = false;
    locked_sim_idx_.reset();
    failed_sim_idx_.reset();
    countdown_start_sec_ = 0.0;
    last_countdown_log_sec_ = -1.0;
}

void FaultScenario::activate_if_due(double now_sec) {
    if (!armed_ || config_.type == ScenarioType::Healthy || lock_active_ || fault_active_) {
        return;
    }

    const double elapsed = now_sec - countdown_start_sec_;
    if (elapsed < config_.healthy_s) {
        if (last_countdown_log_sec_ < 0.0 || now_sec - last_countdown_log_sec_ >= 1.0) {
            const double remaining = config_.healthy_s - elapsed;
            std::cout << "healthy policy phase: " << remaining << "s until "
                      << (config_.type == ScenarioType::RandomLock ? "calf lock" : "calf failure") << std::endl;
            last_countdown_log_sec_ = now_sec;
        }
        return;
    }

    const int selected_sim_idx = pick_calf_sim_idx();
    if (config_.type == ScenarioType::RandomLock) {
        locked_sim_idx_ = selected_sim_idx;
        lock_active_ = true;
        std::cout << "Calf lock active: " << calf_name_for_sim_idx(selected_sim_idx)
                  << "_calf_joint (sim_idx=" << selected_sim_idx << ") target=" << lock_angle_rad_ << " rad"
                  << std::endl;
    } else {
        failed_sim_idx_ = selected_sim_idx;
        failed_alpha_ = static_cast<float>(config_.fault_sthres);
        fault_active_ = true;
        std::cout << "Calf motor failure active: " << calf_name_for_sim_idx(selected_sim_idx)
                  << "_calf_joint (sim_idx=" << selected_sim_idx << ") PD gain scale alpha=" << failed_alpha_
                  << std::endl;
    }
}

void FaultScenario::apply_lock_target(std::vector<float>& targets) const {
    if (lock_active_ && locked_sim_idx_.has_value()) {
        targets[*locked_sim_idx_] = lock_angle_rad_;
    }
}

void FaultScenario::scale_motor_gains(int sim_idx, float base_kp, float base_kd, float& kp, float& kd) const {
    kp = base_kp;
    kd = base_kd;
    if (fault_active_ && failed_sim_idx_.has_value() && sim_idx == *failed_sim_idx_) {
        kp *= failed_alpha_;
        kd *= failed_alpha_;
    }
}

int FaultScenario::pick_calf_sim_idx() {
    if (fixed_calf_sim_idx_.has_value()) {
        return *fixed_calf_sim_idx_;
    }
    std::uniform_int_distribution<int> dist(0, static_cast<int>(kCalfSimIndices.size()) - 1);
    return kCalfSimIndices[static_cast<size_t>(dist(rng_))];
}

std::string FaultScenario::calf_name_for_sim_idx(int sim_idx) const {
    for (size_t i = 0; i < kCalfSimIndices.size(); ++i) {
        if (kCalfSimIndices[i] == sim_idx) {
            return kLegNames[i];
        }
    }
    return "UNKNOWN";
}

ScenarioStatus FaultScenario::status(const std::vector<std::string>& joint_names) const {
    ScenarioStatus out;
    if (config_.type == ScenarioType::Healthy) {
        out.phase = "HEALTHY";
        return out;
    }

    if (config_.type == ScenarioType::RandomLock) {
        out.phase = lock_active_ ? "LOCKED" : "HEALTHY";
        if (lock_active_ && locked_sim_idx_.has_value()) {
            const int sim_idx = *locked_sim_idx_;
            std::ostringstream info;
            info << std::fixed << std::setprecision(1)
                 << "  locked_dof=" << joint_names[static_cast<size_t>(sim_idx)]
                 << "  target=" << config_.lock_angle_deg << " deg";
            out.info = info.str();
        }
        return out;
    }

    out.phase = fault_active_ ? "FAULT" : "HEALTHY";
    if (fault_active_ && failed_sim_idx_.has_value()) {
        const int sim_idx = *failed_sim_idx_;
        std::ostringstream info;
        info << std::fixed << std::setprecision(3)
             << "  failed_dof=" << joint_names[static_cast<size_t>(sim_idx)]
             << "  alpha=" << failed_alpha_;
        out.info = info.str();
    }
    return out;
}

}  // namespace go2_deploy

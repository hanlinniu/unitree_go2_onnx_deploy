#pragma once

#include <optional>
#include <random>
#include <string>
#include <vector>

#include "go2_deploy/scenario_config.hpp"

namespace go2_deploy {

class FaultScenario {
public:
    explicit FaultScenario(ScenarioConfig config);

    void arm(double now_sec);
    void clear();
    void activate_if_due(double now_sec);

    void apply_lock_target(std::vector<float>& targets) const;
    void scale_motor_gains(int sim_idx, float base_kp, float base_kd, float& kp, float& kd) const;

    bool lock_active() const { return lock_active_; }
    bool fault_active() const { return fault_active_; }

private:
    int pick_calf_sim_idx();
    std::string calf_name_for_sim_idx(int sim_idx) const;

    ScenarioConfig config_;
    std::mt19937 rng_;

    bool armed_{false};
    bool lock_active_{false};
    bool fault_active_{false};
    double countdown_start_sec_{0.0};
    double last_countdown_log_sec_{-1.0};

    std::optional<int> fixed_calf_sim_idx_;
    std::optional<int> locked_sim_idx_;
    std::optional<int> failed_sim_idx_;
    float lock_angle_rad_{0.f};
    float failed_alpha_{0.f};
};

}  // namespace go2_deploy

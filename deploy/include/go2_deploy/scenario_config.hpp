#pragma once

#include <string>

namespace go2_deploy {

enum class ScenarioType {
    Healthy,
    RandomLock,
    RandomFault,
};

struct ScenarioConfig {
    ScenarioType type{ScenarioType::Healthy};
    double healthy_s{3.0};
    double fault_sthres{0.1};
    double lock_angle_deg{-120.0};
    std::string fault_calf{"RANDOM"};
};

ScenarioType normalize_scenario_type(const std::string& scenario);
ScenarioConfig parse_scenario_config(
    const std::string& scenario,
    double healthy_s,
    double fault_sthres,
    double lock_angle_deg,
    const std::string& fault_calf);
std::string scenario_type_name(ScenarioType type);
void print_scenario_config(const ScenarioConfig& config);

}  // namespace go2_deploy

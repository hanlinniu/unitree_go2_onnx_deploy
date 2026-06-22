#include "go2_deploy/scenario_config.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>

namespace go2_deploy {

namespace {

std::string to_upper_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

}  // namespace

ScenarioType normalize_scenario_type(const std::string& scenario) {
    const std::string normalized = to_lower_ascii(scenario);
    if (normalized == "healthy") {
        return ScenarioType::Healthy;
    }
    if (normalized == "random_lock" || normalized == "lock" || normalized == "locked" ||
        normalized == "random_locked") {
        return ScenarioType::RandomLock;
    }
    if (normalized == "random_fault" || normalized == "fault" || normalized == "zero_torque" ||
        normalized == "severe_fault") {
        return ScenarioType::RandomFault;
    }
    throw std::runtime_error(
        "Unknown scenario '" + scenario + "'. Expected healthy, random_lock, or random_fault.");
}

ScenarioConfig parse_scenario_config(
    const std::string& scenario,
    double healthy_s,
    double fault_sthres,
    double lock_angle_deg,
    const std::string& fault_calf) {
    ScenarioConfig config;
    config.type = normalize_scenario_type(scenario);
    config.healthy_s = healthy_s;
    config.fault_sthres = fault_sthres;
    config.lock_angle_deg = lock_angle_deg;
    config.fault_calf = to_upper_ascii(fault_calf);
    if (config.fault_calf != "RANDOM" && config.fault_calf != "FL" && config.fault_calf != "FR" &&
        config.fault_calf != "RL" && config.fault_calf != "RR") {
        throw std::runtime_error(
            "Invalid --fault_calf '" + fault_calf + "'. Expected RANDOM, FL, FR, RL, or RR.");
    }
    return config;
}

std::string scenario_type_name(ScenarioType type) {
    switch (type) {
        case ScenarioType::Healthy:
            return "healthy";
        case ScenarioType::RandomLock:
            return "random_lock";
        case ScenarioType::RandomFault:
            return "random_fault";
    }
    return "unknown";
}

void print_scenario_config(const ScenarioConfig& config) {
    std::cout << "Scenario: " << scenario_type_name(config.type) << " healthy_s=" << config.healthy_s
              << " fault_calf=" << config.fault_calf << " lock_angle=" << config.lock_angle_deg
              << "deg alpha=" << config.fault_sthres << std::endl;
}

}  // namespace go2_deploy

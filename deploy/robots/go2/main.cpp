#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "go2_deploy/go2_controller.hpp"
#include "go2_deploy/go2_lowlevel.hpp"
#include "go2_deploy/go2_sport.hpp"
#include "go2_deploy/deploy_params.hpp"
#include "go2_deploy/scenario_config.hpp"

namespace {

// Unitree GO2 wireless controller key bits (matches Kaixin unitree_ros2_real.py).
constexpr uint32_t kButtonR1 = 1u << 0;
constexpr uint32_t kButtonR2 = 1u << 4;
constexpr uint32_t kButtonY = 1u << 11;

std::filesystem::path resolve_policy_dir(const std::filesystem::path& config_dir, const std::string& policy_dir_raw) {
    std::filesystem::path policy_dir(policy_dir_raw);
    if (policy_dir.is_absolute()) {
        return policy_dir;
    }
    return (config_dir / policy_dir).lexically_normal();
}

std::filesystem::path resolve_config_file(
    const std::filesystem::path& exe_dir,
    const std::filesystem::path& config_path) {
    if (config_path.is_absolute()) {
        return config_path;
    }

    const std::filesystem::path candidates[] = {
        exe_dir / config_path,
        exe_dir / ".." / config_path,
        exe_dir / "config" / "config.yaml",
        exe_dir / ".." / "config" / "config.yaml",
    };
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return exe_dir / config_path;
}

bool button_pressed(uint32_t keys, uint32_t last_keys, uint32_t button) {
    return (keys & button) && !(last_keys & button);
}

}  // namespace

int main(int argc, char** argv) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("help,h", "Show help")
        ("network,n", boost::program_options::value<std::string>()->default_value(""), "DDS network interface (e.g. eth0)")
        ("config,c", boost::program_options::value<std::string>()->default_value("../config/config.yaml"), "FSM config path")
        ("scenario", boost::program_options::value<std::string>()->default_value("healthy"),
            "Scenario: healthy, random_fault, random_lock (aliases: fault, lock)")
        ("healthy_s", boost::program_options::value<double>()->default_value(3.0),
            "Seconds of healthy policy before random_fault/random_lock activates")
        ("fault_sthres", boost::program_options::value<double>()->default_value(0.1),
            "PD gain scale alpha for random_fault")
        ("lock_angle_deg", boost::program_options::value<double>()->default_value(-120.0),
            "Target calf angle for random_lock")
        ("fault_calf", boost::program_options::value<std::string>()->default_value("RANDOM"),
            "Calf leg to affect: RANDOM, FL, FR, RL, or RR");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    const auto exe_path = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    const auto config_path = std::filesystem::path(vm["config"].as<std::string>());
    const auto config_file = resolve_config_file(exe_path, config_path);
    if (!std::filesystem::exists(config_file)) {
        std::cerr << "Config not found: " << config_file << "\n"
                  << "Expected deploy/robots/go2/config/config.yaml. "
                  << "Try: -c ../config/config.yaml\n";
        return 1;
    }
    const auto config_dir = config_file.parent_path();

    YAML::Node config = YAML::LoadFile(config_file.string());
    const auto policy_dir = resolve_policy_dir(config_dir, config["FSM"]["Velocity"]["policy_dir"].as<std::string>());
    const auto deploy_yaml = policy_dir / "params" / "deploy.yaml";
    const auto onnx_model = policy_dir / "exported" / "policy.onnx";

    if (!std::filesystem::exists(deploy_yaml)) {
        std::cerr << "Missing deploy.yaml: " << deploy_yaml << "\n";
        return 1;
    }
    if (!std::filesystem::exists(onnx_model)) {
        std::cerr << "Missing policy.onnx: " << onnx_model << "\n";
        return 1;
    }

    go2_deploy::DeployParams deploy_params = go2_deploy::load_deploy_params(deploy_yaml.string());

    go2_deploy::ScenarioConfig scenario_config;
    try {
        scenario_config = go2_deploy::parse_scenario_config(
            vm["scenario"].as<std::string>(),
            vm["healthy_s"].as<double>(),
            vm["fault_sthres"].as<double>(),
            vm["lock_angle_deg"].as<double>(),
            vm["fault_calf"].as<std::string>());
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    go2_deploy::print_scenario_config(scenario_config);

    go2_deploy::Go2LowLevel lowlevel;
    lowlevel.init(vm["network"].as<std::string>());
    std::cout << "Waiting for robot connection..." << std::endl;
    if (!lowlevel.wait_for_connection(10.0)) {
        std::cerr << "Timed out waiting for lowstate." << std::endl;
        return 1;
    }
    std::cout << "Connected." << std::endl;

    go2_deploy::Go2Sport sport;
    sport.init();

    go2_deploy::Go2OnnxController controller(
        lowlevel,
        deploy_params,
        onnx_model.string(),
        config["FSM"],
        scenario_config);
    controller.start();

    std::cout << "Controls:\n"
              << "  R1 -> disable sport mode, FixStand\n"
              << "  Y  -> ONNX policy (after FixStand completes)\n"
              << "  R2 -> stop custom control, return to sport mode\n"
              << "Starting in sport mode (Unitree controller active).\n";

    uint32_t last_keys = 0;
    while (true) {
        const auto wireless = lowlevel.get_wireless();
        const uint32_t keys = wireless.keys();

        if (button_pressed(keys, last_keys, kButtonR1)) {
            if (sport.disable_sport_mode()) {
                controller.request_mode(go2_deploy::ControlMode::FixStand);
                std::cout << "Mode: FixStand" << std::endl;
            }
        }

        if (button_pressed(keys, last_keys, kButtonY)) {
            if (controller.fixstand_ready()) {
                if (controller.mode() != go2_deploy::ControlMode::Velocity) {
                    controller.request_mode(go2_deploy::ControlMode::Velocity);
                    std::cout << "Mode: Velocity (ONNX)" << std::endl;
                }
            } else {
                std::cout << "Y pressed, but FixStand is not ready yet" << std::endl;
            }
        }

        if (button_pressed(keys, last_keys, kButtonR2)) {
            controller.request_mode(go2_deploy::ControlMode::Sport);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (sport.enable_sport_mode()) {
                sport.balance_stand();
            }
            std::cout << "Mode: Sport" << std::endl;
        }

        last_keys = keys;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    controller.stop();
    return 0;
}

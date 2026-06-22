#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

#include "go2_deploy/go2_controller.hpp"
#include "go2_deploy/go2_lowlevel.hpp"
#include "go2_deploy/deploy_params.hpp"

namespace {

constexpr uint32_t kButtonA = 1u << 8;
constexpr uint32_t kButtonB = 1u << 9;
constexpr uint32_t kButtonStart = 1u << 2;
constexpr uint32_t kButtonL2 = 1u << 5;

std::filesystem::path resolve_policy_dir(const std::filesystem::path& config_dir, const std::string& policy_dir_raw) {
    std::filesystem::path policy_dir(policy_dir_raw);
    if (policy_dir.is_absolute()) {
        return policy_dir;
    }
    return (config_dir / policy_dir).lexically_normal();
}

}  // namespace

int main(int argc, char** argv) {
    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("help,h", "Show help")
        ("network,n", boost::program_options::value<std::string>()->default_value(""), "DDS network interface (e.g. eth0)")
        ("config,c", boost::program_options::value<std::string>()->default_value("config/config.yaml"), "FSM config path");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    const auto exe_path = std::filesystem::read_symlink("/proc/self/exe").parent_path();
    const auto config_path = std::filesystem::path(vm["config"].as<std::string>());
    const auto config_file = config_path.is_absolute() ? config_path : exe_path / config_path;
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

    go2_deploy::Go2LowLevel lowlevel;
    lowlevel.init(vm["network"].as<std::string>());
    std::cout << "Waiting for robot connection..." << std::endl;
    if (!lowlevel.wait_for_connection(10.0)) {
        std::cerr << "Timed out waiting for lowstate." << std::endl;
        return 1;
    }
    std::cout << "Connected." << std::endl;

    go2_deploy::Go2OnnxController controller(
        lowlevel,
        deploy_params,
        onnx_model.string(),
        config["FSM"]);
    controller.start();

    std::cout << "Controls:\n"
              << "  L2 + A     -> FixStand\n"
              << "  Start      -> Velocity (ONNX policy)\n"
              << "  L2 + B     -> Passive (damping)\n";

    uint32_t last_keys = 0;
    while (true) {
        const auto wireless = lowlevel.get_wireless();
        const uint32_t keys = wireless.keys();
        const uint32_t pressed = keys & ~last_keys;

        if ((keys & kButtonL2) && (keys & kButtonA) &&
            !((last_keys & kButtonL2) && (last_keys & kButtonA))) {
            controller.request_mode(go2_deploy::ControlMode::FixStand);
            std::cout << "Mode: FixStand" << std::endl;
        }
        if ((keys & kButtonL2) && (keys & kButtonB) &&
            !((last_keys & kButtonL2) && (last_keys & kButtonB))) {
            controller.request_mode(go2_deploy::ControlMode::Passive);
            std::cout << "Mode: Passive" << std::endl;
        }
        if ((pressed & kButtonStart) || ((keys & kButtonStart) && !(last_keys & kButtonStart))) {
            controller.request_mode(go2_deploy::ControlMode::Velocity);
            std::cout << "Mode: Velocity (ONNX)" << std::endl;
        }

        last_keys = keys;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    controller.stop();
    return 0;
}

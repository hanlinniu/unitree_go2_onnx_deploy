#include "go2_deploy/command_config.hpp"

namespace go2_deploy {

CommandSource parse_command_source(const std::string& value) {
    if (value == "joystick") {
        return CommandSource::Joystick;
    }
    if (value == "fixed") {
        return CommandSource::Fixed;
    }
    throw std::runtime_error(
        "Invalid --command_source '" + value + "'. Expected joystick or fixed.");
}

CommandConfig parse_command_config(
    const std::string& command_source,
    float vx,
    float vy,
    float vyaw,
    bool no_fixed_command_joystick_yaw) {
    CommandConfig config;
    config.source = parse_command_source(command_source);
    config.vx = vx;
    config.vy = vy;
    config.vyaw = vyaw;
    config.fixed_command_joystick_yaw = !no_fixed_command_joystick_yaw;
    return config;
}

void print_command_config(const CommandConfig& config) {
    if (config.source == CommandSource::Joystick) {
        std::cout << "Command source: joystick" << std::endl;
        return;
    }

    std::cout << "Command source: fixed vx=" << config.vx << " vy=" << config.vy;
    if (config.fixed_command_joystick_yaw) {
        std::cout << " vyaw=joystick";
    } else {
        std::cout << " vyaw=" << config.vyaw;
    }
    std::cout << std::endl;
}

}  // namespace go2_deploy

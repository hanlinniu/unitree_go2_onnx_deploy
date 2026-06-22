#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

namespace go2_deploy {

enum class CommandSource {
    Joystick,
    Fixed,
};

struct CommandConfig {
    CommandSource source{CommandSource::Joystick};
    float vx{0.5f};
    float vy{0.f};
    float vyaw{0.f};
    bool fixed_command_joystick_yaw{true};
};

CommandSource parse_command_source(const std::string& value);
CommandConfig parse_command_config(
    const std::string& command_source,
    float vx,
    float vy,
    float vyaw,
    bool no_fixed_command_joystick_yaw);
void print_command_config(const CommandConfig& config);

}  // namespace go2_deploy

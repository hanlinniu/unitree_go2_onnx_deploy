#pragma once

#include <iostream>
#include <string>

#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

namespace go2_deploy {

class Go2Sport {
public:
    void init();

    bool disable_sport_mode();
    bool enable_sport_mode();
    bool balance_stand();

private:
    unitree::robot::go2::RobotStateClient robot_state_;
    unitree::robot::go2::SportClient sport_;
    bool initialized_{false};
};

}  // namespace go2_deploy

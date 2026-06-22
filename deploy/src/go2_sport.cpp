#include "go2_deploy/go2_sport.hpp"

namespace go2_deploy {

void Go2Sport::init() {
    if (initialized_) {
        return;
    }
    robot_state_.SetTimeout(10.0f);
    robot_state_.Init();
    sport_.SetTimeout(10.0f);
    sport_.Init();
    initialized_ = true;
}

bool Go2Sport::disable_sport_mode() {
    if (!initialized_) {
        return false;
    }
    int32_t status = 0;
    const int32_t ret = robot_state_.ServiceSwitch("sport_mode", 0, status);
    if (ret != 0) {
        std::cerr << "ServiceSwitch(sport_mode, 0) failed: " << ret << " status=" << status << std::endl;
        return false;
    }
    std::cout << "Disabled sport_mode (low-level control enabled)" << std::endl;
    return true;
}

bool Go2Sport::enable_sport_mode() {
    if (!initialized_) {
        return false;
    }
    int32_t status = 0;
    const int32_t ret = robot_state_.ServiceSwitch("sport_mode", 1, status);
    if (ret != 0) {
        std::cerr << "ServiceSwitch(sport_mode, 1) failed: " << ret << " status=" << status << std::endl;
        return false;
    }
    std::cout << "Enabled sport_mode" << std::endl;
    return true;
}

bool Go2Sport::balance_stand() {
    if (!initialized_) {
        return false;
    }
    const int32_t ret = sport_.BalanceStand();
    if (ret != 0) {
        std::cerr << "BalanceStand() failed: " << ret << std::endl;
        return false;
    }
    std::cout << "Sport mode: BalanceStand" << std::endl;
    return true;
}

}  // namespace go2_deploy

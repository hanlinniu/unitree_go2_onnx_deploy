#include "go2_deploy/go2_lowlevel.hpp"

#include <chrono>
#include <cstring>
#include <thread>

#include <unitree/robot/channel/channel_factory.hpp>

namespace go2_deploy {

uint32_t crc32_unitree_lowcmd(unitree_go::msg::dds_::LowCmd_& cmd) {
    cmd.crc() = 0;
    const auto* bytes = reinterpret_cast<const uint32_t*>(&cmd);
    constexpr uint32_t len = (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1;

    uint32_t crc = 0xFFFFFFFF;
    constexpr uint32_t polynomial = 0x04C11DB7;
    for (uint32_t i = 0; i < len; ++i) {
        uint32_t data = bytes[i];
        uint32_t xbit = 1u << 31;
        for (int bit = 0; bit < 32; ++bit) {
            if (crc & 0x80000000) {
                crc <<= 1;
                crc ^= polynomial;
            } else {
                crc <<= 1;
            }
            if (data & xbit) {
                crc ^= polynomial;
            }
            xbit >>= 1;
        }
    }
    return crc;
}

void Go2LowLevel::init(const std::string& network_interface) {
    unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);
    lowcmd_pub_.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>("rt/lowcmd"));
    lowcmd_pub_->InitChannel();
    lowstate_sub_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>("rt/lowstate"));
    lowstate_sub_->InitChannel([this](const void* msg) { on_low_state(msg); }, 1);
    wireless_sub_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
        "rt/wirelesscontroller"));
    wireless_sub_->InitChannel([this](const void* msg) { on_wireless(msg); }, 1);
}

bool Go2LowLevel::wait_for_connection(double timeout_sec) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (low_state_.tick() > 0) {
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

unitree_go::msg::dds_::LowState_ Go2LowLevel::get_low_state() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return low_state_;
}

unitree_go::msg::dds_::WirelessController_ Go2LowLevel::get_wireless() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return wireless_;
}

void Go2LowLevel::publish_lowcmd(unitree_go::msg::dds_::LowCmd_ cmd) {
    cmd.crc() = crc32_unitree_lowcmd(cmd);
    lowcmd_pub_->Write(cmd);
}

void Go2LowLevel::on_low_state(const void* message) {
    const auto* msg = static_cast<const unitree_go::msg::dds_::LowState_*>(message);
    std::lock_guard<std::mutex> lock(state_mutex_);
    low_state_ = *msg;
}

void Go2LowLevel::on_wireless(const void* message) {
    const auto* msg = static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
    std::lock_guard<std::mutex> lock(state_mutex_);
    wireless_ = *msg;
}

}  // namespace go2_deploy

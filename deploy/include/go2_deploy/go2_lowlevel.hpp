#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace go2_deploy {

uint32_t crc32_unitree_lowcmd(unitree_go::msg::dds_::LowCmd_& cmd);

class Go2LowLevel {
public:
    void init(const std::string& network_interface);
    bool wait_for_connection(double timeout_sec = 10.0);

    unitree_go::msg::dds_::LowState_ get_low_state() const;
    unitree_go::msg::dds_::WirelessController_ get_wireless() const;

    void publish_lowcmd(unitree_go::msg::dds_::LowCmd_ cmd);

private:
    void on_low_state(const void* message);
    void on_wireless(const void* message);

    unitree_go::msg::dds_::LowState_ low_state_{};
    unitree_go::msg::dds_::WirelessController_ wireless_{};
    mutable std::mutex state_mutex_;

    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_pub_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_sub_;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> wireless_sub_;
};

}  // namespace go2_deploy

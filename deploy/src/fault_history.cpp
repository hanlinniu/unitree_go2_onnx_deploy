#include "go2_deploy/fault_history.hpp"

#include <algorithm>
#include <stdexcept>

namespace go2_deploy {

ProprioHistoryBuffer::ProprioHistoryBuffer(int history_len, int proprio_dim)
    : history_len_(history_len), proprio_dim_(proprio_dim), history_(history_len, std::vector<float>(proprio_dim, 0.f)) {
    if (history_len_ <= 0 || proprio_dim_ <= 0) {
        throw std::runtime_error("ProprioHistoryBuffer requires positive history_len and proprio_dim");
    }
}

void ProprioHistoryBuffer::reset() {
    step_count_ = 0;
    for (auto& frame : history_) {
        std::fill(frame.begin(), frame.end(), 0.f);
    }
}

void ProprioHistoryBuffer::push(const std::vector<float>& proprio) {
    if (static_cast<int>(proprio.size()) != proprio_dim_) {
        throw std::runtime_error("ProprioHistoryBuffer push size mismatch");
    }

    step_count_ += 1;
    if (step_count_ <= 1) {
        for (auto& frame : history_) {
            frame = proprio;
        }
        return;
    }

    for (int i = 0; i < history_len_ - 1; ++i) {
        history_[static_cast<size_t>(i)] = history_[static_cast<size_t>(i + 1)];
    }
    history_.back() = proprio;
}

std::vector<float> ProprioHistoryBuffer::flatten() const {
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(history_len_ * proprio_dim_));
    for (const auto& frame : history_) {
        flat.insert(flat.end(), frame.begin(), frame.end());
    }
    return flat;
}

}  // namespace go2_deploy

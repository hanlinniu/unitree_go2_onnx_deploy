#pragma once

#include <cstddef>
#include <vector>

namespace go2_deploy {

class ProprioHistoryBuffer {
public:
    ProprioHistoryBuffer(int history_len, int proprio_dim);

    void reset();
    void push(const std::vector<float>& proprio);
    std::vector<float> flatten() const;

    int history_len() const { return history_len_; }
    int proprio_dim() const { return proprio_dim_; }

private:
    int history_len_;
    int proprio_dim_;
    int step_count_{0};
    std::vector<std::vector<float>> history_;
};

}  // namespace go2_deploy

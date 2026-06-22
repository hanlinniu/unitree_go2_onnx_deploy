#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace go2_deploy {

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& model_path);

    std::vector<float> infer(const std::vector<float>& observations);

    int num_actions() const { return num_actions_; }

private:
    static Ort::SessionOptions make_session_options();

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> input_names_storage_;
    std::vector<std::string> output_names_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    int num_actions_ = 12;
};

}  // namespace go2_deploy

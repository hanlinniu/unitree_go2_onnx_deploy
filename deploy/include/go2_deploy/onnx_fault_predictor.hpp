#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace go2_deploy {

struct FaultDiagnosis {
    std::vector<float> failed_joint_prob;
    std::vector<float> severity;
    std::vector<float> locked_joint_prob;
};

class OnnxFaultPredictor {
public:
    explicit OnnxFaultPredictor(const std::string& model_path);

    FaultDiagnosis infer(const std::vector<float>& fault_obs_history) const;

    int fault_observation_dim() const { return fault_obs_dim_; }
    int num_joints() const { return num_joints_; }

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
    int fault_obs_dim_{0};
    int num_joints_{12};
};

void print_fault_diagnosis(const FaultDiagnosis& diagnosis, const std::vector<std::string>& joint_names);

}  // namespace go2_deploy

#include "go2_deploy/onnx_fault_predictor.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace go2_deploy {

Ort::SessionOptions OnnxFaultPredictor::make_session_options() {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    return options;
}

OnnxFaultPredictor::OnnxFaultPredictor(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "go2_fault_predictor"),
      session_options_(make_session_options()),
      session_(env_, model_path.c_str(), session_options_) {
    if (session_.GetInputCount() != 1) {
        throw std::runtime_error("Expected exactly one fault predictor input");
    }
    if (session_.GetOutputCount() != 3) {
        throw std::runtime_error("Expected three fault predictor outputs");
    }

    Ort::TypeInfo type_info = session_.GetInputTypeInfo(0);
    const auto input_shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape.size() != 2) {
        throw std::runtime_error("Expected fault_obs_history shape [batch, fault_obs_dim]");
    }
    fault_obs_dim_ = static_cast<int>(input_shape[1]);

    auto input_name = session_.GetInputNameAllocated(0, allocator_);
    input_names_storage_.push_back(input_name.get());
    input_names_.push_back(input_names_storage_.back().c_str());

    for (size_t i = 0; i < 3; ++i) {
        auto output_name = session_.GetOutputNameAllocated(i, allocator_);
        const char* raw_name = output_name.get();
        output_names_storage_.push_back(raw_name != nullptr ? raw_name : "");
        output_names_.push_back(output_names_storage_.back().c_str());
    }

    if (output_names_storage_.size() != 3) {
        throw std::runtime_error("Failed to read fault predictor output names");
    }
    std::cout << "Fault predictor outputs:";
    for (const auto& name : output_names_storage_) {
        std::cout << " " << (name.empty() ? "<unnamed>" : name);
    }
    std::cout << std::endl;

    Ort::TypeInfo failed_type = session_.GetOutputTypeInfo(0);
    const auto failed_shape = failed_type.GetTensorTypeAndShapeInfo().GetShape();
    if (failed_shape.size() == 2) {
        num_joints_ = static_cast<int>(failed_shape[1]) - 1;
    }
}

FaultDiagnosis OnnxFaultPredictor::infer(const std::vector<float>& fault_obs_history) {
    if (static_cast<int>(fault_obs_history.size()) != fault_obs_dim_) {
        throw std::runtime_error("fault_obs_history size does not match ONNX model");
    }

    std::vector<int64_t> shape = {1, static_cast<int64_t>(fault_obs_dim_)};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(fault_obs_history.data()),
        fault_obs_history.size(),
        shape.data(),
        shape.size());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        nullptr,
        0);

    if (outputs.size() != 3) {
        throw std::runtime_error("Expected three fault predictor outputs from ONNX Runtime");
    }

    auto copy_output = [](Ort::Value& value) {
        float* data = value.GetTensorMutableData<float>();
        const auto out_shape = value.GetTensorTypeAndShapeInfo().GetShape();
        size_t count = 1;
        for (auto dim : out_shape) {
            count *= static_cast<size_t>(dim);
        }
        return std::vector<float>(data, data + count);
    };

    FaultDiagnosis diagnosis;
    diagnosis.failed_joint_prob = copy_output(outputs[0]);
    diagnosis.severity = copy_output(outputs[1]);
    diagnosis.locked_joint_prob = copy_output(outputs[2]);
    return diagnosis;
}

void print_fault_diagnosis(const FaultDiagnosis& diagnosis, const std::vector<std::string>& joint_names) {
    if (diagnosis.failed_joint_prob.empty() || diagnosis.severity.empty() || diagnosis.locked_joint_prob.empty()) {
        return;
    }

    const int healthy_idx = static_cast<int>(diagnosis.failed_joint_prob.size()) - 1;
    int top_failed_idx = 0;
    for (int i = 1; i <= healthy_idx; ++i) {
        if (diagnosis.failed_joint_prob[static_cast<size_t>(i)] >
            diagnosis.failed_joint_prob[static_cast<size_t>(top_failed_idx)]) {
            top_failed_idx = i;
        }
    }

    int top_locked_idx = 0;
    for (int i = 1; i < static_cast<int>(diagnosis.locked_joint_prob.size()); ++i) {
        if (diagnosis.locked_joint_prob[static_cast<size_t>(i)] >
            diagnosis.locked_joint_prob[static_cast<size_t>(top_locked_idx)]) {
            top_locked_idx = i;
        }
    }

    const float healthy_prob = diagnosis.failed_joint_prob[static_cast<size_t>(healthy_idx)];
    std::ostringstream failed_desc;
    failed_desc << std::fixed << std::setprecision(2);
    if (top_failed_idx == healthy_idx) {
        failed_desc << "none/healthy p=" << healthy_prob;
    } else if (top_failed_idx >= 0 && top_failed_idx < static_cast<int>(joint_names.size())) {
        failed_desc << joint_names[static_cast<size_t>(top_failed_idx)] << " p="
                    << diagnosis.failed_joint_prob[static_cast<size_t>(top_failed_idx)] << " strength="
                    << diagnosis.severity[static_cast<size_t>(top_failed_idx)];
    }

    std::ostringstream locked_desc;
    locked_desc << std::fixed << std::setprecision(2)
                  << joint_names[static_cast<size_t>(top_locked_idx)] << " p="
                  << diagnosis.locked_joint_prob[static_cast<size_t>(top_locked_idx)];

    std::cout << "[world_model] predicted_failed=" << failed_desc.str()
              << " predicted_locked=" << locked_desc.str() << " healthy_prob=" << healthy_prob << std::endl;
}

}  // namespace go2_deploy

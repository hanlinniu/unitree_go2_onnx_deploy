#include "go2_deploy/onnx_policy.hpp"

#include "go2_deploy/onnx_io_util.hpp"

#include <iostream>
#include <stdexcept>

namespace go2_deploy {

Ort::SessionOptions OnnxPolicy::make_session_options() {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    return options;
}

OnnxPolicy::OnnxPolicy(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "go2_onnx_policy"),
      session_options_(make_session_options()),
      session_(env_, model_path.c_str(), session_options_) {
    const size_t num_inputs = session_.GetInputCount();
    if (num_inputs != 1) {
        throw std::runtime_error("Expected exactly one ONNX input");
    }

    Ort::TypeInfo type_info = session_.GetInputTypeInfo(0);
    input_shape_ = type_info.GetTensorTypeAndShapeInfo().GetShape();
    if (input_shape_.size() != 2) {
        throw std::runtime_error("Expected input shape [batch, obs_dim]");
    }
    if (input_shape_[0] <= 0) {
        input_shape_[0] = 1;
    }

    input_names_storage_.push_back(
        resolve_onnx_input_name(session_, allocator_, 0, "observations"));
    bind_onnx_name_ptrs(input_names_storage_, input_names_);

    const size_t num_outputs = session_.GetOutputCount();
    if (num_outputs != 1) {
        throw std::runtime_error("Expected exactly one ONNX output");
    }
    output_names_storage_.push_back(
        resolve_onnx_output_name(session_, allocator_, 0, "actions"));
    bind_onnx_name_ptrs(output_names_storage_, output_names_);

    std::cout << "Policy ONNX IO: input=" << input_names_storage_[0]
              << " output=" << output_names_storage_[0] << std::endl;

    Ort::TypeInfo output_type = session_.GetOutputTypeInfo(0);
    const auto output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 2) {
        throw std::runtime_error("Expected output shape [batch, action_dim]");
    }
    num_actions_ = static_cast<int>(output_shape[1]);
}

std::vector<float> OnnxPolicy::infer(const std::vector<float>& observations) {
    if (static_cast<int>(observations.size()) != static_cast<int>(input_shape_[1])) {
        throw std::runtime_error("Observation size does not match ONNX model");
    }

    std::vector<int64_t> shape = input_shape_;
    shape[0] = 1;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(observations.data()),
        observations.size(),
        shape.data(),
        shape.size());

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size());

    if (outputs.empty()) {
        throw std::runtime_error("Policy ONNX Runtime returned no outputs");
    }

    float* out_data = outputs[0].GetTensorMutableData<float>();
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = 1;
    for (auto dim : out_shape) {
        count *= static_cast<size_t>(dim);
    }
    return std::vector<float>(out_data, out_data + count);
}

}  // namespace go2_deploy

#pragma once

#include <onnxruntime_cxx_api.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace go2_deploy {

inline std::string resolve_onnx_input_name(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    size_t index,
    const char* fallback) {
    auto name = session.GetInputNameAllocated(index, allocator);
    const char* raw = name.get();
    if (raw != nullptr && raw[0] != '\0') {
        return std::string(raw);
    }
    return std::string(fallback);
}

inline std::string resolve_onnx_output_name(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    size_t index,
    const char* fallback) {
    auto name = session.GetOutputNameAllocated(index, allocator);
    const char* raw = name.get();
    if (raw != nullptr && raw[0] != '\0') {
        return std::string(raw);
    }
    return std::string(fallback);
}

inline void bind_onnx_name_ptrs(
    const std::vector<std::string>& names,
    std::vector<const char*>& name_ptrs) {
    name_ptrs.clear();
    name_ptrs.reserve(names.size());
    for (const auto& name : names) {
        if (name.empty()) {
            throw std::runtime_error("ONNX tensor name cannot be empty");
        }
        name_ptrs.push_back(name.c_str());
    }
}

}  // namespace go2_deploy

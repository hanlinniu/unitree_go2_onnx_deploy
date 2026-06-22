# Third-party dependencies

## ONNX Runtime

Download and extract one of the following into this directory:

- **x86_64 Linux (dev PC):** [onnxruntime-linux-x64-1.23.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz)
- **aarch64 / Orin NX:** [onnxruntime-linux-aarch64-1.23.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz)

Expected layout:

```
deploy/thirdparty/onnxruntime-linux-x64-1.23.2/
  include/onnxruntime_cxx_api.h
  lib/libonnxruntime.so*
```

Edit `deploy/robots/go2/CMakeLists.txt` if your folder name differs.

## unitree_sdk2

Install from [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) and ensure CMake can find `unitree_sdk2`.

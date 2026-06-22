# Third-party dependencies

## ONNX Runtime

Download and extract **one** bundle into this directory (`deploy/thirdparty/`).

### Orin NX (aarch64) — matches [unitree_cpp_deploy](https://github.com/hanlinniu/unitree_cpp_deploy) tutorial

```bash
cd deploy/thirdparty
wget https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2
tar -xjf onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2
```

Expected folder: `onnxruntime-linux-aarch64-gpu-1.16.0/`

### x86_64 Linux (dev PC)

```bash
cd deploy/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz
tar -xzf onnxruntime-linux-x64-1.23.2.tgz
```

Or run: `../../scripts/setup_onnxruntime.sh` (x64 / aarch64 CPU 1.23.2).

### Layout (both)

```
deploy/thirdparty/onnxruntime-linux-.../
  include/onnxruntime_cxx_api.h
  lib/libonnxruntime.so*
```

## Build

From `deploy/robots/go2/build`:

```bash
rm -rf *
cmake ..          # auto-picks aarch64 or x64 default
make -j$(nproc)
```

Override path (absolute or relative to `deploy/robots/go2/`):

```bash
cmake .. -DONNXRUNTIME_ROOT=../../thirdparty/onnxruntime-linux-aarch64-gpu-1.16.0
```

**Note:** Do not use `../../thirdparty/...` from inside `build/` on the command line unless you pass an **absolute** path — CMake resolves `-D` relative paths from `deploy/robots/go2/`, not from `build/`. Plain `cmake ..` is recommended.

## unitree_sdk2

Install from [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2):

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. && sudo make install
```

This also installs **CycloneDDS** headers under `/usr/local/include/ddscxx` (required for compilation).

If you see `dds/topic/TopicTraits.hpp: No such file or directory`, unitree_sdk2 or CycloneDDS is not fully installed — rebuild and `sudo make install` unitree_sdk2.

# unitree_go2_onnx_deploy

C++ ONNX Runtime deployment for policies trained with [unitree_legged_gym](https://github.com/unitreerobotics/unitree_rl_gym) (GO2 velocity / fault-world-model with proprio-only actor).

Inspired by [unitree_cpp_deploy](https://github.com/wty-yy/unitree_cpp_deploy): FixStand uses high PD gains; RL inference uses sim-matched **Kp=20 / Kd=0.5** from `deploy.yaml`.

## Features

- Export `model_*.pt` → `policy.onnx` + `params/deploy.yaml`
- 45-dim proprio observation (matches Isaac Gym / Kaixin ROS2 deploy)
- Sim joint order internally (FL, FR, RL, RR); remaps to Unitree motor order on `LowCmd`
- FSM: **Sport** → **FixStand** (R1) → **Velocity** (Y) → back to **Sport** (R2)
- ONNX Runtime inference in a real-time thread (default 50 Hz)

## Requirements

- Ubuntu 20.04/22.04 on x86_64 (dev) or aarch64 (Orin NX onboard)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)
- ONNX Runtime 1.23.x (see `deploy/thirdparty/README.md`)
- `libyaml-cpp-dev`, `libboost-program-options-dev`, `libeigen3-dev`
- Python 3 + PyTorch (export only, on training machine)

Training repo path (default): `~/unitree_legged_gym`

## 1. Export policy from checkpoint

Export both the walking actor (`policy.onnx`) and fault diagnosis stack (`fault_predictor.onnx`) from a `model_*.pt` checkpoint:

```bash
cd ~/unitree_go2_onnx_deploy

python3 scripts/export_onnx_from_checkpoint.py \
  --checkpoint ~/unitree_legged_gym/unitree_rl_gym/logs/unitree_legged_gym/20260616_after8000_30000iteration_samethigh0.8_onestepworldmodel_10stephistoryfourthmlp_independentfaultpredictor-alienware/model_30000.pt \
  --output_dir policies/fault_predictor_20260616 \
  --policy_kp 40 \
  --policy_kd 1
```

Or from any other training run:

```bash
python3 scripts/export_onnx_from_checkpoint.py \
  --checkpoint ~/unitree_legged_gym/unitree_rl_gym/logs/rough_go2/<your_run>/model_30000.pt \
  --output_dir policies/my_run \
  --policy_kp 20 \
  --policy_kd 0.5
```

This creates:

```
policies/my_run/
  exported/policy.onnx
  exported/fault_predictor.onnx
  params/deploy.yaml
  export_meta.json
```

**Note:** Export supports `fault_belief_in_actor=False` (standard GO2 fault-world-model config). The actor is proprio-only (45→12). The fault predictor runs in parallel for onboard diagnostics/logging (same role as Kaixin `fault_explanation()`); it does not change actor actions with this training config.

Update `deploy/robots/go2/config/config.yaml`:

```yaml
Velocity:
  policy_dir: ../../../../policies/my_run
```

## 2. Build C++ deploy

```bash
# Install ONNX Runtime under deploy/thirdparty/ (see deploy/thirdparty/README.md)
# Orin NX example:
#   cd deploy/thirdparty
#   wget .../onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2 && tar -xjf ...

cd deploy/robots/go2
mkdir -p build && cd build
rm -rf *    # if re-configuring

cmake ..    # auto-detects aarch64 (Orin) vs x64
make -j$(nproc)
```

Optional override (path relative to `deploy/robots/go2/`, not `build/`):

```bash
cmake .. -DONNXRUNTIME_ROOT=../../thirdparty/onnxruntime-linux-aarch64-gpu-1.16.0
```

## 3. Run on robot

Keep Unitree **sport_mode** enabled at startup. The deploy binary starts in sport mode and only takes low-level control after **R1**.

```bash
cd ~/unitree_go2_onnx_deploy/deploy/robots/go2/build
```

### Gamepad workflow

1. Start `./go2_onnx_ctrl` (robot stays in sport mode).
2. **R1** — disable sport mode, run FixStand (~2 s).
3. **Y** — start ONNX policy (scenario countdown begins here).
4. After `--healthy_s` seconds, the fault or lock activates on the chosen calf.
5. **R2** — stop custom control and return to sport mode (BalanceStand).

| Input | Mode |
|-------|------|
| **R1** | Disable sport mode → FixStand |
| **Y** | ONNX policy (after FixStand completes) |
| **R2** | Stop custom control → re-enable sport mode (BalanceStand) |

### Calf failure (`random_fault`)

Weakens PD gains on one calf joint after `--healthy_s` seconds. Use `--fault_sthres` as the gain scale (e.g. `0.4` = 40% of normal Kp/Kd).

**Random calf, fixed forward command:**

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario random_fault \
  --healthy_s 3.0 \
  --fault_sthres 0.4 \
  --fault_calf RANDOM \
  --command_source fixed \
  --vx 0.4 \
  --vy 0.0
```

**Specific leg (RR calf), joystick commands:**

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario random_fault \
  --healthy_s 3.0 \
  --fault_sthres 0.1 \
  --fault_calf RR
```

**Severe fault on a random calf (10% PD gains):**

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario random_fault \
  --healthy_s 3.0 \
  --fault_sthres 0.1 \
  --fault_calf RANDOM \
  --command_source fixed \
  --vx 0.4 \
  --vy 0.0 \
  --vyaw 0.0 \
  --no_fixed_command_joystick_yaw
```

### Calf lock (`random_lock`)

Forces one calf joint to `--lock_angle_deg` after `--healthy_s` seconds of healthy walking.

**Random calf, lock at -120°:**

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario random_lock \
  --healthy_s 3.0 \
  --lock_angle_deg -120 \
  --fault_calf RANDOM \
  --command_source fixed \
  --vx 0.4 \
  --vy 0.0
```

**Specific leg (RR calf), lock at -120°:**

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario random_lock \
  --healthy_s 3.0 \
  --lock_angle_deg -120 \
  --fault_calf RR \
  --command_source fixed \
  --vx 0.4 \
  --vy 0.0
```

**Other legs:** set `--fault_calf` to `FL`, `FR`, `RL`, or `RR`.

### Healthy baseline (no fault injection)

```bash
./go2_onnx_ctrl -n eth0 \
  --scenario healthy \
  --command_source fixed \
  --vx 0.4 \
  --vy 0.0
```

### Scenario / command arguments

| Argument | Meaning |
|----------|---------|
| `--scenario` | `healthy`, `random_fault`, or `random_lock` (aliases: `fault`, `lock`) |
| `--healthy_s` | Seconds of normal policy before fault/lock activates |
| `--fault_sthres` | PD gain scale on failed calf (`random_fault`) |
| `--lock_angle_deg` | Locked calf target angle in degrees (`random_lock`) |
| `--fault_calf` | `RANDOM`, `FL`, `FR`, `RL`, or `RR` |
| `--command_source` | `joystick` (default) or `fixed` |
| `--vx`, `--vy`, `--vyaw` | Fixed velocity commands when `--command_source fixed` |
| `--no_fixed_command_joystick_yaw` | Also fix vyaw; default is right-stick yaw with fixed vx/vy |

## Observation layout (45)

Same as `legged_robot.py` / Kaixin deploy:

| Index | Content |
|-------|---------|
| 0–2 | Base angular velocity × 0.25 |
| 3–5 | Projected gravity |
| 6–8 | Velocity commands × [2, 2, 0.25] |
| 9–20 | (q − q_default) |
| 21–32 | qdot × 0.05 |
| 33–44 | Last actions |

Commands come from the wireless controller sticks (`ly`, `-lx`, `-rx`) scaled by `max_cmd` in `deploy.yaml`, or from `--command_source fixed --vx ... --vy ...` (vyaw from right stick by default).

## PD gains

| Phase | Kp / Kd | Config |
|-------|---------|--------|
| FixStand | 60/80/80, 5/4/4 per leg | `config/config.yaml` |
| Policy | 20 / 0.5 uniform | `params/deploy.yaml` (`policy_stiffness` / `policy_damping`) |

Re-export with `--policy_kp` / `--policy_kd` if you change training/deploy gains.

## Comparison with Kaixin Python deploy

| | Kaixin ROS2 + PyTorch | This repo |
|--|----------------------|-----------|
| Inference | TorchScript / CUDA | ONNX Runtime C++ |
| Middleware | ROS2 | unitree_sdk2 DDS |
| Fault diagnostics | `fault_explanation()` | `fault_predictor.onnx` + throttled `[fault_predictor]` logs |
| Warm-up | 3 dummy GPU runs | 3 dummy ORT runs on Velocity enter |

To add world-model dynamics prediction later, export `dynamics_head` as a separate ONNX model.

## License

MIT

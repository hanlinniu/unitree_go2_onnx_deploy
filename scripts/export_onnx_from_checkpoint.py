#!/usr/bin/env python3
"""Export a unitree_legged_gym GO2 checkpoint to ONNX + deploy.yaml for C++ deploy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


# Default sim joint order: FL, FR, RL, RR (hip, thigh, calf).
DEFAULT_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Sim index -> Unitree motor index (FR, FL, RR, RL each x3).
DEFAULT_JOINT_IDS_MAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def class_to_dict(obj: Any) -> Any:
    if not hasattr(obj, "__dict__"):
        return obj
    result: Dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if callable(value):
            continue
        result[key] = class_to_dict(value)
    return result


def load_actor_critic(checkpoint: Path, unitree_gym_root: Path, task: str, device: str):
    sys.path.insert(0, str(unitree_gym_root / "rsl_rl"))
    sys.path.insert(0, str(unitree_gym_root / "unitree_rl_gym"))

    from rsl_rl import modules as rsl_modules

    if task != "go2":
        raise RuntimeError(f"Unsupported task {task}")

    from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO

    env_cfg = GO2RoughCfg()
    train_cfg = GO2RoughCfgPPO()
    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    policy_class = getattr(rsl_modules, policy_class_name)

    num_actor_obs = env_cfg.env.num_observations
    num_critic_obs = env_cfg.env.num_privileged_obs or num_actor_obs
    num_actions = env_cfg.env.num_actions

    model = policy_class(
        num_actor_obs,
        num_critic_obs,
        num_actions,
        **policy_cfg,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, env_cfg, train_cfg, policy_cfg


class ActorOnnxWrapper(torch.nn.Module):
    """Exportable actor for GO2 proprio-only policies (fault_belief_in_actor=False)."""

    def __init__(self, actor: torch.nn.Module):
        super().__init__()
        self.actor = actor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)


def build_deploy_yaml(
    env_cfg,
    policy_cfg,
    stiffness: float,
    damping: float,
    clip_actions: float,
) -> dict:
    default_angles = env_cfg.init_state.default_joint_angles
    default_joint_pos = [float(default_angles[name]) for name in DEFAULT_JOINT_NAMES]

    train_stiffness = getattr(env_cfg.control, "stiffness", {"joint": stiffness})
    train_damping = getattr(env_cfg.control, "damping", {"joint": damping})
    if isinstance(train_stiffness, dict) and "joint" in train_stiffness:
        sim_kp = float(train_stiffness["joint"])
    else:
        sim_kp = stiffness
    if isinstance(train_damping, dict) and "joint" in train_damping:
        sim_kd = float(train_damping["joint"])
    else:
        sim_kd = damping

    return {
        "num_observations": int(env_cfg.env.num_observations),
        "num_actions": int(env_cfg.env.num_actions),
        "step_dt": float(env_cfg.sim.dt * env_cfg.control.decimation),
        "sim_dt": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.control.decimation),
        "joint_ids_map": DEFAULT_JOINT_IDS_MAP,
        "default_joint_pos": default_joint_pos,
        "stiffness": [sim_kp] * 12,
        "damping": [sim_kd] * 12,
        "policy_stiffness": [stiffness] * 12,
        "policy_damping": [damping] * 12,
        "action_scale": float(env_cfg.control.action_scale),
        "clip_actions": float(clip_actions),
        "obs_scales": {
            "ang_vel": float(env_cfg.normalization.obs_scales.ang_vel),
            "dof_pos": float(env_cfg.normalization.obs_scales.dof_pos),
            "dof_vel": float(env_cfg.normalization.obs_scales.dof_vel),
        },
        "commands_scale": [
            float(env_cfg.normalization.obs_scales.lin_vel),
            float(env_cfg.normalization.obs_scales.lin_vel),
            float(env_cfg.normalization.obs_scales.ang_vel),
        ],
        "clip_observations": float(env_cfg.normalization.clip_observations),
        "fault_belief_in_actor": bool(policy_cfg.get("fault_belief_in_actor", False)),
        "fault_history_len": int(policy_cfg.get("fault_history_len", 1)),
        "joint_names": DEFAULT_JOINT_NAMES,
        "commands": {
            "max_cmd": [1.0, 1.0, 1.0],
        },
    }


def max_supported_onnx_opset() -> int:
    """Highest ONNX opset supported by the installed PyTorch build."""
    try:
        import torch.onnx.symbolic_helper as symbolic_helper

        return int(symbolic_helper._export_onnx_opset_version)
    except (ImportError, AttributeError, TypeError, ValueError):
        return 11


def resolve_onnx_opset(requested: int) -> int:
    """Clamp requested opset to what this PyTorch build can export."""
    max_opset = max_supported_onnx_opset()
    opset = min(requested, max_opset)
    if opset != requested:
        print(
            f"Warning: ONNX opset {requested} unsupported by torch {torch.__version__}; "
            f"using opset {opset} instead."
        )
    return opset


def is_torchscript_module(module: torch.nn.Module) -> bool:
    if isinstance(module, torch.jit.ScriptModule):
        return True
    return type(module).__name__ in {"RecursiveScriptModule", "ScriptModule", "TopLevelTracedModule"}


def clone_script_sequential_to_eager(
    script_module: torch.nn.Module, num_obs: int
) -> torch.nn.Sequential:
    """Rebuild a scripted nn.Sequential actor as an eager module for ONNX tracing."""
    state = script_module.state_dict()
    linear_indices = sorted(
        int(key.split(".", 1)[0])
        for key in state
        if key.endswith(".weight") and key.count(".") == 1
    )
    if not linear_indices:
        raise RuntimeError("Could not infer Linear layers from TorchScript state_dict")

    activations = (torch.nn.ELU, torch.nn.ReLU, torch.nn.Tanh, torch.nn.LeakyReLU)
    dummy = torch.zeros(1, num_obs, dtype=torch.float32)
    with torch.no_grad():
        reference = script_module(dummy)

    def build_model(act: type) -> torch.nn.Sequential:
        layers: list[torch.nn.Module] = []
        for i, idx in enumerate(linear_indices):
            weight = state[f"{idx}.weight"]
            layers.append(
                torch.nn.Linear(
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                    bias=f"{idx}.bias" in state,
                )
            )
            if i + 1 < len(linear_indices):
                layers.append(act())
        model = torch.nn.Sequential(*layers)
        model.load_state_dict(state, strict=True)
        return model

    for act in activations:
        model = build_model(act)
        model.eval()
        with torch.no_grad():
            candidate = model(dummy)
        if torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5):
            print(
                "Rebuilt TorchScript actor as eager Sequential "
                f"with {act.__name__} activations for ONNX export"
            )
            return model

    raise RuntimeError("Could not rebuild TorchScript actor with known activations")


def _run_torch_onnx_export(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    out_path: Path,
    opset: int,
    *,
    dynamic_batch: bool,
) -> None:
    kwargs = {
        "input_names": ["observations"],
        "output_names": ["actions"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        kwargs["dynamic_axes"] = {
            "observations": {0: "batch"},
            "actions": {0: "batch"},
        }
    torch.onnx.export(model, dummy, str(out_path), **kwargs)


def export_onnx(actor: torch.nn.Module, out_path: Path, num_obs: int, opset: int) -> None:
    opset = resolve_onnx_opset(opset)
    dummy = torch.zeros(1, num_obs, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_torchscript_module(actor):
        script_actor = actor.cpu().eval()
        print("Exporting TorchScript actor directly to ONNX")
        try:
            _run_torch_onnx_export(script_actor, dummy, out_path, opset, dynamic_batch=False)
            return
        except Exception as exc:
            print(f"Direct TorchScript ONNX export failed ({exc}); rebuilding eager copy...")
            actor = clone_script_sequential_to_eager(script_actor, num_obs)

    wrapper = ActorOnnxWrapper(actor).cpu().eval()
    _run_torch_onnx_export(wrapper, dummy, out_path, opset, dynamic_batch=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model_*.pt from unitree_legged_gym training.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Policy bundle directory (will create exported/ and params/).",
    )
    parser.add_argument(
        "--unitree_gym_root",
        type=Path,
        default=Path(os.environ.get("UNITREE_LEGGED_GYM_ROOT", "~/unitree_legged_gym")).expanduser(),
    )
    parser.add_argument("--task", default="go2", choices=("go2",))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--policy_kp",
        type=float,
        default=20.0,
        help="Real-robot policy Kp (uniform), written to deploy.yaml policy_stiffness.",
    )
    parser.add_argument(
        "--policy_kd",
        type=float,
        default=0.5,
        help="Real-robot policy Kd (uniform), written to deploy.yaml policy_damping.",
    )
    parser.add_argument(
        "--clip_actions",
        type=float,
        default=1.2,
        help="Action clip before action_scale on hardware (matches Kaixin deploy).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (auto-clamped to PyTorch max if unsupported).",
    )
    args = parser.parse_args()

    model, env_cfg, train_cfg, policy_cfg = load_actor_critic(
        args.checkpoint.expanduser(),
        args.unitree_gym_root.expanduser(),
        args.task,
        args.device,
    )

    if bool(getattr(model, "fault_belief_in_actor", False)):
        raise RuntimeError(
            "This exporter currently supports fault_belief_in_actor=False only "
            "(actor input = proprio). Re-export with a GO2 config where the actor "
            "does not consume fault latent, or extend ActorOnnxWrapper."
        )

    num_obs = int(env_cfg.env.num_observations)
    output_dir = args.output_dir.expanduser()
    exported_dir = output_dir / "exported"
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = exported_dir / "policy.onnx"
    export_onnx(model.actor, onnx_path, num_obs, args.opset)

    deploy = build_deploy_yaml(
        env_cfg,
        policy_cfg,
        stiffness=args.policy_kp,
        damping=args.policy_kd,
        clip_actions=args.clip_actions,
    )
    deploy_path = params_dir / "deploy.yaml"
    with deploy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(deploy, f, sort_keys=False)

    meta = {
        "checkpoint": str(args.checkpoint.expanduser()),
        "onnx": str(onnx_path),
        "deploy_yaml": str(deploy_path),
        "num_observations": num_obs,
        "num_actions": int(env_cfg.env.num_actions),
        "policy_class": train_cfg.runner.policy_class_name,
    }
    with (output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported ONNX: {onnx_path}")
    print(f"Wrote deploy params: {deploy_path}")
    print(f"Policy PD for real robot: Kp={args.policy_kp}, Kd={args.policy_kd}")


if __name__ == "__main__":
    main()

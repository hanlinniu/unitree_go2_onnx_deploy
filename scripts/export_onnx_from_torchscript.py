#!/usr/bin/env python3
"""Export a Kaixin / legged_gym TorchScript policy (.pt) to ONNX + deploy.yaml."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from export_onnx_from_checkpoint import (
    DEFAULT_JOINT_IDS_MAP,
    DEFAULT_JOINT_NAMES,
    ActorOnnxWrapper,
    export_onnx,
)


NUM_OBS = 45
NUM_ACTIONS = 12


def default_deploy_yaml(policy_kp: float, policy_kd: float, clip_actions: float) -> dict:
    """GO2 defaults aligned with Kaixin onboard + unitree_legged_gym base config."""
    default_joint_pos = [
        0.1, 0.8, -1.5,   # FL
        -0.1, 0.8, -1.5,  # FR
        0.1, 0.8, -1.5,   # RL
        -0.1, 0.8, -1.5,  # RR
    ]
    return {
        "num_observations": NUM_OBS,
        "num_actions": NUM_ACTIONS,
        "step_dt": 0.02,
        "sim_dt": 0.005,
        "decimation": 4,
        "joint_ids_map": DEFAULT_JOINT_IDS_MAP,
        "default_joint_pos": default_joint_pos,
        "stiffness": [25.0] * 12,
        "damping": [0.5] * 12,
        "policy_stiffness": [policy_kp] * 12,
        "policy_damping": [policy_kd] * 12,
        "action_scale": 0.25,
        "clip_actions": clip_actions,
        "obs_scales": {"ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
        "commands_scale": [2.0, 2.0, 0.25],
        "clip_observations": 100.0,
        "fault_belief_in_actor": False,
        "fault_history_len": 10,
        "joint_names": DEFAULT_JOINT_NAMES,
        "commands": {"max_cmd": [1.0, 1.0, 1.0]},
    }


class TorchScriptActorWrapper(nn.Module):
    """Export actor-only inference from a TorchScript bundle."""

    def __init__(self, module: torch.jit.ScriptModule):
        super().__init__()
        self.module = module

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if hasattr(self.module, "act_inference"):
            return self.module.act_inference(observations)
        return self.module(observations)


def resolve_actor_module(ts: torch.jit.ScriptModule) -> nn.Module:
    """Prefer standalone actor (stateless). Fall back to act_inference wrapper."""
    if hasattr(ts, "actor"):
        actor = ts.actor
        dummy = torch.zeros(1, NUM_OBS)
        with torch.no_grad():
            out = actor(dummy)
        if out.shape[-1] == NUM_ACTIONS:
            print("Using TorchScript submodule: actor (proprio-only, recommended for ONNX)")
            return actor
    print(
        "Warning: exporting via act_inference (may include world-model side effects). "
        "Prefer exporting from raw model_*.pt checkpoint when possible."
    )
    return TorchScriptActorWrapper(ts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript", type=Path, required=True, help="Path to policy_*.pt")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--policy_kp", type=float, default=20.0)
    parser.add_argument("--policy_kd", type=float, default=0.5)
    parser.add_argument("--clip_actions", type=float, default=1.2)
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (auto-clamped to PyTorch max if unsupported).",
    )
    args = parser.parse_args()

    ts_path = args.torchscript.expanduser()
    output_dir = args.output_dir.expanduser()
    exported_dir = output_dir / "exported"
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    ts = torch.jit.load(str(ts_path), map_location="cpu")
    ts.eval()

    actor = resolve_actor_module(ts)
    onnx_path = exported_dir / "policy.onnx"
    export_onnx(actor, onnx_path, NUM_OBS, args.opset)

    deploy = default_deploy_yaml(args.policy_kp, args.policy_kd, args.clip_actions)
    deploy_path = params_dir / "deploy.yaml"
    with deploy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(deploy, f, sort_keys=False)

    meta = {
        "torchscript": str(ts_path),
        "onnx": str(onnx_path),
        "deploy_yaml": str(deploy_path),
        "num_observations": NUM_OBS,
        "num_actions": NUM_ACTIONS,
    }
    with (output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported ONNX: {onnx_path}")
    print(f"Wrote deploy params: {deploy_path}")


if __name__ == "__main__":
    main()

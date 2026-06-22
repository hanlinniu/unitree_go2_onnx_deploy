#!/usr/bin/env python3
"""Export a traced TorchScript GO2 policy (.pt) to ONNX + deploy.yaml for C++ deploy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from export_onnx_from_checkpoint import (
    DEFAULT_JOINT_IDS_MAP,
    DEFAULT_JOINT_NAMES,
    export_onnx,
)


def default_deploy_yaml(
    policy_kp: float,
    policy_kd: float,
    clip_actions: float,
) -> dict:
    """Deploy params aligned with Kaixin / unitree_legged_gym GO2 defaults."""
    default_joint_pos = [
        0.1, 0.8, -1.5,   # FL
        -0.1, 0.8, -1.5,  # FR
        0.1, 0.8, -1.5,   # RL
        -0.1, 0.8, -1.5,  # RR
    ]
    return {
        "num_observations": 45,
        "num_actions": 12,
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
        "source_torchscript": True,
    }


def load_actor_from_torchscript(path: Path, device: str) -> torch.nn.Module:
    module = torch.jit.load(str(path), map_location=device)
    module.eval()

    if hasattr(module, "actor"):
        actor = module.actor
        actor.eval()
        return actor

    if hasattr(module, "act_inference"):
        class ActInferenceWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self.wrapped.act_inference(observations)

        return ActInferenceWrapper(module).eval()

    raise RuntimeError(
        f"{path} has no .actor or .act_inference. "
        "Re-export from unitree_legged_gym or pass a raw model_*.pt checkpoint."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torchscript",
        type=Path,
        required=True,
        help="Path to traced policy_*.pt (Kaixin traced_fault_tolerant export).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Policy bundle dir (creates exported/policy.onnx and params/deploy.yaml).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_obs", type=int, default=45)
    parser.add_argument("--policy_kp", type=float, default=20.0)
    parser.add_argument("--policy_kd", type=float, default=0.5)
    parser.add_argument("--clip_actions", type=float, default=1.2)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ts_path = args.torchscript.expanduser()
    output_dir = args.output_dir.expanduser()
    actor = load_actor_from_torchscript(ts_path, args.device)

    onnx_path = output_dir / "exported" / "policy.onnx"
    export_onnx(actor, onnx_path, args.num_obs, args.opset)

    deploy = default_deploy_yaml(args.policy_kp, args.policy_kd, args.clip_actions)
    deploy["source_file"] = str(ts_path)

    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    deploy_path = params_dir / "deploy.yaml"
    with deploy_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(deploy, f, sort_keys=False)

    meta = {
        "torchscript": str(ts_path),
        "onnx": str(onnx_path),
        "deploy_yaml": str(deploy_path),
    }
    with (output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported ONNX: {onnx_path}")
    print(f"Wrote deploy params: {deploy_path}")


if __name__ == "__main__":
    main()

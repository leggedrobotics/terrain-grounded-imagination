"""Export a training checkpoint in the fpm formats.

Writes ``model_<it>.pt`` (rsl-rl 3.3.0 ActorCritic, observation normalization baked into the actor),
``foot_model_<it>.pt`` (TorchScript FPM) and ``exported/`` (policy and FPM as ONNX, ``deploy.json``) for sim2sim.

    python scripts/export.py --ckpt logs/<run>/ckpt_100000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import re

import torch
import torch.nn as nn

import tgi.deploy
from tgi.mdp.interface import (ACTION_SCALE, DEFAULT_POSE, FOOTHOLD_SCAN_DIM, ISAAC_JOINTS, POLICY_OBS_DIM, PROPRIO_DIM,
                               SYSTEM_STATE_DIM)  # fmt: skip
from tgi.policy_io import load_policy_bundle

CRITIC_OBS_DIM = PROPRIO_DIM + 33 * 21 + FOOTHOLD_SCAN_DIM  # base env CriticCfg: policy terms + height_scan + foothold_scan
# --scan full: the critic has no foothold_scan
CRITIC_OBS_DIM_FULL = PROPRIO_DIM + 33 * 21
# the base env's actuator gains (Isaac joint order), used by sim2sim
GAINS = os.path.join(os.path.dirname(os.path.abspath(tgi.deploy.__file__)), "g1_gains.json")


def baked_actor(pi) -> nn.Sequential:
    """Fold the policy's running normalizers into the first linear layer -> plain MLP on raw observations."""
    mean = torch.cat((pi.obs_norm.mean, pi.extra_norm.mean))
    std = torch.cat((pi.obs_norm.std(), pi.extra_norm.std()))
    layers = [m for m in pi.net]
    first = layers[0]
    lin = nn.Linear(first.in_features, first.out_features)
    with torch.no_grad():
        lin.weight.copy_(first.weight / std)
        lin.bias.copy_(first.bias - (first.weight / std) @ mean)
    return nn.Sequential(lin, *layers[1:]).float().eval()


class _ActorCritic(nn.Module):
    """Parameter layout of rsl-rl-lib 3.3.0's ``ActorCritic`` (scalar std), without importing rsl-rl."""

    def __init__(self, actor: nn.Sequential, std: float, critic_dim: int = CRITIC_OBS_DIM):
        super().__init__()
        self.std = nn.Parameter(std * torch.ones(29))
        self.actor = actor
        dims, layers = [critic_dim, 512, 256, 128], []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d0, d1), nn.ELU()]
        self.critic = nn.Sequential(*layers, nn.Linear(128, 1))


def rsl_rl_checkpoint(actor: nn.Sequential, iteration: int, std: float = 0.3, critic_dim: int = CRITIC_OBS_DIM) -> dict:
    ac = _ActorCritic(actor, std, critic_dim)
    opt = torch.optim.Adam(ac.parameters(), lr=1e-3)
    return {"model_state_dict": ac.state_dict(), "optimizer_state_dict": opt.state_dict(), "iter": iteration,
            "infos": {"source": "Terrain Grounded Imagination (world-model training)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    b = load_policy_bundle(a.ckpt, "cpu")
    it = int(b.ck["iteration"])
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), f"export_{it:06d}")
    os.makedirs(os.path.join(out, "exported"), exist_ok=True)

    actor = baked_actor(b.pi)
    # the baked actor must match the trained policy
    full = b.fpm is None and b.args.get("scan", "foothold") == "full"
    o, e = torch.randn(64, PROPRIO_DIM), torch.randn(64, b.pi.extra_dim) * 0.3
    with torch.no_grad():
        err = (actor(torch.cat((o, e), -1)) - b.pi(o.unsqueeze(1), e)).abs().max().item()
    assert err < 1e-3, err
    torch.save(rsl_rl_checkpoint(actor, it, critic_dim=CRITIC_OBS_DIM_FULL if full else CRITIC_OBS_DIM),
               os.path.join(out, f"model_{it}.pt"))
    if b.fpm is None:  # --scan full (evaluate with FPM_POLICY_SCAN=full) or --fpm 0 (fixed footholds): no foot model
        torch.onnx.export(actor, (torch.zeros(1, PROPRIO_DIM + b.pi.extra_dim),), os.path.join(out, "exported", "policy.onnx"),
                          input_names=["obs"], output_names=["actions"], dynamo=False)
        kind = "full-height-scan" if full else "fixed-foothold"
        print(f"exported {kind} policy to {out} (baked-actor max err {err:.2e}); no foot model")
        return

    foot = torch.jit.script(b.fpm.exportable().eval())
    torch.jit.save(foot, os.path.join(out, f"foot_model_{it}.pt"))
    s = torch.randn(8, SYSTEM_STATE_DIM)
    assert (torch.jit.load(os.path.join(out, f"foot_model_{it}.pt"))(s) - b.fpm(s).flatten(-2)).abs().max() < 1e-4

    torch.onnx.export(actor, (torch.zeros(1, POLICY_OBS_DIM),), os.path.join(out, "exported", "policy.onnx"),
                      input_names=["obs"], output_names=["actions"], dynamo=False)
    torch.onnx.export(b.fpm.exportable().eval(), (torch.zeros(1, 1, SYSTEM_STATE_DIM),),
                      os.path.join(out, "exported", "foot_model.onnx"), input_names=["obs"], output_names=["pred"],
                      dynamic_axes={"obs": {0: "batch_size"}, "pred": {0: "batch_size"}}, dynamo=False)
    g = json.load(open(GAINS))
    assert g["joint_names"] == ISAAC_JOINTS
    cfg = {k: g[k] for k in ("kp", "kd", "effort_limit")}
    q0 = [next((v for p, v in DEFAULT_POSE.items() if re.fullmatch(p, j)), 0.0) for j in ISAAC_JOINTS]
    cfg.update({
        "interface": "fpm G1 locomotion env",
        "joint_names": ISAAC_JOINTS, "default_q": q0, "action_scale": ACTION_SCALE, "control_dt": 0.02,
        "policy_obs": "base_lin_vel(3), 0.2*base_ang_vel(3), projected_gravity(3), cmd(3), q-q0(29), 0.05*qd(29), "
                      "last_action(29, clip 10), foothold_scan(2 x [foothold_xy(2), 7x7 heights])",
        "fpm_input": "system_state(70): base_lin_vel, base_ang_vel, projected_gravity, cmd, q-q0, qd",
        "foothold_scan": "torso-mounted yaw-aligned 1.6x1.0 m grid, 0.05 m; 7x7 nodes around the nearest node to each "
                         "foothold (x-major); value clip(torso_z - terrain_z - 0.5, -1, 1)",
        "source_ckpt": os.path.abspath(a.ckpt), "iteration": it, "train_hours": b.ck["wall_time"] / 3600,
    })  # fmt: skip
    json.dump(cfg, open(os.path.join(out, "exported", "deploy.json"), "w"), indent=1)
    print(f"exported to {out} (baked-actor max err {err:.2e})")


if __name__ == "__main__":
    main()

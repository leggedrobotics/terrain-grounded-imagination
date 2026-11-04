"""Differentiable MDP heads shared by the simulator and imagination: state, rewards and terminations.

State ``S = (p, R, v, w, q, qd)``: pelvis position, rotation, linear and angular velocity, joint positions and
velocities. Contact-dependent rewards use the world model's foot-contact probabilities and the terrain under the feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .interface import JointMap, policy_proprio, system_state
from .kinematics import G1Kinematics
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat
from ..constants import NUM_J, POSE_STD, match_values
from .terrain import TerrainFeatures

INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J  # height comes from the terrain features


@dataclass
class RewardCfg:
    w_lin: float = 2.0
    std_lin: float = math.sqrt(0.25)
    w_ang: float = 2.0
    std_ang: float = math.sqrt(0.5)
    w_upright: float = 0.25
    std_upright: float = math.sqrt(0.2)
    w_pose: float = 2.5  # standing commands only (the base env's stand_still)
    w_limits: float = -1.0
    w_action_rate: float = -0.1
    w_clearance: float = -2.0  # only swing feet below the target height above the terrain
    clearance_target: float = 0.1
    w_slip: float = -0.1
    w_body_ang_vel: float = -0.05
    w_gait: float = 0.25
    w_foothold: float = -3.0  # contact x (unsupported sole share + 1 - support field)
    # per m of predicted foothold beyond reach_x / reach_y (root frame)
    w_reach: float = -10.0
    reach_x: float = 0.6
    reach_y: float = 0.3
    w_still: float = -1.0  # standing commands: per foot out of contact
    # the base env's joint-deviation L1 regularizers (fpm/env_cfg.py RewardsCfg)
    w_dev_waist: float = -1.0
    w_dev_hip: float = -1.0
    w_dev_arms: float = -0.1
    # tracking on v_xy and w_z, with weak lin_vel_z and ang_vel_xy penalties as in the base env
    w_lin_z: float = -0.5
    w_ang_xy: float = -0.05
    cmd_threshold: float = 0.05
    walking_threshold: float = 0.05
    running_threshold: float = 1.5
    term_angle: float = math.radians(70.0)
    term_height: float = 0.3
    term_hole: float = 0.3  # >0: terminate when the pelvis is less than this above the highest nearby terrain


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(S) -> torch.Tensor:
    """Heading/position-invariant features: tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1), (Rt @ w.unsqueeze(-1)).squeeze(-1),
         q, qd), -1)  # fmt: skip


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """Dynamics-relevant context only; the encoder bias (xi[4:33]) only affects observations."""
    e = xi[..., 4 + NUM_J :]
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:4] / 0.03, e[..., 0:1] / 0.15 - 1, e[..., 1:2] / 0.075 - 1,
                      (e[..., 2:3] - 1.1) / 0.4, e[..., 3:4] / 2 - 1, (e[..., 4:5] - 1) / 2), -1)  # fmt: skip


XI_N_DIM = 9


class MDP:
    def __init__(self, kin: G1Kinematics, terrain: TerrainFeatures, default_q: torch.Tensor, joint_names: list[str],
                 soft_lo: torch.Tensor, soft_hi: torch.Tensor, action_scale: torch.Tensor,
                 cfg: RewardCfg = RewardCfg()):
        self.kin, self.terrain, self.cfg = kin, terrain, cfg
        self.default_q = default_q
        self.action_scale = action_scale
        self.soft_lo, self.soft_hi = soft_lo, soft_hi
        dev = default_q.device
        self.jmap = JointMap(joint_names, dev)
        import re

        def mask(pat):
            return torch.tensor([1.0 if re.fullmatch(pat, n) else 0.0 for n in joint_names], device=dev)

        self.m_waist = mask(r"waist.*")
        self.m_hip = mask(r".*_hip_(roll|yaw)_joint")
        self.m_arms = mask(r".*_(shoulder_.*|elbow|wrist_.*)_joint")
        self.q0_isaac = self.jmap.to_isaac(default_q)
        self.std = {k: torch.tensor(match_values(POSE_STD[k], joint_names), device=dev)
                    for k in ("std_standing", "std_walking", "std_running")}
        self.critic_dim = INV_DIM + terrain.wm_dim + 2 + 6 + 3 + XI_N_DIM + terrain.critic_dim

    def context(self, S) -> dict:
        p, R, v, w, q, qd = S
        fp, fv, rot, bpos = self.kin.forward(p, R, v, w, q, qd)
        tf, clear, rough, unsup = self.terrain.wm_features(p, R, fp, rot[..., 1:3, :, :])
        return {"fp": fp, "fv": fv, "rot": rot, "tf": tf, "clear": clear, "rough": rough, "unsup": unsup, "hb": tf[..., 0],
                "torso_p": bpos[..., 0, :], "torso_R": rot[..., 0, :, :], "hole": self.terrain.hole_margin(p, R)}

    def wm_terrain(self, S) -> torch.Tensor:
        """Only the world model's terrain input ``context(S)["tf"]``: skips the eager-only hole margin and the rest."""
        p, R, v, w, q, qd = S
        fp, fv, rot, bpos = self.kin.forward(p, R, v, w, q, qd)
        return self.terrain.wm_features(p, R, fp, rot[..., 1:3, :, :])[0]

    def proprio(self, S, a_prev_mj, cmd, xi, noise: bool = True) -> torch.Tensor:
        """Policy proprioception (99): sim encoder bias xi[4:33] corrupts the joint positions."""
        return policy_proprio(S, self.jmap.to_isaac(a_prev_mj), cmd, self.q0_isaac, self.jmap,
                              bias_mj=xi[..., 4 : 4 + NUM_J], noise=noise)

    def system_state(self, S, cmd) -> torch.Tensor:
        return system_state(S, cmd, self.q0_isaac, self.jmap)

    def joint_target(self, a_mj: torch.Tensor) -> torch.Tensor:
        """Joint position setpoint, the action-convention-free world-model input."""
        return self.default_q + self.action_scale * a_mj

    def reward(self, S, K, c, a, a_prev, cmd, dt: float, terms: bool = False):
        """Per-step reward (scaled by dt). ``c``: foot contact probabilities (..., 2)."""
        cf = self.cfg
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
        fv, rot = K["fv"], K["rot"]
        speed = cmd[..., :2].norm(dim=-1) + cmd[..., 2].abs()
        active = (speed > cf.cmd_threshold).float()

        e_lin = ((cmd[..., :2] - v_b[..., :2]) ** 2).sum(-1)
        r_lin = torch.exp(-e_lin / cf.std_lin**2)
        e_ang = (cmd[..., 2] - w_b[..., 2]) ** 2
        r_ang = torch.exp(-e_ang / cf.std_ang**2)
        r_extra = cf.w_lin_z * v_b[..., 2] ** 2 + cf.w_ang_xy * (w_b[..., :2] ** 2).sum(-1)
        g_torso = -rot[..., 0, 2, :]
        r_up = torch.exp(-(g_torso[..., :2] ** 2).sum(-1) / cf.std_upright**2)
        stand = (speed < cf.walking_threshold).float().unsqueeze(-1)
        run = (speed >= cf.running_threshold).float().unsqueeze(-1)
        std = self.std["std_standing"] * stand + self.std["std_running"] * run + self.std["std_walking"] * (1 - stand - run)
        r_pose = torch.exp(-(((q - self.default_q) / std) ** 2).mean(-1)) * stand.squeeze(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r_rate = ((a - a_prev) ** 2).sum(-1)
        vxy = fv[..., :2].norm(dim=-1)
        dc = torch.relu(cf.clearance_target - K["clear"])
        r_clear = (dc * vxy).sum(-1) * active
        r_slip = ((fv[..., :2] ** 2).sum(-1) * c).sum(-1) * active
        r_bav = (w[..., :2] ** 2).sum(-1)
        cl, cr = c[..., 0], c[..., 1]
        walking = (speed > 0.2).float()
        r_gait = walking * (cl * (1 - cr) + cr * (1 - cl) - 2 * (1 - cl) * (1 - cr))
        # sole-support term (also stair edges) + smooth support field (gaps, wide gradient band)
        u = K["unsup"].sum(-1)
        r_foothold = (c * u).sum(-1)
        r_still = stand.squeeze(-1) * (1 - c).sum(-1)
        r_reach = K["reach"] if "reach" in K else torch.zeros_like(r_still)
        dev_abs = (q - self.default_q).abs()
        r_dev = (cf.w_dev_waist * (dev_abs * self.m_waist).sum(-1) + cf.w_dev_hip * (dev_abs * self.m_hip).sum(-1)
                 + cf.w_dev_arms * (dev_abs * self.m_arms).sum(-1))
        r = (cf.w_lin * r_lin + cf.w_ang * r_ang + cf.w_upright * r_up + cf.w_pose * r_pose + cf.w_limits * r_lim
             + cf.w_action_rate * r_rate + cf.w_clearance * r_clear + cf.w_slip * r_slip + cf.w_body_ang_vel * r_bav
             + cf.w_gait * r_gait + cf.w_foothold * r_foothold + cf.w_still * r_still + cf.w_reach * r_reach + r_dev + r_extra) * dt  # fmt: skip
        if terms:
            return r, {"lin": r_lin, "ang": r_ang, "upright": r_up, "pose": r_pose, "limits": r_lim, "rate": r_rate,
                       "clear": r_clear, "slip": r_slip, "gait": r_gait, "foothold": r_foothold, "dev": r_dev, "still": r_still, "reach": r_reach}  # fmt: skip
        return r

    def termination(self, S, K) -> torch.Tensor:
        R = S[1]
        out = (R[..., 2, 2] < math.cos(self.cfg.term_angle)) | (K["hb"] < self.cfg.term_height)
        if self.cfg.term_hole > 0:
            out = out | (K["hole"] < self.cfg.term_hole)
        return out

    def soft_termination(self, S, K, tz: float = 0.02, tt: float = 0.03) -> torch.Tensor:
        R = S[1]
        alive = torch.sigmoid((R[..., 2, 2] - math.cos(self.cfg.term_angle)) / tt) * torch.sigmoid(
            (K["hb"] - self.cfg.term_height) / tz)
        if self.cfg.term_hole > 0:
            alive = alive * torch.sigmoid((K["hole"] - self.cfg.term_hole) / tz)
        return 1 - alive

    def critic_features(self, S, K, c, cmd, xi_n) -> torch.Tensor:
        p, R = S[0], S[1]
        H = yaw_mat(R)
        feet = (H.transpose(-1, -2).unsqueeze(-3) @ (K["fp"] - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1).flatten(-2)
        scan = self.terrain.critic_scan(p, R)
        return torch.cat((invariant_features(S), K["tf"], c, feet, cmd, xi_n, scan), -1)

"""Observation and action conventions of the fpm G1 env, so exported policies play there.

* joint order: Isaac Lab breadth-first (``ISAAC_JOINTS``); action: ``q_target = q0 + 0.25 a``;
* policy observation (201): base_lin_vel, 0.2 base_ang_vel, projected_gravity, command, joint_pos - q0,
  0.05 joint_vel, last_action, foothold_scan (2 x (2 + 7 x 7)); noise before scaling;
* FPM input ``system_state`` (70) -> next footholds (x, y) of both feet in the pelvis frame;
* foothold_scan: 7 x 7 nodes around each foothold on the torso's yaw-aligned 1.6 x 1.0 m grid (0.05 m),
  ``clip(torso_z - terrain_z - 0.5, -1, 1)``.
"""

from __future__ import annotations

import torch

from .rotations import yaw_mat

ISAAC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip

DEFAULT_POSE = {  # the base env's default joint positions (q0)
    ".*_hip_pitch_joint": -0.18,
    ".*_knee_joint": 0.40,
    ".*_ankle_pitch_joint": -0.22,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
ACTION_SCALE = 0.25
ACTION_CLIP = 10.0

PROPRIO_DIM = 3 + 3 + 3 + 3 + 29 + 29 + 29
SYSTEM_STATE_DIM = 70
PATCH_N = 7
SCAN_RES = 0.05
SCAN_SIZE = (1.6, 1.0)
SCAN_OFFSET = 0.5
FOOTHOLD_SCAN_DIM = 2 * (2 + PATCH_N**2)
# full height scan: the base env's height_scan grid (1.6 x 1.0 m at 0.05 m, y outer, x inner)
FULL_SCAN_NX, FULL_SCAN_NY = int(round(SCAN_SIZE[0] / SCAN_RES)) + 1, int(round(SCAN_SIZE[1] / SCAN_RES)) + 1
FULL_SCAN_DIM = FULL_SCAN_NX * FULL_SCAN_NY
POLICY_OBS_DIM = PROPRIO_DIM + FOOTHOLD_SCAN_DIM
BOOTSTRAP_FOOTHOLDS = ((0.0, 0.1), (0.0, -0.1))

# Isaac observation noise (uniform half-widths, applied before scaling) and scales
LIN_VEL_NOISE, ANG_VEL_NOISE, GRAV_NOISE, QPOS_NOISE, QVEL_NOISE, SCAN_NOISE = 0.15, 0.2, 0.05, 0.01, 1.5, 0.1
ANG_VEL_SCALE, QVEL_SCALE = 0.2, 0.05
XY_DRIFT, Z_DRIFT = 0.05, 0.05


class JointMap:
    """Permutations between the simulator's (MJCF) joint order and Isaac Lab's order."""

    def __init__(self, mj_joints: list[str], device: str):
        assert sorted(mj_joints) == sorted(ISAAC_JOINTS)
        self.m2i = torch.tensor([mj_joints.index(n) for n in ISAAC_JOINTS], device=device)  # x_isaac = x_mj[m2i]
        self.i2m = torch.tensor([ISAAC_JOINTS.index(n) for n in mj_joints], device=device)  # x_mj = x_isaac[i2m]

    def to_isaac(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.m2i]

    def to_mj(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.i2m]


def _u(x: torch.Tensor, h: float) -> torch.Tensor:
    return (2 * torch.rand_like(x) - 1) * h


def base_quantities(S):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
    w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
    g_b = -R[..., 2, :]
    return v_b, w_b, g_b


def policy_proprio(S, a_prev_isaac, cmd, q0_isaac, jmap: JointMap, bias_mj=None, noise: bool = True) -> torch.Tensor:
    """First 99 entries of the base env's policy observation (Isaac order). ``bias_mj``: encoder bias (sim DR)."""
    q, qd = S[4], S[5]
    v_b, w_b, g_b = base_quantities(S)
    q_rel = jmap.to_isaac(q if bias_mj is None else q + bias_mj) - q0_isaac
    qd_i = jmap.to_isaac(qd)
    if noise:
        spike = (torch.rand_like(v_b[..., :1]) < 1e-4).float()
        v_b = v_b + _u(v_b, LIN_VEL_NOISE) + spike * _u(v_b, 5.0)
        w_b = w_b + _u(w_b, ANG_VEL_NOISE) + (torch.rand_like(w_b[..., :1]) < 1e-4).float() * _u(w_b, 10.0)
        g_b = g_b + _u(g_b, GRAV_NOISE)
        q_rel = q_rel + _u(q_rel, QPOS_NOISE)
        qd_i = qd_i + _u(qd_i, QVEL_NOISE)
    return torch.cat((v_b, ANG_VEL_SCALE * w_b, g_b, cmd, q_rel, QVEL_SCALE * qd_i,
                      a_prev_isaac.clamp(-ACTION_CLIP, ACTION_CLIP)), -1)  # fmt: skip


def system_state(S, cmd, q0_isaac, jmap: JointMap) -> torch.Tensor:
    """FPM input (70): base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel."""
    v_b, w_b, g_b = base_quantities(S)
    return torch.cat((v_b, w_b, g_b, cmd, jmap.to_isaac(S[4]) - q0_isaac, jmap.to_isaac(S[5])), -1)


class FootholdScan:
    """The base env's ``foothold_scan`` term on top of a global heightmap (stand-in for the torso-mounted ray caster)."""

    def __init__(self, hm, device: str):
        self.hm = hm
        self.nx = int(SCAN_SIZE[0] / SCAN_RES) + 1
        self.ny = int(SCAN_SIZE[1] / SCAN_RES) + 1
        k = torch.arange(-(PATCH_N // 2), PATCH_N // 2 + 1, device=device)
        ox, oy = torch.meshgrid(k, k, indexing="ij")
        self.off_x, self.off_y = ox.flatten(), oy.flatten()

    def __call__(self, p, R, torso_p, torso_R, footholds_root, xy_drift=None, z_drift=None, noise: bool = True,
                 return_points: bool = False):
        """footholds_root: (..., 2, 2) in the pelvis frame. Returns (..., 102)."""
        fb = torch.cat((footholds_root, torch.zeros_like(footholds_root[..., :1])), -1)
        feet_w = p.unsqueeze(-2) + (R.unsqueeze(-3) @ fb.unsqueeze(-1)).squeeze(-1)
        rel = feet_w - torso_p.unsqueeze(-2)
        feet_s = (torso_R.transpose(-1, -2).unsqueeze(-3) @ rel.unsqueeze(-1)).squeeze(-1)  # full sensor rotation
        if xy_drift is not None:
            feet_s = torch.cat((feet_s[..., :2] + xy_drift.unsqueeze(-2), feet_s[..., 2:]), -1)
        ax = torch.round((feet_s[..., 0] + SCAN_SIZE[0] / 2) / SCAN_RES).long()
        ay = torch.round((feet_s[..., 1] + SCAN_SIZE[1] / 2) / SCAN_RES).long()
        ix = (ax.unsqueeze(-1) + self.off_x).clamp(0, self.nx - 1)
        iy = (ay.unsqueeze(-1) + self.off_y).clamp(0, self.ny - 1)
        gx = ix.float() * SCAN_RES - SCAN_SIZE[0] / 2
        gy = iy.float() * SCAN_RES - SCAN_SIZE[1] / 2
        Y = yaw_mat(torso_R)[..., :2, :2].unsqueeze(-3).unsqueeze(-3)  # yaw-aligned grid
        pts = torso_p[..., None, None, :2] + (Y @ torch.stack((gx, gy), -1).unsqueeze(-1)).squeeze(-1)
        hz = self.hm.query_nearest(pts)
        h = torso_p[..., 2, None, None] - hz - SCAN_OFFSET
        if z_drift is not None:  # drift before noise and clipping, as in the base env
            h = h + z_drift[..., None, None]
        out = torch.cat((footholds_root, h), -1)  # (..., 2, 2 + 49)
        if noise:
            out = out + _u(out, SCAN_NOISE)
        out = out.clamp(-1.0, 1.0).flatten(-2)
        if return_points:
            return out, pts, hz
        return out


class FullHeightScan:
    """The whole torso-mounted, yaw-aligned height grid (the base env's ``height_scan`` term) instead of FPM-gated
    foothold patches: clip(torso_z - terrain_z - 0.5 + noise, -1, 1) on 33 x 21 nodes, y outer / x inner."""

    def __init__(self, hm, device: str):
        self.hm = hm
        x = torch.arange(FULL_SCAN_NX, device=device).float() * SCAN_RES - SCAN_SIZE[0] / 2
        y = torch.arange(FULL_SCAN_NY, device=device).float() * SCAN_RES - SCAN_SIZE[1] / 2
        gx, gy = torch.meshgrid(x, y, indexing="xy")  # (ny, nx): Isaac Lab's "xy" ordering
        self.grid = torch.stack((gx.flatten(), gy.flatten()), -1)

    def __call__(self, p, R, torso_p, torso_R, footholds_root=None, xy_drift=None, z_drift=None, noise: bool = True,
                 return_points: bool = False):
        g = self.grid
        if xy_drift is not None:  # sensor-frame shift, as the foothold scan's drift
            g = g + xy_drift.unsqueeze(-2)
        Y = yaw_mat(torso_R)[..., :2, :2].unsqueeze(-3)
        pts = torso_p[..., None, :2] + (Y @ g.unsqueeze(-1)).squeeze(-1)
        hz = self.hm.query_nearest(pts)
        h = torso_p[..., 2, None] - hz - SCAN_OFFSET
        if z_drift is not None:
            h = h + z_drift[..., None]
        if noise:
            h = h + _u(h, SCAN_NOISE)
        out = h.clamp(-1.0, 1.0)
        if return_points:
            return out, pts, hz
        return out


def mirror_full_scan(e: torch.Tensor) -> torch.Tensor:
    """Mirror the full height scan left/right: flip the grid along y."""
    return e.reshape(*e.shape[:-1], FULL_SCAN_NY, FULL_SCAN_NX).flip(-2).reshape(*e.shape[:-1], FULL_SCAN_DIM)


# Joint permutation and signs from the base env's fpm/mdp/symmetry.py (Isaac joint order).
_JOINT_PAIRS = [(0, 1, 1.0), (3, 4, -1.0), (6, 7, -1.0), (9, 10, 1.0), (11, 12, 1.0), (13, 14, 1.0), (15, 16, -1.0),
                (17, 18, -1.0), (19, 20, -1.0), (21, 22, 1.0), (23, 24, -1.0), (25, 26, 1.0), (27, 28, -1.0)]
_JOINT_SINGLES = [(2, -1.0), (5, -1.0), (8, 1.0)]
JOINT_PERM = list(range(29))
JOINT_SIGN = [1.0] * 29
for _a, _b, _s in _JOINT_PAIRS:
    JOINT_PERM[_a], JOINT_PERM[_b] = _b, _a
    JOINT_SIGN[_a] = JOINT_SIGN[_b] = _s
for _i, _s in _JOINT_SINGLES:
    JOINT_SIGN[_i] = _s


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
    return x[..., JOINT_PERM] * torch.tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)


def mirror_proprio(o: torch.Tensor) -> torch.Tensor:
    """Mirror the 99-dim proprioceptive policy observation (the base env's vector signs)."""
    s3 = lambda x, s: x * torch.tensor(s, device=x.device, dtype=x.dtype)  # noqa: E731
    return torch.cat((s3(o[..., 0:3], (1.0, -1.0, 1.0)), s3(o[..., 3:6], (-1.0, 1.0, -1.0)), s3(o[..., 6:9], (1.0, -1.0, 1.0)),
                      s3(o[..., 9:12], (1.0, -1.0, -1.0)), mirror_joints(o[..., 12:41]), mirror_joints(o[..., 41:70]),
                      mirror_joints(o[..., 70:99])), -1)  # fmt: skip


def mirror_scan(e: torch.Tensor) -> torch.Tensor:
    """Mirror the foothold scan: swap feet, negate foothold y, flip each 7x7 patch along y (the base env's convention)."""
    k = PATCH_N
    stride = 2 + k * k
    feet = [e[..., :stride], e[..., stride:]]
    out = []
    for src in reversed(feet):
        patch = src[..., 2:].reshape(*src.shape[:-1], k, k).flip(-1).reshape(*src.shape[:-1], k * k)
        out.append(torch.cat((src[..., 0:1], -src[..., 1:2], patch), -1))
    return torch.cat(out, -1)

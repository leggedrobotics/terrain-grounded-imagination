"""Differentiable forward kinematics of the G1 from the MuJoCo model (foot sites and selected bodies only)."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import quat_to_mat, skew


class G1Kinematics:
    def __init__(self, model: mujoco.MjModel, joint_names: list[str], device: str,
                 site_names=("left_foot", "right_foot"), body_names=("torso_link", "left_ankle_roll_link", "right_ankle_roll_link")):
        self.device = device
        m = model
        jid = {}
        for i in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            jid[nm] = jid[nm.split("/")[-1]] = i
        # joint index (in our action/state ordering) driving each body
        self.body_joint = {}
        for k, name in enumerate(joint_names):
            j = jid[name]
            assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            assert np.allclose(m.jnt_pos[j], 0.0), name
            self.body_joint[int(m.jnt_bodyid[j])] = (k, torch.tensor(m.jnt_axis[j], dtype=torch.float32, device=device))
        root = int(m.jnt_bodyid[[i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE][0]])
        self.root = root

        def bid(n):
            for i in range(m.nbody):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                if nm == n or (nm and nm.endswith("/" + n)):
                    return i
            raise KeyError(n)

        def sid(n):
            for i in range(m.nsite):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
                if nm == n or (nm and nm.endswith("/" + n)):
                    return i
            raise KeyError(n)

        targets = []  # (body id, local offset)
        for s in site_names:
            i = sid(s)
            targets.append((int(m.site_bodyid[i]), torch.tensor(m.site_pos[i], dtype=torch.float32, device=device)))
        for b in body_names:
            targets.append((bid(b), torch.zeros(3, device=device)))
        self.n_sites = len(site_names)
        self.targets = targets
        # chain of bodies root(excluded) -> target, for each target
        self.chains = []
        for b, _ in targets:
            chain = []
            while b != root:
                chain.append(b)
                b = int(m.body_parentid[b])
            self.chains.append(chain[::-1])
        self.body_pos = {b: torch.tensor(m.body_pos[b], dtype=torch.float32, device=device) for c in self.chains for b in c}
        self.body_R = {
            b: quat_to_mat(torch.tensor(m.body_quat[b], dtype=torch.float32, device=device)) for c in self.chains for b in c
        }
        lo, hi = [], []
        for name in joint_names:
            j = jid[name]
            lo.append(m.jnt_range[j][0]), hi.append(m.jnt_range[j][1])
        self.q_lo = torch.tensor(lo, device=device)
        self.q_hi = torch.tensor(hi, device=device)

    @staticmethod
    def _hinge(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        K = skew(axis)
        s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        return eye + s * K + (1 - c) * (K @ K)

    def forward(self, p, R, v, w, q, qd):
        """Returns site positions/velocities (..., n_sites, 3), body rotations (..., n_bodies, 3, 3), body positions."""
        cache = {}
        pos_out, vel_out, rot_out, bpos_out = [], [], [], []
        for t, ((b_t, off), chain) in enumerate(zip(self.targets, self.chains)):
            Rb, pb = R, p
            axes_w, origins = [], []
            joints = []
            for b in chain:
                if b in cache:
                    Rb, pb, axes_w, origins, joints = cache[b]
                    continue
                pb = pb + (Rb @ self.body_pos[b].unsqueeze(-1)).squeeze(-1)
                Rb = Rb @ self.body_R[b]
                axes_w, origins, joints = list(axes_w), list(origins), list(joints)
                if b in self.body_joint:
                    k, ax = self.body_joint[b]
                    axes_w.append((Rb @ ax.unsqueeze(-1)).squeeze(-1))
                    origins.append(pb)
                    joints.append(k)
                    Rb = Rb @ self._hinge(ax, q[..., k])
                cache[b] = (Rb, pb, axes_w, origins, joints)
            x = pb + (Rb @ off.unsqueeze(-1)).squeeze(-1)
            if t < self.n_sites:
                vel = v + torch.cross(w, x - p, dim=-1)
                for a, o, k in zip(axes_w, origins, joints):
                    vel = vel + qd[..., k : k + 1] * torch.cross(a, x - o, dim=-1)
                pos_out.append(x)
                vel_out.append(vel)
            else:
                rot_out.append(Rb)
                bpos_out.append(x)
        return torch.stack(pos_out, -2), torch.stack(vel_out, -2), torch.stack(rot_out, -3), torch.stack(bpos_out, -2)

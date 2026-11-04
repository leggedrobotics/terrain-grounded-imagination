"""NumPy/ONNX controller with the base G1 env's interfaces (exported policy + FPM), used by sim2sim and deployment.

Inputs per control step: base pose and velocity, IMU gyro, joint states (Isaac order), command, and a height function
``height_at(x, y) -> z`` (None: flat ground).
"""

from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np
import onnxruntime as ort

from ..constants import UNITREE_G1

PATCH_N, RES, SIZE, OFFSET = 7, 0.05, (1.6, 1.0), 0.5
NX, NY = int(SIZE[0] / RES) + 1, int(SIZE[1] / RES) + 1


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


class Controller:
    def __init__(self, run_dir: str, height_fn=None, noise_rng=None):
        ex = os.path.join(run_dir, "exported")
        self.cfg = json.load(open(os.path.join(ex, "deploy.json")))
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        self.pi = ort.InferenceSession(os.path.join(ex, "policy.onnx"), opt, providers=["CPUExecutionProvider"])
        self.fpm = ort.InferenceSession(os.path.join(ex, "foot_model.onnx"), opt, providers=["CPUExecutionProvider"])
        self.names = self.cfg["joint_names"]
        self.q0 = np.array(self.cfg["default_q"])
        self.scale = float(self.cfg["action_scale"])
        self.kp, self.kd = np.array(self.cfg["kp"]), np.array(self.cfg["kd"])
        self.eff = np.array(self.cfg["effort_limit"])
        self.height_fn = height_fn
        self.rng = noise_rng  # sensor noise as in training (None = clean)
        self.a_prev = np.zeros(29)
        self.vis = None
        # kinematic model for the torso pose
        self.km = mujoco.MjModel.from_xml_path(UNITREE_G1)
        self.kd_ = mujoco.MjData(self.km)
        self.kq = [self.km.jnt_qposadr[self.km.joint(n).id] for n in self.names]
        self.torso = self.km.body("torso_link").id
        k = np.arange(-(PATCH_N // 2), PATCH_N // 2 + 1)
        ox, oy = np.meshgrid(k, k, indexing="ij")
        self.off = np.stack((ox.ravel(), oy.ravel()), -1)

    def _u(self, shape, h):
        return self.rng.uniform(-h, h, shape) if self.rng is not None else 0.0

    def torso_pose(self, base_pos, quat, q):
        self.kd_.qpos[:3], self.kd_.qpos[3:7] = base_pos, quat
        self.kd_.qpos[self.kq] = q
        mujoco.mj_kinematics(self.km, self.kd_)
        return self.kd_.xpos[self.torso].copy(), self.kd_.xmat[self.torso].reshape(3, 3).copy()

    def foothold_scan(self, base_pos, R, torso_p, torso_R, fh):
        yaw = math.atan2(torso_R[1, 0], torso_R[0, 0])
        Y = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        out, pts_all, hz_all = [], [], []
        for f in range(2):
            fw = base_pos + R @ np.array([fh[f, 0], fh[f, 1], 0.0])
            fs = torso_R.T @ (fw - torso_p)
            ax = int(round((fs[0] + SIZE[0] / 2) / RES))
            ay = int(round((fs[1] + SIZE[1] / 2) / RES))
            ix = np.clip(ax + self.off[:, 0], 0, NX - 1)
            iy = np.clip(ay + self.off[:, 1], 0, NY - 1)
            g = np.stack((ix * RES - SIZE[0] / 2, iy * RES - SIZE[1] / 2), -1)
            pts = torso_p[:2] + g @ Y.T
            if self.height_fn is None:
                hz = np.full(len(pts), base_pos[2] - 0.78)  # flat ground under the robot
            else:
                hz = np.array([self.height_fn(x, y) for x, y in pts])
            h = torso_p[2] - hz - OFFSET
            o = np.concatenate((fh[f], h))
            o = np.clip(o + self._u(o.shape, 0.1), -1, 1)
            out.append(o), pts_all.append(pts), hz_all.append(hz)
        self.vis = {"pts": np.array(pts_all), "hz": np.array(hz_all)}
        return np.concatenate(out)

    def step(self, base_pos, quat, v_b, gyro, q, dq, cmd) -> np.ndarray:
        """All joint quantities in Isaac order. Returns joint position targets (Isaac order)."""
        R = quat_to_mat(quat)
        g_b = R.T @ np.array([0, 0, -1.0])
        state = np.concatenate((v_b, gyro, g_b, cmd, q - self.q0, dq)).astype(np.float32)
        fh = self.fpm.run(None, {"obs": state[None, None]})[0].reshape(2, 2)
        self.last_fh = fh
        torso_p, torso_R = self.torso_pose(base_pos, quat, q)
        scan = self.foothold_scan(base_pos, R, torso_p, torso_R, fh)
        prop = np.concatenate((v_b + self._u(3, 0.15), 0.2 * (gyro + self._u(3, 0.2)), g_b + self._u(3, 0.05), cmd,
                               q - self.q0 + self._u(29, 0.01), 0.05 * (dq + self._u(29, 1.5)),
                               np.clip(self.a_prev, -10, 10)))  # fmt: skip
        obs = np.concatenate((prop, scan)).astype(np.float32)[None]
        a = self.pi.run(None, {"obs": obs})[0][0].astype(np.float64)
        self.a_prev = a
        return self.q0 + self.scale * a

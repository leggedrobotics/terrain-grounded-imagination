"""Stand-still check in sim2sim: zero command for 8 s, from spawn and after 2 s of walking.

Counts steps and the share of time a foot is in the air; a policy that stands still scores 0 on both.

    python scripts/stand_check.py --export_dir logs/<run>/export_<it>
"""

import argparse

import mujoco
import numpy as np

from sim2sim import SimController, build_model

SETTLE, STAND = 1.0, 8.0


def run(export_dir: str, seed: int, walk_s: float):
    rng = np.random.default_rng(seed)
    m = build_model("flat", seed)
    d = mujoco.MjData(m)
    ctl = SimController(export_dir, m, d, rng, True)
    d.qpos[2] = 0.80
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[ctl.qadr] = ctl.q0
    mujoco.mj_forward(m, d)
    dt = ctl.cfg["control_dt"]
    n_sub, lat = int(round(dt / m.opt.timestep)), int(round(0.02 / m.opt.timestep))
    feet = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    target, pending, k = ctl.q0.copy(), [], 0
    z0, up_prev, steps, air = None, np.zeros(2, bool), 0, 0
    for t in range(int((walk_s + STAND) / dt)):
        c = np.array([0.5, 0.0, 0.0]) if t * dt < walk_s else np.zeros(3)
        pending.append((k + lat, ctl.act_(m, d, c, rng, True)))
        for _ in range(n_sub):
            while pending and pending[0][0] <= k:
                target = pending.pop(0)[1]
            q, qd = d.qpos[ctl.qadr], d.qvel[ctl.vadr]
            d.ctrl[ctl.act] = np.clip(ctl.kp * (target - q) - ctl.kd * qd, -ctl.eff, ctl.eff)
            mujoco.mj_step(m, d)
            k += 1
        if t * dt < walk_s + SETTLE:
            continue
        z = d.xpos[feet][:, 2]
        z0 = z.copy() if z0 is None else z0
        up = z - z0 > 0.02
        steps += int((up & ~up_prev).sum())
        air += int(up.any())
        up_prev = up
    n = int((STAND - SETTLE) / dt)
    return steps / (STAND - SETTLE), air / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    r = np.array([run(a.export_dir, s, w) for s in range(a.seeds) for w in (0.0, 2.0)])
    print(f"  stand-still  steps/s {r[:, 0].mean():.2f}  foot-in-air share {r[:, 1].mean():.2f}")


if __name__ == "__main__":
    main()

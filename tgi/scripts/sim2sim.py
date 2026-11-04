"""Sim-to-sim courses for an exported controller in plain MuJoCo with Unitree's official G1 model.

Own 500 Hz PD loop, actuation latency, ONNX inference and ray-cast perception. Courses (``--terrains``): flat,
``stairs:rise``, ``blocks:max_height``, ``stones:gap[:size]``, ``fpmstones:difficulty``, ``drop:h``, ``pit:h``,
``noise:h[:cell]``, ``slope``. A course is crossed if the robot is upright and past it.

    python scripts/sim2sim.py --export_dir logs/<run>/export_100000 --terrains stairs:0.20 fpmstones:0.6 --repeats 2
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool

import mujoco
import numpy as np

from tgi.constants import UNITREE_SCENE as SCENE
from tgi.deploy.controller import Controller
from tgi.env.terrains import stone_rects

TERRAIN_GROUP = 2


def build_model(terrain: str, seed: int = 0) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(SCENE)
    for g in spec.worldbody.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.group = TERRAIN_GROUP
    rng = np.random.default_rng(seed)

    def box(x, y, z, sx, sy, sz, shade=0.0):
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [x, y, z]
        g.size = [sx, sy, sz]
        g.group = TERRAIN_GROUP
        g.rgba = [0.55 - shade, 0.48 - shade, 0.42 - shade, 1]

    if terrain.startswith("stairs"):  # 'stairs[:rise]': 8 steps up (35 cm run) onto a long plateau
        h = float(terrain.split(":")[1]) if ":" in terrain else 0.10
        w, x0, x_far = 0.35, 1.5, 40.0
        for k in range(8):
            xk = x0 + k * w
            box((xk + x_far) / 2, 0, (k + 1) * h / 2, (x_far - xk) / 2, 5.0, (k + 1) * h / 2, 0.15 * (k % 2))
    elif terrain.startswith("blocks"):  # 'blocks[:max_height]': random stepping blocks, heights U[0.2, 1] * max
        hmax = float(terrain.split(":")[1]) if ":" in terrain else 0.10
        for _ in range(120):
            x, y = rng.uniform(0.6, 8.0), rng.uniform(-2.0, 2.0)
            hz = rng.uniform(0.2 * hmax, hmax)
            box(x, y, hz / 2, rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4), hz / 2, 0.25 * hz / hmax)
    elif terrain.startswith("stones"):  # 'stones:gap': platform, 0.35 m stones over a 0.5 m deep pit, platform
        parts = terrain.split(":")  # 'stones:gap[:size]'
        gap = float(parts[1]) if len(parts) > 1 else 0.15
        top, size = 0.5, float(parts[2]) if len(parts) > 2 else 0.35
        box(-8.5, 0, top / 2, 10.0, 5.0, top / 2)  # start platform x < 1.5
        x = 1.5 + gap + size / 2
        while x < 6.0:
            for y in np.arange(-1.5, 1.51, size + gap):
                dz = rng.uniform(-0.03, 0.03)
                box(x + rng.uniform(-0.03, 0.03), y, (top + dz) / 2, size / 2, size / 2, (top + dz) / 2, 0.1)
            x += size + gap
        box(x - size / 2 + gap + 10.0, 0, top / 2, 10.0, 5.0, top / 2)  # landing platform
    elif terrain.startswith("fpmstones"):  # 'fpmstones:difficulty': the base env's training stones (0.1 m pixels) over a 0.5 m pit
        d = float(terrain.split(":")[1]) if ":" in terrain else 0.5
        top = 0.5
        box(-8.5, 0, top / 2, 10.0, 5.0, top / 2)  # start platform x < 1.5
        for x0, x1, y0, y1, dz in stone_rects(d, (4.5, 4.0), rng, res=0.1):
            box(1.5 + (x0 + x1) / 2, -2.0 + (y0 + y1) / 2, (top + dz) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (top + dz) / 2, 0.1)
        box(6.0 + 10.0, 0, top / 2, 10.0, 5.0, top / 2)  # landing platform
    elif terrain.startswith("noise"):  # 'noise:h[:cell]': uniform height noise U[0, h] in 2 cm steps on cell x cell
        parts = terrain.split(":")  # tiles for x in [1, 8]
        h, cell = float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.1
        for x in np.arange(1.0, 8.0, cell):
            for y in np.arange(-1.0, 1.0, cell):
                hz = max(round(rng.uniform(0, h) / 0.02) * 0.02, 0.005)
                box(x + cell / 2, y + cell / 2, hz / 2, cell / 2, cell / 2, hz / 2, 0.2 * hz / max(h, 1e-6))
    elif terrain.startswith("drop"):  # 'drop:h': start on a raised platform (x < 1.5), step down to the ground
        h = float(terrain.split(":")[1]) if ":" in terrain else 0.15
        box(-8.5, 0, h / 2, 10.0, 5.0, h / 2)
    elif terrain.startswith("pit"):  # 'pit:h': start in a 2 x 2 m pit h deep, step up out of it and walk on
        h = float(terrain.split(":")[1]) if ":" in terrain else 0.20
        box(11.0, 0, h / 2, 10.0, 11.0, h / 2)  # x > 1
        box(-11.0, 0, h / 2, 10.0, 11.0, h / 2)  # x < -1
        box(0, 6.0, h / 2, 1.0, 5.0, h / 2)  # |x| < 1, y > 1
        box(0, -6.0, h / 2, 1.0, 5.0, h / 2)
    elif terrain == "slope":  # 10 deg ramp from x=1 to x=5, then a plateau
        ang, L = math.radians(10), 4.0
        top = L * math.tan(ang)
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        half = L / (2 * math.cos(ang))
        g.size = [half, 5.0, 0.05]
        g.pos = [1.0 + L / 2 + 0.05 * math.sin(ang), 0, top / 2 - 0.05 * math.cos(ang)]
        g.quat = [math.cos(-ang / 2), 0, math.sin(-ang / 2), 0]
        g.group = TERRAIN_GROUP
        box(5.0 + 20.0, 0, top / 2, 20.0, 5.0, top / 2)
    m = spec.compile()
    m.opt.timestep = 0.002
    return m


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


class SimController:
    """Wraps the deployable Controller: reads the MuJoCo state like the robot's estimator would."""

    def __init__(self, run_dir: str, m: mujoco.MjModel, d: mujoco.MjData, rng, noise: bool):
        gg = np.zeros(6, np.uint8)
        gg[TERRAIN_GROUP] = 1
        gid = np.zeros(1, np.int32)

        def height_at(x, y):
            dist = mujoco.mj_ray(m, d, np.array([x, y, 10.0]), np.array([0, 0, -1.0]), gg, 1, -1, gid)
            return 10.0 - dist if dist >= 0 else -1.0

        self.c = Controller(run_dir, height_at, rng if noise else None)
        self.cfg = self.c.cfg
        names = self.c.names
        self.qadr = [m.jnt_qposadr[m.joint(j).id] for j in names]
        self.vadr = [m.jnt_dofadr[m.joint(j).id] for j in names]
        self.act = [[a for a in range(m.nu) if m.actuator_trnid[a, 0] == m.joint(j).id][0] for j in names]
        self.kp, self.kd, self.eff, self.q0 = self.c.kp, self.c.kd, self.c.eff, self.c.q0

    @property
    def vis(self):
        return self.c.vis

    def act_(self, m, d, cmd, rng, noise):
        R = d.xmat[1].reshape(3, 3)
        return self.c.step(d.qpos[0:3].copy(), d.qpos[3:7].copy(), R.T @ d.qvel[0:3], d.qvel[3:6].copy(),
                           d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), np.asarray(cmd, dtype=np.float64))


def _cmap(t):
    """t in [0, 1] -> rgb in [0, 1] (viridis)."""
    import matplotlib

    return np.array(matplotlib.colormaps["viridis"](float(np.clip(t, 0, 1)))[:3])


def draw_perception(scn, vis, rng_h: float = 0.25):
    """3D overlay: FPM footholds (large spheres) and the 7x7 height samples the policy reads."""
    if vis is None:
        return
    for f in range(2):
        pts, hz = vis["pts"][f], vis["hz"][f]
        hc = hz[len(hz) // 2]
        for (x, y), h in zip(pts, hz):
            if scn.ngeom >= scn.maxgeom:
                return
            c = _cmap((h - hc + rng_h) / (2 * rng_h))  # 3D markers: height relative to the foothold
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.011, 0, 0]),
                                np.array([x, y, h + 0.008]), np.eye(3).flatten(), np.append(c, 0.95).astype(np.float32))
            scn.ngeom += 1
        x, y = pts[len(pts) // 2]
        col = np.array([0.1, 0.9, 0.2, 0.9] if f == 0 else [0.95, 0.2, 0.9, 0.9], np.float32)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.04, 0.004, 0]),
                            np.array([x, y, hc + 0.012]), np.eye(3).flatten(), col)
        scn.ngeom += 1


def run_episode(job):
    export_dir, terrain, cmd, seconds, latency, push, seed, noise = job[:8]
    video = job[8] if len(job) > 8 else None
    frames, renderer, cam = [], None, None
    if video:
        renderer = None  # created after the model below
    rng = np.random.default_rng(seed)
    m = build_model(terrain, seed)
    d = mujoco.MjData(m)
    ctl = SimController(export_dir, m, d, rng, noise)
    lift = 0.5 if "stones" in terrain else (float(terrain.split(":")[1]) if terrain.startswith("drop:") else 0.15 if terrain == "drop" else 0.0)
    d.qpos[2] = 0.80 + lift
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[ctl.qadr] = ctl.q0
    mujoco.mj_forward(m, d)
    n_sub = int(round(ctl.cfg["control_dt"] / m.opt.timestep))
    lat_sub = int(round(latency / m.opt.timestep))
    target = ctl.q0.copy()
    pending = []  # (apply_at_substep, target)
    errs, wz_errs, alive, k = [], [], True, 0
    waist_yaw, torso_rel_yaw = [], []
    wy_adr = m.jnt_qposadr[m.joint("waist_yaw_joint").id]
    torso_id = m.body("torso_link").id
    steps = int(seconds / ctl.cfg["control_dt"])
    x0 = d.qpos[0]
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = 1
        cam.distance, cam.elevation, cam.azimuth = 3.2, -22.0, 125.0
    heading_hold = terrain != "flat"  # terrain courses: operator keeps the heading (mjlab heading command, k=0.5)
    for t in range(steps):
        c = np.array(cmd, dtype=np.float64)
        if heading_hold:
            yaw = yaw_of(d.xmat[1].reshape(3, 3))
            c[2] = float(np.clip(0.5 * math.remainder(0.0 - yaw, 2 * math.pi), -1.0, 1.0))
        pending.append((k + lat_sub, ctl.act_(m, d, c, rng, noise)))
        if push and t > 0 and t % 150 == 0:
            d.qvel[0:2] += rng.uniform(-0.5, 0.5, 2)
        for _ in range(n_sub):
            while pending and pending[0][0] <= k:
                target = pending.pop(0)[1]
            q, qd = d.qpos[ctl.qadr], d.qvel[ctl.vadr]
            tau = np.clip(ctl.kp * (target - q) - ctl.kd * qd, -ctl.eff, ctl.eff)
            d.ctrl[ctl.act] = tau
            mujoco.mj_step(m, d)
            k += 1
        if video and t % 2 == 0:
            renderer.update_scene(d, cam)
            vis = getattr(ctl, "vis", None)
            draw_perception(renderer.scene, vis)
            frames.append(renderer.render())
        R = d.xmat[1].reshape(3, 3)
        if R[2, 2] < math.cos(math.radians(70)) or d.qpos[2] < 0.35 + (0 if terrain == "flat" else -10) or (
                "stones" in terrain and d.qpos[2] < 0.75):  # fell into the pit
            alive = False
            break
        if t > 50:
            vb = R.T @ d.qvel[0:3]
            errs.append(np.linalg.norm(vb[:2] - cmd[:2]))
            waist_yaw.append(d.qpos[wy_adr])
            torso_rel_yaw.append(math.remainder(yaw_of(d.xmat[torso_id].reshape(3, 3)) - yaw_of(R), 2 * math.pi))
            wz_errs.append(abs(d.qvel[5] - c[2]))
    if video:
        import mediapy

        mediapy.write_video(video, frames, fps=int(round(1 / (2 * ctl.cfg["control_dt"]))))
    return {"terrain": terrain, "cmd": list(cmd), "alive": alive, "t": t * ctl.cfg["control_dt"],
            "lin_err": float(np.mean(errs)) if errs else float("nan"), "yaw_err": float(np.mean(wz_errs)) if wz_errs else float("nan"),
            "dist_x": float(d.qpos[0] - x0), "y": float(d.qpos[1]), "z": float(d.qpos[2]),
            "waist_yaw_mean": float(np.mean(waist_yaw)) if waist_yaw else float("nan"),
            "torso_rel_yaw_mean": float(np.mean(torso_rel_yaw)) if torso_rel_yaw else float("nan")}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True, help="scripts/export.py output dir (contains exported/)")
    ap.add_argument("--terrains", nargs="+", default=["flat"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--latency", type=float, default=0.02)
    ap.add_argument("--push", type=int, default=0)
    ap.add_argument("--noise", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1, help="repeat each course with new seeds")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None, help="render one episode (first terrain, cmd --video_cmd) to this mp4")
    ap.add_argument("--video_cmd", type=float, nargs=3, default=[0.6, 0.0, 0.0])
    ap.add_argument("--video_seed", type=int, default=0)
    a = ap.parse_args()
    if a.video:
        print(run_episode((a.export_dir, a.terrains[0], tuple(a.video_cmd), a.seconds, a.latency, a.push, a.video_seed, bool(a.noise), a.video)))
        return
    fwd = [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2
    cmds = {
        "flat": [(vx, vy, wz) for vx in (-0.5, 0.0, 0.5, 1.0) for vy in (0.0, 0.4) for wz in (0.0, 0.5)],
        "stairs": [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2,
        "stairs:0.05": [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2,
        "stairs:0.075": [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2,
        "blocks": [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2,
        "slope": [(vx, 0.0, 0.0) for vx in (0.4, 0.6, 0.8, 1.0)] * 2,
    }
    jobs = []
    for terr in a.terrains:
        for r in range(a.repeats):
            for i, c in enumerate(cmds.get(terr, fwd)):
                jobs.append((a.export_dir, terr, c, a.seconds, a.latency, a.push, 1000 * r + i, bool(a.noise)))
    with Pool(a.workers) as pool:
        res = pool.map(run_episode, jobs)
    def crossed(x):
        t = x["terrain"]
        h = float(t.split(":")[1]) if ":" in t else 0.10
        if t.startswith("stairs"):
            return x["alive"] and x["z"] > 8 * h + 0.5  # on the plateau
        if t.startswith("blocks"):
            return x["alive"] and x["dist_x"] > 8.5  # past the block field
        if t == "slope":
            return x["alive"] and x["dist_x"] > 5.5
        if "stones" in t:
            return x["alive"] and x["dist_x"] > 6.5 and x["z"] > 1.0  # reached the landing platform
        if t.startswith("drop"):
            return x["alive"] and x["dist_x"] > 3.0  # stepped down and walked on
        if t.startswith("pit"):
            return x["alive"] and x["dist_x"] > 3.0  # stepped out (rim at x = 1) and walked on
        if t.startswith("noise"):
            return x["alive"] and x["dist_x"] > 7.5  # past the noise field
        return x["alive"]

    summary = {}
    for terr in a.terrains:
        r = [x for x in res if x["terrain"] == terr]
        summary[terr] = {"episodes": len(r), "survival": float(np.mean([x["alive"] for x in r])),
                         "crossed": float(np.mean([crossed(x) for x in r])),
                         "waist_yaw_mean": float(np.nanmean([x["waist_yaw_mean"] for x in r])),
                         "abs_waist_yaw_mean": float(np.nanmean([abs(x["waist_yaw_mean"]) for x in r])),
                         "lin_err": float(np.nanmean([x["lin_err"] for x in r])),
                         "yaw_err": float(np.nanmean([x["yaw_err"] for x in r])),
                         "mean_dist_x": float(np.mean([x["dist_x"] for x in r]))}  # fmt: skip
    for x in res:
        if not x["alive"]:
            print("FAIL", x)
    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump({"summary": summary, "episodes": res, "args": vars(a)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Batched G1 simulator for world-model transitions: the base G1 env on Isaac Lab 3 with Newton physics.

MJCF joint order and wxyz quaternions throughout. Differences to the base env: stepping stones from
``tgi/env/terrains.py``, the success-guided curriculum (``tgi/env/sgs.py``), our own domain randomization (reported in
``xi``) and pushes, and the pre-reset state of finished episodes, so ``s1`` is never a reset state.
"""

from __future__ import annotations

import math

import torch

from ..constants import G1_MJCF, MAX_LAG, NUM_J
from .env_cfg import make_isaac_env_cfg

_APP = None


def launch_app(headless: bool = True):
    """Isaac Lab needs its app (Kit) before any isaaclab.* simulation import."""
    global _APP
    if _APP is None:
        from isaaclab.app import AppLauncher

        _APP = AppLauncher(headless=headless).app
    return _APP


def _xyzw_to_wxyz(q):
    return torch.cat((q[..., 3:4], q[..., :3]), -1)


class IsaacLocoSim:
    def __init__(self, num_envs: int, device: str, seed: int = 0, stones_share: float = 0.6, push_interval_s=(1.0, 3.0),
                 push_vel=None, play: bool = False, cfg=None, sgs_spawns: int = 4, sgs_dirs: int = 4, sgs_window: int = 16,
                 sgs_neighbor: float = 0.25, sgs_target: float = 0.8, pit_step: float = 0.05,
                 straight_frac: float = 0.0):
        launch_app()
        import mujoco
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.terrains import TerrainImporter

        cfg = cfg or make_isaac_env_cfg(num_envs, seed, stones_share, pit_step=pit_step)
        if sgs_spawns > 1:  # spawn patches tile the reset event's square instead of widening it
            n = int(round(sgs_spawns ** 0.5))
            assert n * n == sgs_spawns, sgs_spawns
            pr = cfg.events.reset_base.params["pose_range"]
            pr["x"], pr["y"] = (pr["x"][0] / n, pr["x"][1] / n), (pr["y"][0] / n, pr["y"][1] / n)
        self.cfg = cfg
        self.device, self.N, self.play = device, num_envs, play

        # keep the terrain mesh for the heightmap
        meshes = {}
        orig_import = TerrainImporter.import_mesh

        def keep(imp, name, mesh):
            meshes[name] = mesh
            return orig_import(imp, name, mesh)

        TerrainImporter.import_mesh = keep
        sim_self = self

        class _Env(ManagerBasedRLEnv):
            def _reset_idx(self, env_ids):
                sim_self._before_reset(env_ids)
                super()._reset_idx(env_ids)
                sim_self._after_reset(env_ids)

        self._snap = None
        self.extra_termination = None
        # zero-yaw episodes: no heading control, ang_vel_z = 0
        self.straight_frac = straight_frac
        self.straight = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.env = _Env(cfg)
        TerrainImporter.import_mesh = orig_import
        self.env._loco_sim = self
        self.terrain_mesh = meshes["terrain"]
        self.robot = self.env.scene["robot"]
        self.contact = self.env.scene["contact_forces"]
        self.dt = self.env.step_dt

        # joint order: MJCF in tgi, breadth-first in Isaac Lab
        self.kin_model = mujoco.MjModel.from_xml_path(G1_MJCF)
        self.joint_names = [self.kin_model.joint(i).name for i in range(self.kin_model.njnt)
                            if self.kin_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        isaac_names = list(self.robot.joint_names)
        assert sorted(isaac_names) == sorted(self.joint_names), isaac_names  # Newton orders depth-first, PhysX breadth-first
        self.i_of_m = torch.tensor([isaac_names.index(n) for n in self.joint_names], device=device)  # mj <- isaac
        self.m_of_i = torch.tensor([self.joint_names.index(n) for n in isaac_names], device=device)  # isaac <- mj
        d = self.robot.data
        self.default_q = d.default_joint_pos.torch[0, self.i_of_m].clone()
        lim = d.soft_joint_pos_limits.torch[0, self.i_of_m]
        self.soft_limits = (lim[:, 0].clone(), lim[:, 1].clone())
        self.action_scale = torch.full((NUM_J,), 0.25, device=device)
        self.feet = self.contact.find_bodies(".*_ankle_roll_link")[0]
        self.torso = self.robot.find_bodies("torso_link")[0][0]
        self.act_term = self.env.action_manager.get_term("joint_pos")

        self._randomize()
        self.push_interval = push_interval_s
        self.push_vel = push_vel if push_vel is not None else {
            "x": 0.5, "y": 0.5, "z": 0.4, "roll": 0.52, "pitch": 0.52, "yaw": 0.78}
        self.push_on = not play
        self.next_push = self._sample_push_time(num_envs)
        self.t = torch.zeros(num_envs, device=device)
        self.a_prev = torch.zeros(num_envs, NUM_J, device=device)
        self.ep_ret = torch.zeros(num_envs, device=device)
        self.ep_len = torch.zeros(num_envs, device=device)

        # success-guided curriculum over terrain cells (mjlab's promotion criterion)
        gen = cfg.scene.terrain.terrain_generator
        self.terrain_size = gen.size[0]
        # terrain types per column, as Isaac Lab's curriculum generator
        prop = torch.tensor([float(t.proportion) for t in gen.sub_terrains.values()], device=device, dtype=torch.float32)
        cum = torch.cumsum(prop / prop.sum(), 0)
        self.col_type = torch.tensor([int((c / gen.num_cols + 0.001 < cum).nonzero()[0]) for c in range(gen.num_cols)],
                                     device=device)
        self.cmd_dist = torch.zeros(num_envs, device=device)
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        # configurations: terrain cell x spawn patch x travel direction
        self.sgs_spawns, self.sgs_dirs = sgs_spawns, sgs_dirs
        self.cfg_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        from .sgs import SuccessGuidedSampler

        imp = self.env.scene.terrain
        rows, cols = imp.terrain_origins.shape[:2]
        k = sgs_spawns * sgs_dirs
        cell_col = torch.arange(rows * cols * k, device=device) // k % cols
        ids = torch.arange(rows * cols * k, device=device)
        prev = torch.where(ids >= cols * k, ids - cols * k, torch.full_like(ids, -1))  # config ids: (row, col, sub)
        self._sgs = SuccessGuidedSampler(rows * cols * k, device, group=self.col_type[cell_col],
                                         group_prior=prop / prop.sum(), window=sgs_window, prev=prev,
                                         neighbor=sgs_neighbor, target=sgs_target)
        self._first = True
        self.env.reset()
        self._first = False
        self.start_xy[:] = self.robot.data.root_link_pose_w.torch[:, :2]

    def sgs_stats(self) -> dict[str, float]:
        """Summary of the success-guided curriculum sampler."""
        return self._sgs.stats()

    def sgs_profile(self) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Per terrain type, by level: (success rate, sampling share within the type, outcomes recorded per config)."""
        imp = self.env.scene.terrain
        rows, cols = imp.terrain_origins.shape[:2]
        k = self.sgs_spawns * self.sgs_dirs
        P, p = self._sgs.probs().view(rows, cols, k), self._sgs.success_rate().view(rows, cols, k)
        n = self._sgs.count.view(rows, cols, k).float()
        out = {}
        for t in self.col_type.unique().tolist():
            m = self.col_type == t
            Pt = P[:, m].sum((1, 2))
            out[t] = (p[:, m].mean((1, 2)), Pt / Pt.sum().clamp(min=1e-12), n[:, m].mean((1, 2)))
        return out

    def _randomize(self):
        N, dev, r = self.N, self.device, self.robot

        def u(lo, hi, *shape):
            return torch.empty(*shape, device=dev).uniform_(lo, hi)

        # foot friction (one value for all robot shapes of an env; Newton has a single coefficient)
        from isaaclab_newton.envs.mdp import events as nev  # noqa: F401  (binding helper lives with the backend)
        import warp as wp

        model = self.env.sim.physics_manager.get_model()
        mu = wp.to_torch(r._root_view.get_attribute("shape_material_mu", model)[:, 0])
        self.fr = u(0.3, 1.2, N)
        mu[:] = self.fr.unsqueeze(-1)
        # torso com offset and added mass
        com = r.data.body_com_pose_b.torch.clone()
        self.com = torch.stack((u(-0.025, 0.025, N), u(-0.025, 0.025, N), u(-0.03, 0.03, N)), -1)
        com[:, self.torso, :3] += self.com
        r.set_coms_index(coms=com[:, [self.torso]], body_ids=[self.torso])
        mass = r.data.body_mass.torch.clone()
        self.dmass = u(-1.0, 3.0, N)
        mass[:, self.torso] += self.dmass
        r.set_masses_index(masses=mass[:, [self.torso]], body_ids=[self.torso])
        # joints: friction (abs), armature (scale), shared over joints per env
        self.jf = u(0.0, 0.3, N)
        r.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=self.jf.unsqueeze(-1).expand(N, NUM_J).contiguous())
        self.arm = u(0.7, 1.5, N)
        arm0 = r.data.joint_armature.torch.clone()
        r.write_joint_armature_to_sim_index(armature=arm0 * self.arm.unsqueeze(-1))
        # encoder bias (observation only, MJCF order) and actuation lag (physics steps)
        self.bias = u(-0.015, 0.015, N, NUM_J)
        self.lag = torch.randint(0, MAX_LAG + 1, (N,), device=dev)
        self.act_term.lag[:] = self.lag

    def xi(self) -> torch.Tensor:
        z = torch.zeros(self.N, 1, device=self.device)
        return torch.cat((self.fr.unsqueeze(-1), self.com, self.bias, self.jf.unsqueeze(-1), z, self.arm.unsqueeze(-1),
                          self.lag.float().unsqueeze(-1), self.dmass.unsqueeze(-1)), -1)

    def flat_state(self) -> torch.Tensor:
        d = self.robot.data
        pose = d.root_link_pose_w.torch
        return torch.cat((pose[:, :3], _xyzw_to_wxyz(pose[:, 3:7]), d.root_link_vel_w.torch,
                          d.joint_pos.torch[:, self.i_of_m], d.joint_vel.torch[:, self.i_of_m]), -1).clone()

    def state(self):
        from ..mdp.mdp import unflatten_state

        return unflatten_state(self.flat_state())

    def command(self) -> torch.Tensor:
        return self.env.command_manager.get_command("base_velocity").clone()

    def foot_contact(self) -> torch.Tensor:
        f = self.contact.data.net_forces_w.torch[:, self.feet]
        return (f.norm(dim=-1) > 1.0).float()

    def _before_reset(self, env_ids):
        if isinstance(env_ids, slice) or env_ids is None:  # Isaac Lab 3 passes slices for "all"
            env_ids = range(self.N)[env_ids or slice(None)]
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not self._first:
            if self._snap is None:
                imp = self.env.scene.terrain
                self._snap = (self.flat_state(), self.foot_contact(), imp.terrain_levels.clone(), imp.terrain_types.clone())
            self._sgs.record(self.cfg_id[ids], self._success(ids))
        imp = self.env.scene.terrain
        cols = imp.terrain_origins.shape[1]
        k = self.sgs_spawns * self.sgs_dirs
        self.cfg_id[ids] = self._sgs.sample(len(ids))
        c, sub = self.cfg_id[ids] // k, self.cfg_id[ids] % k
        imp.terrain_levels[ids] = c // cols
        imp.terrain_types[ids] = c % cols
        imp.env_origins[ids] = imp.terrain_origins[imp.terrain_levels[ids], imp.terrain_types[ids]]
        if self.sgs_spawns > 1:  # spawn patch: a cell of an n x n grid over the reset event's (+-0.5 m) square
            n = int(round(self.sgs_spawns ** 0.5))
            sp = sub // self.sgs_dirs
            off = torch.stack([sp // n, sp % n], -1).float()
            pr = self.cfg.events.reset_base.params["pose_range"]
            span = torch.tensor([pr["x"][1] - pr["x"][0], pr["y"][1] - pr["y"][0]], device=self.device) * n
            imp.env_origins[ids, :2] += ((off + 0.5) / n - 0.5) * span  # reset_base adds U(+-span / 2n)

    def _after_reset(self, env_ids):
        """Heading target for the sampled travel direction; zero-yaw episodes spawn facing it, without heading control."""
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = range(self.N)[env_ids or slice(None)]
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        cmd = self.env.command_manager.get_term("base_velocity")
        if self.straight_frac > 0:
            self.straight[ids] = torch.rand(len(ids), device=self.device) < self.straight_frac
        if self.sgs_dirs <= 1:
            self._enforce_straight(cmd)
            return
        b = (self.cfg_id[ids] % self.sgs_dirs).float()
        w = 2 * math.pi / self.sgs_dirs
        travel = (b + torch.rand_like(b)) * w - math.pi
        v = cmd.vel_command_b[ids]
        h = travel - torch.atan2(v[:, 1], v[:, 0])
        cmd.heading_target[ids] = torch.atan2(torch.sin(h), torch.cos(h))
        sids = ids[self.straight[ids]]
        if len(sids) > 0:  # face the travel direction from the start (yaw-only spawn orientation, xyzw)
            pose = self.robot.data.root_link_pose_w.torch[sids].clone()
            hy = cmd.heading_target[sids] * 0.5
            pose[:, 3:7] = torch.stack((torch.zeros_like(hy), torch.zeros_like(hy), torch.sin(hy), torch.cos(hy)), -1)
            self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=sids)
        self._enforce_straight(cmd)

    def _enforce_straight(self, cmd=None):
        if self.straight_frac <= 0:
            return
        cmd = cmd or self.env.command_manager.get_term("base_velocity")
        m = self.straight
        cmd.is_heading_env[m] = False
        cmd.vel_command_b[m, 2] = 0.0
    def _success(self, ids):
        e = self.env
        xy = self.robot.data.root_link_pose_w.torch[ids, :2]
        dist = (xy - self.start_xy[ids]).norm(dim=-1)
        time_left = (e.max_episode_length - e.episode_length_buf[ids]).clamp(min=0) * self.dt
        commanded = self.cmd_dist[ids] + self.command()[ids, :2].norm(dim=-1) * time_left
        walked = dist > torch.clamp(0.8 * commanded, max=self.terrain_size / 2)
        survived = e.episode_length_buf[ids] >= e.max_episode_length - 1
        return torch.where(commanded >= 1.0, walked, survived)

    def _sample_push_time(self, n):
        lo, hi = self.push_interval
        return torch.empty(n, device=self.device).uniform_(lo, hi)

    def pre_step(self):
        if not self.push_on:
            return torch.zeros(self.N, dtype=torch.bool, device=self.device)
        push = self.t >= self.next_push
        ids = push.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            vel = self.robot.data.root_link_vel_w.torch[ids].clone()
            r = self.push_vel
            span = torch.tensor([r["x"], r["y"], r["z"], r["roll"], r["pitch"], r["yaw"]], device=self.device)
            vel += (2 * torch.rand_like(vel) - 1) * span
            self.robot.write_root_link_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
            self.t[ids] = 0.0
            self.next_push[ids] = self._sample_push_time(len(ids))
        return push

    def step(self, a: torch.Tensor) -> dict:
        """a: actions in MJCF joint order (q_target = q0 + 0.25 a)."""
        s = self.flat_state()
        cmd = self.command()
        xi = self.xi()
        a_prev = self.a_prev.clone()
        self.cmd_dist += cmd[:, :2].norm(dim=-1) * self.dt
        self._snap = None
        _, rew, term, timeout, extras = self.env.step(a[:, self.m_of_i])
        self._enforce_straight()  # (the 10 s command resample redraws heading envs and ang_vel_z)
        s1, c1 = self.flat_state(), self.foot_contact()
        done = term | timeout
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:  # finished episodes: their pre-reset state, captured inside the reset
            s1[ids], c1[ids] = self._snap[0][ids], self._snap[1][ids]
        out = {"s": s, "a": a.clone(), "a_prev": a_prev, "s1": s1, "c1": c1, "cmd": cmd, "xi": xi,
               "r_sim": rew.clone(), "term": term.clone(), "done": done.clone(), "ep_step": self.ep_len.clone()}
        self.ep_ret += rew
        self.ep_len += 1
        self.t += self.dt
        self.a_prev = a.clone()
        if len(ids) > 0:
            out["ep_len"] = self.ep_len[ids].clone()
            out["ep_ret"] = self.ep_ret[ids].clone()
            out["ep_level"], out["ep_type"] = self._snap[2][ids], self.col_type[self._snap[3][ids]]
            out["ep_ids"] = ids
            out["episode_log"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in extras.get("log", {}).items()
                                  if k.startswith("Episode_Reward/") or k.startswith("Episode_Termination/")}
            self.a_prev[ids] = 0.0
            self.ep_ret[ids] = 0.0
            self.ep_len[ids] = 0.0
            self.t[ids] = 0.0
            self.cmd_dist[ids] = 0.0
            self.start_xy[ids] = self.robot.data.root_link_pose_w.torch[ids, :2]
        out["log_step"] = {}
        self.pre_step()
        return out

    def heightmap(self, device: str, res: float = 0.04):
        from ..mdp.heightmap import HeightMap

        return HeightMap.from_trimesh(self.terrain_mesh, device, res=res)

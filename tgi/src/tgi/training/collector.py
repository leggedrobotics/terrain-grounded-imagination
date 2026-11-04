"""Simulator collection with the deployable controller (FPM footholds -> foothold scan -> policy).

With ``--async_collect`` the simulator runs in a thread on the default CUDA stream while the updates run on their own
stream; it acts with copies of the policy and FPM that are refreshed every iteration.
"""

from __future__ import annotations

import copy
import threading

import torch

from ..mdp import normalize_xi, unflatten_state
from ..policy_io import SimRunner


class Collector:
    def __init__(self, sim, mdp, pi, fpm, wm, buffer, logger, action_noise: float, threaded: bool):
        self.sim, self.mdp, self.pi, self.wm, self.buffer, self.logger = sim, mdp, pi, wm, buffer, logger
        self.action_noise = action_noise
        self.threaded = threaded
        self.steps = 0  # simulator transitions collected (environments x steps)
        if threaded:
            pi_act = copy.deepcopy(pi).requires_grad_(False)
            fpm_act = copy.deepcopy(fpm).requires_grad_(False) if fpm is not None else None
            self._copies = [(d, s) for a, b in ((pi_act, pi), (fpm_act, fpm)) if a is not None
                            for d, s in zip(a.state_dict().values(), b.state_dict().values())]
            self.sim_stream = torch.cuda.default_stream()
            self.train_stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            torch.cuda.set_stream(self.train_stream)
            self.runner = SimRunner(pi_act, fpm_act, mdp, sim, noise=True)
        else:
            self.train_stream = None
            self.runner = SimRunner(pi, fpm, mdp, sim, noise=True)

    @torch.no_grad()
    def _simulate(self, n_steps: int, use_fpm: bool):
        self.runner.use_fpm_model = use_fpm
        records = []
        for _ in range(n_steps):
            o, extra = self.runner.observe()
            a = self.mdp.jmap.to_mj(self.runner.pi(o.unsqueeze(1), extra))
            a = torch.clamp(a + self.action_noise * torch.randn_like(a), -10, 10)
            records.append((o, extra, self.sim.step(a)))
        return records, self.sim.env.scene.terrain.terrain_levels.float().mean()

    def _simulate_job(self, n_steps: int, use_fpm: bool, job: dict):
        with torch.cuda.stream(self.sim_stream):
            job["result"] = self._simulate(n_steps, use_fpm)
            job["done"] = torch.cuda.Event()
            job["done"].record(self.sim_stream)

    def start(self, n_steps: int, use_fpm: bool, overlap: bool = False) -> dict:
        """Simulate n_steps; with overlap (threaded mode only), in a thread that ``finish`` joins."""
        job = {}
        if not self.threaded:
            job["result"] = self._simulate(n_steps, use_fpm)
            return job
        with torch.no_grad():  # refresh the simulator's copies of the networks on the update stream
            torch._foreach_copy_([d for d, _ in self._copies], [s for _, s in self._copies])
        self.sim_stream.wait_stream(self.train_stream)
        if overlap:
            job["thread"] = threading.Thread(target=self._simulate_job, args=(n_steps, use_fpm, job), daemon=True)
            job["thread"].start()
        else:
            self._simulate_job(n_steps, use_fpm, job)
        return job

    @torch.no_grad()
    def finish(self, job: dict):
        if "thread" in job:
            job["thread"].join()
        if "result" not in job:
            raise RuntimeError("the simulator thread failed (see its traceback above)")
        if self.threaded:
            self.train_stream.wait_event(job["done"])
        records, level = job["result"]
        mdp, wm = self.mdp, self.wm
        for o, extra, out in records:
            self.pi.obs_norm.update(o)
            self.pi.extra_norm.update(extra)
            S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
            self.buffer.add(out)
            wm.in_norm.update(wm.inputs(S0, mdp.wm_terrain(S0), mdp.joint_target(out["a"]), mdp.joint_target(out["a_prev"]),
                                        normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            self.steps += self.sim.N
            self.logger.add_transition(out, S1, self.sim.dt)
        self.logger.add_level(level)

    def collect(self, n_steps: int, use_fpm: bool):
        self.finish(self.start(n_steps, use_fpm))

"""Terrain Grounded Imagination training loop.

Each iteration collects simulator steps, fits the world model (terrain looked up from the heightmap at every step)
and the FPM, then backpropagates the return of ``horizon`` imagined steps through the world model into the policy,
with a critic bootstrap and the gradient cut at imagined touchdowns on unsupported terrain.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

from ..env.sim import IsaacLocoSim
from ..foot_pred import FootholdPredictor, FPMTrainer
from ..mdp import RewardCfg, normalize_xi, unflatten_state
from ..mdp.interface import XY_DRIFT, Z_DRIFT, mirror_full_scan, mirror_joints, mirror_proprio, mirror_scan
from ..policy_io import bootstrap_footholds, build_mdp, make_policy, perceive, set_scan_mode
from ..world_model import Critic, ReplayBuffer, WorldModel, networks
from .collector import Collector
from .config import REPO
from .logger import TrainLogger
from .optim import MuonAdam, cosine_lr, load_optimizer_state, make_optimizer, set_lr, skip_if_nonfinite

# floor of the per-dimension error scale: position, rotation, velocity, angular velocity, q, qd
ERR_FLOOR = (3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3 + (3e-3,) * 29 + (5e-2,) * 29


class Profiler:
    """Per-phase wall time in ms per iteration (``TGI_PROF=1``); syncs only the update stream."""

    def __init__(self, stream: torch.cuda.Stream | None):
        self.enabled = os.environ.get("TGI_PROF") == "1"
        self.stream, self.acc, self.t = stream, {}, time.perf_counter()

    def tick(self, name: str):
        if not self.enabled:
            return
        self.stream.synchronize() if self.stream is not None else torch.cuda.synchronize()
        t = time.perf_counter()
        self.acc[name] = self.acc.get(name, 0.0) + t - self.t
        self.t = t

    def report(self, iters: int):
        if self.enabled:
            total = sum(self.acc.values())
            print("   prof ms/it: " + " ".join(f"{k}={1e3 * v / iters:.1f}" for k, v in self.acc.items())
                  + f" total={1e3 * total / iters:.1f}", flush=True)
            self.acc.clear()


def requires_grad(*modules: nn.Module | None, flag: bool):
    for m in modules:
        if m is not None:
            m.requires_grad_(flag)


class Trainer:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.dev = dev = "cuda:0"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        if cfg.amp == "bf16" and torch.cuda.get_device_capability()[0] < 8:
            print("no native bf16 on this GPU (compiled bf16 graphs would fall back to eager): --amp none", flush=True)
            cfg.amp = "none"
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        set_scan_mode(cfg.scan)
        self.mirror_extra = mirror_full_scan if cfg.scan == "full" else mirror_scan
        os.makedirs(cfg.log_dir, exist_ok=True)
        json.dump(vars(cfg), open(os.path.join(cfg.log_dir, "args.json"), "w"), indent=2)

        # simulator first; torch gets the rest of --vram_gb
        free0, total = torch.cuda.mem_get_info()
        self.sim = sim = IsaacLocoSim(cfg.num_envs, dev, seed=cfg.seed, stones_share=cfg.stepping_stones,
                                      sgs_spawns=cfg.sgs_spawns, sgs_dirs=cfg.sgs_dirs, sgs_window=cfg.sgs_window,
                                      sgs_neighbor=cfg.sgs_neighbor, sgs_target=cfg.sgs_target, pit_step=cfg.pit_step,
                                      straight_frac=cfg.straight_frac)
        sim_gb = (free0 - torch.cuda.mem_get_info()[0]) / 1e9
        torch_gb = cfg.vram_gb - sim_gb
        assert torch_gb > 1.0, f"--vram_gb {cfg.vram_gb} is too small: the simulator alone uses {sim_gb:.2f} GB"
        torch.cuda.set_per_process_memory_fraction(torch_gb * 1e9 / total)
        print(f"VRAM budget {cfg.vram_gb:.1f} GB: simulator {sim_gb:.2f} GB, torch cap {torch_gb:.2f} GB", flush=True)

        reward_cfg = RewardCfg(w_gait=cfg.w_gait, w_foothold=cfg.w_foothold, w_dev_waist=cfg.w_dev_waist,
                               w_dev_hip=cfg.w_dev_hip, w_dev_arms=cfg.w_dev_arms, w_upright=cfg.w_upright, w_pose=cfg.w_pose,
                               w_still=cfg.w_still, w_reach=cfg.w_reach, reach_x=cfg.reach_x, reach_y=cfg.reach_y,
                               term_hole=cfg.term_hole, term_angle=cfg.term_angle)  # fmt: skip
        self.mdp = mdp = build_mdp(sim, dev, reward_cfg, field_window=cfg.field_window, field_hole=cfg.field_hole,
                                   hm_res=cfg.hm_res, hm_exact=bool(cfg.hm_exact))
        if cfg.grad_blur > 0:
            mdp.terrain.hm.set_grad_blur(cfg.grad_blur)
            print(f"heightmap gradients from a {cfg.grad_blur * 100:.1f} cm Gaussian-blurred map", flush=True)
        if cfg.term_hole > 0:  # the same gap termination in the simulator as in imagination
            sim.extra_termination = lambda S: mdp.terrain.hole_margin(S[0], S[1]) < cfg.term_hole

        self.wm = WorldModel(mdp.terrain.wm_dim, tuple(cfg.wm_hidden)).to(dev)
        self.pi = make_policy(dev)  # base env observation layout, Isaac joint order
        self.critic = Critic(mdp.critic_dim).to(dev)
        self.fpm = FootholdPredictor().to(dev) if cfg.fpm else None
        self.fpm_trainer = FPMTrainer(self.fpm) if cfg.fpm else None
        self.opt_wm = make_optimizer(cfg.wm_opt, self.wm, cfg.lr, muon_lr_mult=cfg.muon_lr_mult)
        self.opt_pi = make_optimizer(cfg.pi_opt, self.pi, cfg.pi_lr, betas=(0.9, 0.95), muon_lr_mult=cfg.muon_lr_mult)
        self.opt_v = make_optimizer("adam", self.critic, cfg.lr)
        for name, opt in (("world model", self.opt_wm), ("policy", self.opt_pi)):
            if isinstance(opt, MuonAdam):
                print(f"{name} optimizer: {opt.describe()}", flush=True)

        self.start_it, self.wall_offset, sim_steps = 0, 0.0, 0
        if cfg.resume:
            self.start_it, self.wall_offset, sim_steps = self._load(cfg.resume)
        self.t_start = time.time() - self.wall_offset
        self.err_floor = torch.tensor(ERR_FLOOR, device=dev)
        self._compile()

        self.buffer = ReplayBuffer(cfg.buffer_steps, cfg.num_envs, dev)
        names = list(sim.cfg.scene.terrain.terrain_generator.sub_terrains.keys())
        self.logger = TrainLogger(cfg.log_dir, resume=self.start_it > 0, log_every=cfg.log_every, terrain_names=names,
                                  device=dev)
        self.collector = Collector(sim, mdp, self.pi, self.fpm, self.wm, self.buffer, self.logger, cfg.action_noise,
                                   threaded=bool(cfg.async_collect))
        self.collector.steps = sim_steps
        self.prof = Profiler(self.collector.train_stream)
        # overlap only once the CUDA graphs are recorded
        self.overlap_from = max(cfg.async_after, cfg.wm_only_iters + 50, cfg.fpm_bootstrap + 50,
                                self.start_it + cfg.resume_wm_only + 50)
        if cfg.async_collect:
            print(f"async collection: the simulator overlaps the updates from iteration "
                  f"{max(self.overlap_from, self.start_it)}", flush=True)
        self.last_sym_loss = torch.zeros((), device=dev)
        self.last_event_share = torch.zeros((), device=dev)

    def _load(self, path: str) -> tuple[int, float, int]:
        ck = torch.load(path, map_location=self.dev)
        self.wm.load_state_dict(ck["wm"])
        self.pi.load_state_dict(ck["pi"])
        self.critic.load_state_dict(ck["critic"])
        if self.fpm is not None and ck.get("fpm") is not None:
            self.fpm.load_state_dict(ck["fpm"])
        for key, opt in (("opt_wm", self.opt_wm), ("opt_pi", self.opt_pi), ("opt_v", self.opt_v)):
            if key in ck and not load_optimizer_state(opt, ck[key], self.dev):
                print(f"resume: {key} optimizer type changed, starting it fresh", flush=True)
        return ck["iteration"], ck["wall_time"], ck["sim_steps"]

    def save(self, it: int):
        ck = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
              "fpm": self.fpm.state_dict() if self.fpm is not None else None, "fpm_active": self.fpm_active(it),
              "interface": "base_g1_env", "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(),
              "opt_v": self.opt_v.state_dict(), "iteration": it, "wall_time": time.time() - self.t_start,
              "sim_steps": self.collector.steps, "args": vars(self.cfg), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(self.cfg.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        keep = ("pi", "wm", "critic", "fpm", "fpm_active", "interface", "iteration", "wall_time", "sim_steps", "args",
                "saved_at")  # the policy checkpoint, without the optimizer states
        torch.save({k: ck[k] for k in keep}, os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.pt"))

    def background_check(self, it: int, quick: bool):
        """Export + sim2sim courses (+ base env validation for full checks), on the CPU, outside the training process."""
        ck_path = os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.pt")
        out = os.path.join(self.cfg.log_dir, f"{'quick' if quick else 'check'}_{it:06d}.txt")
        subprocess.Popen(f"QUICK={int(quick)} TGI_PYTHON={sys.executable} bash scripts/check.sh {ck_path} 2 > {out} 2>&1",
                         shell=True, cwd=REPO, start_new_session=True)

    def fpm_active(self, it: int) -> bool:
        return self.fpm is not None and it >= self.cfg.fpm_bootstrap

    def _compile(self):
        """The per-step functions, as closures (torch.compile; CUDA graphs for the world-model loss and imagination)."""
        cfg, mdp, wm, pi, fpm, dev, dt = self.cfg, self.mdp, self.wm, self.pi, self.fpm, self.dev, self.sim.dt
        err_floor, jmap = self.err_floor, mdp.jmap

        def wm_loss(s, a, a_prev, xi, s1, c1, mask):
            S = unflatten_state(s[:, 0])
            preds, logits = [], []
            for j in range(cfg.wm_window):
                S, lg = wm.step(S, mdp.wm_terrain(S), mdp.joint_target(a[:, j]), mdp.joint_target(a_prev[:, j]),
                                normalize_xi(xi[:, j]))
                preds.append(S), logits.append(lg)
            S_hat = tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))
            logits = torch.stack(logits, 1)
            std = torch.maximum(wm.out_norm.std(), err_floor)
            e = ((wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2).mean(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, c1, reduction="none").mean(-1)
            B = s1.shape[0]
            l_dyn, l_c = (e * mask).sum() / B, (bce * mask).sum() / B
            acc = ((((logits > 0).float() == c1).float().mean(-1)) * mask).sum() / mask.sum()
            return l_dyn + cfg.contact_w * l_c, l_dyn, e[:, -1], acc

        def imag_step(S, K, o, extra, a_prev, cmd, xi, xi_n, xy_d, z_d, use_fpm: bool, eps):
            mu = pi(o.unsqueeze(1), extra)  # Isaac joint order
            a_i = mu + eps  # reparameterized exploration noise (zero without the likelihood-ratio term)
            a = jmap.to_mj(a_i)
            S1, lg = wm.step(S, K["tf"], mdp.joint_target(a), mdp.joint_target(a_prev), xi_n)
            K1 = mdp.context(S1)
            o1 = mdp.proprio(S1, a, cmd, xi)
            if use_fpm and cfg.w_reach != 0:
                # frozen FPM: footholds beyond the patch range are penalized; perception sees detached footholds
                fh = fpm(mdp.system_state(S1, cmd))
                reach = torch.relu(fh[..., 0].abs() - cfg.reach_x) + torch.relu(fh[..., 1].abs() - cfg.reach_y)
                K1 = dict(K1, reach=reach.sum(-1))
                fh = fh.detach()
            else:
                with torch.no_grad():
                    fh = fpm(mdp.system_state(S1, cmd).detach()) if use_fpm else bootstrap_footholds(o.shape[0], dev)
            extra = perceive(mdp, S1, K1, fh, xy_d, z_d)
            logp = -((a_i.detach() - mu) ** 2).sum(-1) / (2 * cfg.imag_noise**2)
            return S1, K1, torch.sigmoid(lg), a, o1, extra, logp

        def reward(S, K, c, a, a_prev, cmd):
            return mdp.reward(S, K, c, a, a_prev, cmd, dt), mdp.soft_termination(S, K)

        def critic_input(S, c, cmd, xi_n):
            return mdp.critic_features(S, mdp.context(S), c, cmd, xi_n)

        def compiled(f, graphs: bool):
            if not cfg.compile:
                return f
            return torch.compile(f, mode="reduce-overhead") if graphs else torch.compile(f)

        self.wm_graphs = bool(cfg.compile and cfg.cudagraphs)
        self.imag_graphs = bool(self.wm_graphs and cfg.cg_imag)
        self.wm_loss = compiled(wm_loss, self.wm_graphs)
        self.imag_step = compiled(imag_step, self.imag_graphs)
        self.reward = compiled(reward, False)
        self.critic_input = compiled(critic_input, False)

    def wm_update(self):
        cfg = self.cfg
        w = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window, keys=("s", "s1", "a", "a_prev", "xi", "c1", "done"))
        self.prof.tick("wm_sample")
        if self.wm_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        loss, l_dyn, e_last, acc = self.wm_loss(w["s"], w["a"], w["a_prev"], w["xi"], w["s1"], w["c1"], w["mask"])
        self.prof.tick("wm_fwd")
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        self.prof.tick("wm_bwd")
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        self.prof.tick("wm_opt")
        m = w["mask"][:, -1]
        # new tensors: CUDA-graph outputs are overwritten by the next replay
        return l_dyn.detach() / cfg.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1), acc.detach().clone()

    def policy_update(self, use_fpm: bool):
        cfg, mdp, dev, pi, critic = self.cfg, self.mdp, self.dev, self.pi, self.critic
        B, H, gamma = cfg.pi_batch, cfg.horizon, cfg.gamma
        if self.imag_graphs:
            torch.compiler.cudagraph_mark_step_begin()

        # start from replayed states, with fresh perception drift
        s, h = self.buffer.sample_starts(B, 2)
        S = unflatten_state(s["s"])
        xi, cmd, a_prev = s["xi"], s["cmd"], s["a_prev"]
        if cfg.imag_stand > 0:  # the world model is command-free: imagination may pick its own commands
            cmd = torch.where(torch.rand(B, 1, device=dev) < cfg.imag_stand, torch.zeros_like(cmd), cmd)
        xi_n = normalize_xi(xi)
        xy_d = (2 * torch.rand(B, 2, device=dev) - 1) * XY_DRIFT
        z_d = (2 * torch.rand(B, device=dev) - 1) * Z_DRIFT
        K = mdp.context(S)
        with torch.no_grad():
            o = mdp.proprio(S, a_prev, cmd, xi)
            c0 = h["c1"][:, 0]
            fh = self.fpm(mdp.system_state(S, cmd)) if use_fpm else bootstrap_footholds(B, dev)
            extra = perceive(mdp, S, K, fh, xy_d, z_d)
        requires_grad(self.wm, critic, self.fpm, flag=False)
        S0 = S
        Ss, Ks, Cs, As, Aps, Lps = [], [], [], [], [], []
        Os, Es = [o.detach()], [extra.detach()]
        use_lr = cfg.lr_coef > 0
        self.prof.tick("pi_setup")
        for t in range(H):
            Aps.append(a_prev)
            eps = cfg.imag_noise * torch.randn(B, 29, device=dev) if use_lr else torch.zeros(B, 29, device=dev)
            S, K, c, a_prev, o, extra, logp = self.imag_step(S, K, o, extra, a_prev, cmd, xi, xi_n, xy_d, z_d, use_fpm, eps)
            Ss.append(S), Ks.append(K), Cs.append(c), As.append(a_prev), Lps.append(logp)
            if t < H - 1:
                Os.append(o.detach()), Es.append(extra.detach())
        self.prof.tick("pi_imag_fwd")

        # rewards and soft terminations of all imagined steps at once
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        K_all = {k: flat([Kt[k] for Kt in Ks]) for k in Ks[0]}
        cmd_all = cmd.unsqueeze(0).expand(H, -1, -1).flatten(0, 1)
        r, d = self.reward(S_all, K_all, flat(Cs), flat(As), flat(Aps), cmd_all)
        r, d = r.view(H, B), d.view(H, B)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        surv = alive[-1]
        r_mean = (c_t * r).detach().sum(0) / H / self.sim.dt
        rews, dones = r.detach(), d.detach()

        # policy objective, differentiable through the world model
        if cfg.pg_lambda > 0 or cfg.pg_event > 0:
            # G_t = r_t + gamma (1 - d_t) ((1 - l_t) V(s_t+1) + l_t G_t+1); l_t = 1 except at cut touchdowns
            xi_all = xi_n.unsqueeze(0).expand(H, -1, -1).flatten(0, 1)
            Vg = critic(self.critic_input(S_all, flat(Cs), cmd_all, xi_all)).view(H, B)
            lam = torch.full((H, B), cfg.pg_lambda, device=dev)
            if cfg.pg_event > 0:  # cut the gradient through the model where it is least reliable
                with torch.no_grad():
                    Ct = torch.stack(Cs)
                    touchdown = Ct * (1 - torch.cat((c0.unsqueeze(0), Ct[:-1]), 0))  # soft, (H, B, 2)
                    unsupported = torch.stack([Kt["unsup"] for Kt in Ks])[..., 0]  # unsupported sole share
                    event = (touchdown * unsupported).amax(-1) > cfg.pg_event
                    lam = torch.where(event, lam, torch.ones_like(lam))
                self.last_event_share = event.float().mean()
            G = r[-1] + gamma * (1 - d[-1]) * Vg[-1]
            for t in range(H - 2, -1, -1):
                G = r[t] + gamma * (1 - d[t]) * ((1 - lam[t]) * Vg[t] + lam[t] * G)
            J = G
        else:  # H-step return + critic bootstrap
            disc = gamma ** torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(-1)
            J = (disc * c_t * r).sum(0) + gamma**H * surv * critic(self.critic_input(S, Cs[-1], cmd, xi_n))

        # critic values and lambda-return targets of the imagined transitions
        with torch.no_grad():
            stack0 = lambda x0, xs: torch.cat((x0.unsqueeze(0), torch.stack(xs)), 0).flatten(0, 1)  # noqa: E731
            S_c = tuple(stack0(S0[i], [St[i].detach() for St in Ss]) for i in range(6))
            c_c = stack0(c0, [c.detach() for c in Cs])
            cmd_c = cmd.unsqueeze(0).expand(H + 1, -1, -1).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(H + 1, -1, -1).flatten(0, 1)
            X = self.critic_input(S_c, c_c, cmd_c, xi_c).view(H + 1, B, -1)
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(H - 1, -1, -1):
                R = rews[t] + gamma * (1 - dones[t]) * ((1 - cfg.lam) * V[t + 1] + cfg.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])

        loss = -J.mean()
        if use_lr:  # likelihood-ratio term: credit actions by their lambda-return advantage
            adv = targets - V[:-1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            loss = loss - cfg.lr_coef * (c_t.detach() * torch.stack(Lps) * adv).mean()
        if cfg.sym_coef > 0:  # mirror consistency: pi(mirror(o)) = mirror(pi(o))
            Of, Ef = torch.cat(Os), torch.cat(Es)
            idx = torch.randint(0, Of.shape[0], (min(4096, Of.shape[0]),), device=dev)
            Of, Ef = Of[idx], Ef[idx]
            a_o = pi(Of.unsqueeze(1), Ef)
            a_m = pi(mirror_proprio(Of).unsqueeze(1), self.mirror_extra(Ef))
            sym_loss = ((a_m - mirror_joints(a_o)) ** 2).mean()
            loss = loss + cfg.sym_coef * sym_loss
            self.last_sym_loss = sym_loss.detach()
        self.prof.tick("pi_return_sym_fwd")
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        skip_if_nonfinite(self.opt_pi, gn)
        self.opt_pi.step()
        requires_grad(self.wm, critic, self.fpm, flag=True)
        self.prof.tick("pi_backward_step")

        critic.norm.update(X)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // cfg.critic_minibatches
        v_loss = torch.zeros((), device=dev)
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                self.opt_v.step()
        self.prof.tick("critic")
        return loss.detach(), v_loss.detach(), r_mean.mean(), gn.detach(), surv.mean().detach()

    def run(self):
        cfg, collector, logger = self.cfg, self.collector, self.logger
        collector.collect(max(cfg.warmup_steps, cfg.wm_window + 1), self.fpm_active(self.start_it))
        self.t_start = time.time() - self.wall_offset  # training time, without the setup
        t_job = logger.t_log = time.time()
        step_acc = 0.0
        for it in range(self.start_it, cfg.iterations):
            set_lr(self.opt_wm, cosine_lr(cfg.lr, it, cfg.lr_decay_iters))
            set_lr(self.opt_pi, cosine_lr(cfg.pi_lr, it, cfg.lr_decay_iters))
            set_lr(self.opt_v, cosine_lr(cfg.lr, it, cfg.lr_decay_iters))
            step_acc += 1.0 / cfg.utd
            n_steps = int(step_acc)
            step_acc -= n_steps
            self.prof.tick("other")

            # in threaded mode the new transitions reach the buffer after this iteration's updates
            job = collector.start(n_steps, self.fpm_active(it), overlap=collector.threaded and it >= self.overlap_from)
            if not collector.threaded:
                collector.finish(job)
            self.prof.tick("collect")
            for _ in range(cfg.wm_batches):
                wm_stats = self.wm_update()
            logger.add_wm(*wm_stats)  # the last update of the iteration
            self.prof.tick("wm")
            if self.fpm is not None and self.buffer.size > 100:
                if self.fpm_trainer.labels is None or it % cfg.fpm_label_every == 0:
                    self.fpm_trainer.refresh_labels(self.buffer, self.mdp.kin)
                self.prof.tick("fpm_labels")
                logger.add_fpm(self.fpm_trainer.update(self.buffer, self.mdp, cfg.fpm_batch, steps=cfg.fpm_steps))
            self.prof.tick("fpm")
            if it >= cfg.wm_only_iters and it >= self.start_it + cfg.resume_wm_only:
                logger.add_policy(*self.policy_update(self.fpm_active(it)))
            if collector.threaded:
                collector.finish(job)
                self.prof.tick("collect_wait")

            if (it + 1) % cfg.log_every == 0:
                extra = {}
                if cfg.sym_coef > 0:
                    extra["sym_loss"] = float(self.last_sym_loss)
                if cfg.pg_event > 0:
                    extra["pg_event steps"] = float(self.last_event_share)
                logger.write(it + 1, (time.time() - self.t_start) / 3600, collector.steps,
                             cosine_lr(cfg.pi_lr, it, cfg.lr_decay_iters), extra, sim=self.sim)
                self.prof.report(cfg.log_every)
            if (it + 1) % cfg.ckpt_every == 0 or it + 1 == cfg.iterations:
                self.save(it + 1)
                if it + 1 < cfg.iterations:  # the final evaluation is scripts/pipeline.sh's
                    if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                        self.background_check(it + 1, quick=False)
                    elif cfg.quick_every and (it + 1) % cfg.quick_every == 0:
                        self.background_check(it + 1, quick=True)
            if (time.time() - t_job) / 3600 > cfg.max_hours:
                self.save(it + 1)
                break

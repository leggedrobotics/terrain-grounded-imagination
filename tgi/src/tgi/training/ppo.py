"""PPO baseline in the TGI training environment (``--algo ppo``).

Same simulator, terrains, curriculum, domain randomization, perception (FPM-gated foothold scan with drift and noise),
policy network and normalizers, FPM co-training, critic features and reward (``mdp.reward`` on the real transitions,
real terminations). Only the policy optimizer differs: PPO with the base env's rsl-rl settings (24 steps, 5 epochs x 4
minibatches, adaptive KL 0.01, clip 0.2, entropy 0.005, mirror augmentation). Checkpoints have the TGI layout, so
``export.py`` and the evaluation suites run unchanged.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn

from ..mdp import normalize_xi, unflatten_state
from ..mdp.interface import mirror_joints, mirror_proprio
from ..policy_io import SimRunner
from .trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        dev = self.dev
        self.log_std = nn.Parameter(torch.full((29,), math.log(cfg.ppo_init_std), device=dev))
        self.params = list(self.pi.parameters()) + [self.log_std] + list(self.critic.parameters())
        self.opt = torch.optim.Adam(self.params, lr=cfg.ppo_lr)
        self.ppo_lr = cfg.ppo_lr
        self.runner = SimRunner(self.pi, self.fpm, self.mdp, self.sim, noise=True)
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=dev)
            self.log_std.data.copy_(ck["log_std"])
            self.opt.load_state_dict(ck["opt"])
            self.ppo_lr = self.opt.param_groups[0]["lr"]
            if "sgs" in ck:  # curriculum state, then redraw every environment's terrain cell from it
                sim = self.sim
                sim._sgs.outcomes.copy_(ck["sgs"]["outcomes"].to(dev)), sim._sgs.count.copy_(ck["sgs"]["count"].to(dev))
                sim._first = True
                sim.env.reset()
                sim._first = False
                sim.start_xy[:] = sim.robot.data.root_link_pose_w.torch[:, :2]

    def critic_x(self, S, c, cmd, xi):
        return self.mdp.critic_features(S, self.mdp.context(S), c, cmd, normalize_xi(xi))

    def logp(self, mu, a):
        std = self.log_std.exp()
        return (-((a - mu) ** 2) / (2 * std**2) - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def save(self, it: int):
        sim = self.sim
        ck = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
              "fpm": self.fpm.state_dict() if self.fpm is not None else None, "fpm_active": self.fpm_active(it),
              "interface": "base_g1_env", "log_std": self.log_std.detach().cpu(), "iteration": it,
              "wall_time": time.time() - self.t_start, "sim_steps": self.steps, "args": vars(self.cfg), "algo": "ppo",
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "sgs": {"outcomes": sim._sgs.outcomes.cpu(), "count": sim._sgs.count.cpu()}}  # fmt: skip
        torch.save(ck, os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.pt"))
        torch.save(dict(ck, opt=self.opt.state_dict()), os.path.join(self.cfg.log_dir, "last.pt"))

    @torch.no_grad()
    def rollout(self, ag: dict):
        cfg, sim, mdp, pi, critic = self.cfg, self.sim, self.mdp, self.pi, self.critic
        O, E, A, X, R, D = [], [], [], [], [], []
        for _ in range(cfg.ppo_steps):
            S, cmd, xi = sim.state(), sim.command(), sim.xi()
            o, extra = self.runner.observe()
            X.append(self.critic_x(S, self.c_now, cmd, xi))
            mu = pi(o.unsqueeze(1), extra)
            a = mu + self.log_std.exp() * torch.randn_like(mu)
            out = sim.step(torch.clamp(mdp.jmap.to_mj(a), -10, 10))
            self.buffer.add(out)
            S1 = unflatten_state(out["s1"])
            r = mdp.reward(S1, mdp.context(S1), out["c1"], out["a"], out["a_prev"], out["cmd"], sim.dt)
            timeout = out["done"] & ~out["term"]
            if bool(timeout.any()):  # time limits bootstrap from the pre-reset state
                r = r + cfg.gamma * timeout.float() * critic(self.critic_x(S1, out["c1"], out["cmd"], out["xi"]))
            self.c_now = sim.foot_contact()
            O.append(o), E.append(extra), A.append(a), R.append(r), D.append(out["done"].float())
            self.steps += sim.N
            ag["r"].append(r.mean() / sim.dt)
            self.logger.add_transition(out, S1, sim.dt)
        self.logger.add_level(sim.env.scene.terrain.terrain_levels.float().mean())
        X_last = self.critic_x(sim.state(), self.c_now, sim.command(), sim.xi())
        O, E, A, X, R, D = (torch.stack(v) for v in (O, E, A, X, R, D))
        # normalizers from this batch, then the old policy and values under the updated normalizers (ratio 1 at start)
        pi.obs_norm.update(O), pi.extra_norm.update(E), critic.norm.update(torch.cat((X, X_last.unsqueeze(0))))
        T, N = R.shape
        MU = pi(O.flatten(0, 1).unsqueeze(1), E.flatten(0, 1)).view(T, N, -1)
        V, v_next = critic(X), critic(X_last)
        adv, gae = torch.zeros_like(R), torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):
            vn = v_next if t == T - 1 else V[t + 1]
            gae = R[t] + cfg.gamma * (1 - D[t]) * vn - V[t] + cfg.gamma * cfg.lam * (1 - D[t]) * gae
            adv[t] = gae
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return [x.flatten(0, 1) for x in (O, E, A, X, self.logp(MU, A), V, ret, adv, MU)]

    def update(self, batch, ag: dict):
        cfg, pi, critic = self.cfg, self.pi, self.critic
        n = batch[0].shape[0]
        mb = n // cfg.ppo_minibatches
        std_old = self.log_std.detach().exp()
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.dev)
            for k in range(cfg.ppo_minibatches):
                o, e, a, x, lp_old, v_old, ret, ad, mu_old = (b[perm[k * mb : (k + 1) * mb]] for b in batch)
                mu = pi(o.unsqueeze(1), e)
                std = self.log_std.exp()
                with torch.no_grad():  # adaptive learning rate from the KL to the rollout policy
                    kl = (torch.log(std / std_old) + (std_old**2 + (mu_old - mu) ** 2) / (2 * std**2) - 0.5).sum(-1).mean()
                    if kl > 2 * cfg.ppo_kl:
                        self.ppo_lr = max(1e-5, self.ppo_lr / 1.5)
                    elif 0 < kl < cfg.ppo_kl / 2:
                        self.ppo_lr = min(1e-2, self.ppo_lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.ppo_lr
                if cfg.ppo_sym:  # mirrored copies share the advantage and the old log-probability
                    mu = torch.cat((mu, pi(mirror_proprio(o).unsqueeze(1), self.mirror_extra(e))))
                    a = torch.cat((a, mirror_joints(a)))
                    lp_old, ad = lp_old.repeat(2), ad.repeat(2)
                ratio = torch.exp(self.logp(mu, a) - lp_old)
                surr = -torch.min(ratio * ad, torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * ad).mean()
                v = critic(x)
                v_cl = v_old + (v - v_old).clamp(-cfg.ppo_clip, cfg.ppo_clip)
                v_loss = torch.max((v - ret) ** 2, (v_cl - ret) ** 2).mean()
                ent = (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                self.opt.zero_grad(set_to_none=True)
                (surr + v_loss - cfg.ppo_entropy * ent).backward()
                gn = nn.utils.clip_grad_norm_(self.params, 1.0)
                self.opt.step()
        ag["pi"].append(surr.detach()), ag["v"].append(v_loss.detach()), ag["gn"].append(gn.detach())
        ag["std"].append(self.log_std.detach().exp().mean())

    def run(self):
        cfg, logger = self.cfg, self.logger
        self.steps = self.collector.steps
        self.c_now = self.sim.foot_contact()
        self.t_start = time.time() - self.wall_offset
        t_job = logger.t_log = time.time()
        ag = {k: [] for k in ("r", "pi", "v", "gn", "std", "fpm")}
        for it in range(self.start_it, cfg.iterations):
            self.runner.use_fpm_model = self.fpm_active(it)
            self.update(self.rollout(ag), ag)
            if self.fpm is not None and self.buffer.size > 100:
                if self.fpm_trainer.labels is None or it % cfg.fpm_label_every == 0:
                    self.fpm_trainer.refresh_labels(self.buffer, self.mdp.kin)
                ag["fpm"].append(self.fpm_trainer.update(self.buffer, self.mdp, cfg.fpm_batch, steps=cfg.fpm_steps))
            if (it + 1) % cfg.log_every == 0:
                for k in ag:
                    ag[k] = float(sum(float(v) for v in ag[k]) / max(len(ag[k]), 1))
                logger.upd["fpm"].append(ag["fpm"])
                logger.upd["pi"].append(ag["pi"]), logger.upd["v"].append(ag["v"]), logger.upd["ir"].append(ag["r"])
                logger.upd["gn"].append(ag["gn"]), logger.upd["surv"].append(ag["std"])
                self.collector.steps = self.steps
                logger.write(it + 1, (time.time() - self.t_start) / 3600, self.steps, self.ppo_lr, {"std": ag["std"]},
                             sim=self.sim)
                ag = {k: [] for k in ag}
            if (it + 1) % cfg.ckpt_every == 0 or it + 1 == cfg.iterations:
                self.save(it + 1)
            if (time.time() - t_job) / 3600 > cfg.max_hours:
                self.save(it + 1)
                break

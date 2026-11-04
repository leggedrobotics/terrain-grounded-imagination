"""Training statistics (``train_log.csv`` and console), accumulated on the GPU and read back only when logging."""

from __future__ import annotations

import os
import time

import torch

COLUMNS = ("iteration,wall_h,sim_steps,wm_loss,wm_err8,contact_acc,fpm_loss,pi_loss,v_loss,imag_rew,pi_gradnorm,"
           "imag_survival,sim_rew,vel_err,ep_len,ep_rew,term_frac,terrain_level,it_per_s,lr")
N_TERRAIN_TYPES = 64


def _mean(xs) -> float:
    return float(sum(float(v) for v in xs) / max(len(xs), 1))


class TrainLogger:
    def __init__(self, log_dir: str, resume: bool, log_every: int, terrain_names: list[str], device: str):
        self.log_every = log_every
        self.terrain_names = terrain_names
        self.csv = open(os.path.join(log_dir, "train_log.csv"), "a")
        if not resume:
            self.csv.write(COLUMNS + "\n")
        self.sim = {k: [] for k in ("r", "vel_err", "ep_len", "ep_ret", "term", "done", "level")}
        self.upd = {k: [] for k in ("wm", "wm8", "acc", "fpm", "pi", "v", "ir", "gn", "surv")}
        self.ep_terms: dict[str, list] = {}
        # per terrain type: episodes, terminations, summed level, summed length
        self.terrain = torch.zeros(N_TERRAIN_TYPES, 4, device=device)
        self.t_log = time.time()

    def add_transition(self, out: dict, S1, dt: float):
        """One simulator step (``IsaacLocoSim.step`` output and the next state)."""
        self.sim["r"].append(out["r_sim"].mean() / dt)
        Rt = S1[1].transpose(-1, -2)
        v_b = (Rt @ S1[2].unsqueeze(-1)).squeeze(-1)
        self.sim["vel_err"].append((v_b[:, :2] - out["cmd"][:, :2]).norm(dim=-1).mean())
        if "ep_len" not in out:
            return
        self.sim["ep_len"].append(out["ep_len"].mean())
        self.sim["ep_ret"].append((out["ep_ret"] / out["ep_len"].clamp(min=1) / dt).mean())
        self.sim["term"].append(out["term"].float().sum())
        self.sim["done"].append(out["done"].float().sum())
        for k, v in out["episode_log"].items():
            self.ep_terms.setdefault(k, []).append(v)
        ty = out["ep_type"].long().clamp(0, N_TERRAIN_TYPES - 1)
        tm = out["term"][out["ep_ids"]].float()
        rows = torch.stack((torch.ones_like(tm), tm, out["ep_level"].float(), out["ep_len"].float()), -1)
        self.terrain.index_add_(0, ty, rows)

    def add_level(self, level: torch.Tensor):
        self.sim["level"].append(level)

    def add_wm(self, loss, err8, acc):
        self.upd["wm"].append(loss), self.upd["wm8"].append(err8), self.upd["acc"].append(acc)

    def add_fpm(self, loss):
        self.upd["fpm"].append(loss)

    def add_policy(self, pi_loss, v_loss, imag_rew, grad_norm, survival):
        for k, v in zip(("pi", "v", "ir", "gn", "surv"), (pi_loss, v_loss, imag_rew, grad_norm, survival)):
            self.upd[k].append(v)

    def write(self, it: int, wall_h: float, sim_steps: int, lr: float, extra: dict[str, float], sim=None):
        """Log line for iteration ``it`` (1-based); ``extra``: further values for the console; ``sim``: curriculum source."""
        ips = self.log_every / (time.time() - self.t_log)
        self.t_log = time.time()
        s, u = self.sim, self.upd
        term_frac = float(sum(float(v) for v in s["term"]) / max(sum(float(v) for v in s["done"]), 1))
        row = [it, wall_h, sim_steps, _mean(u["wm"]), _mean(u["wm8"]), _mean(u["acc"]), _mean(u["fpm"]), _mean(u["pi"]),
               _mean(u["v"]), _mean(u["ir"]), _mean(u["gn"]), _mean(u["surv"]), _mean(s["r"]), _mean(s["vel_err"]),
               _mean(s["ep_len"]), _mean(s["ep_ret"]), term_frac, _mean(s["level"]), ips, lr]  # fmt: skip
        self.csv.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        self.csv.flush()
        print(f"it {it} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.3f} cacc {row[5]:.3f} fpm {row[6]:.4f} | "
              f"pi {row[7]:.3f} v {row[8]:.4f} imag_r {row[9]:.3f} gn {row[10]:.2f} surv {row[11]:.3f} | sim_r {row[12]:.3f} "
              f"verr {row[13]:.3f} ep_len {row[14]:.0f} ep_r {row[15]:.3f} term {term_frac:.2f} lvl {row[17]:.2f} | "
              f"{ips:.2f} it/s | torch {torch.cuda.max_memory_reserved() / 1e9:.2f} GB", flush=True)  # fmt: skip
        for k, v in extra.items():
            print(f"   {k} {v:.4f}", flush=True)
        if it % (4 * self.log_every) == 0 and self.ep_terms:
            self._print_episodes()
            if sim is not None:
                self._print_curriculum(sim, profile=it % (20 * self.log_every) == 0)
        self.sim = {k: [] for k in self.sim}
        self.upd = {k: [] for k in self.upd}

    def _name(self, ty: int) -> str:
        return self.terrain_names[ty] if ty < len(self.terrain_names) else str(ty)

    def _print_episodes(self):
        print("   episode terms: " + " ".join(f"{k.split('/')[-1]}={float(sum(v)) / len(v):.3f}"
                                              for k, v in self.ep_terms.items()), flush=True)
        self.ep_terms.clear()
        by_name: dict[str, list] = {}
        for ty, (n, tm, lv, ln) in enumerate(self.terrain.tolist()):
            if n > 0:
                b = by_name.setdefault(self._name(ty), [0, 0, 0.0, 0.0])
                b[0] += n
                b[1] += tm
                b[2] += lv
                b[3] += ln
        self.terrain.zero_()
        if by_name:
            print("   terrain: " + " ".join(f"{k}(n={n:.0f} fall={tm / max(n, 1):.2f} lvl={lv / max(n, 1):.1f} "
                                            f"len={ln / max(n, 1):.0f})" for k, (n, tm, lv, ln) in sorted(by_name.items())),
                  flush=True)

    def _print_curriculum(self, sim, profile: bool):
        print("   sgs: " + " ".join(f"{k}={v:.3f}" for k, v in sim.sgs_stats().items()), flush=True)
        if profile:
            for ty, (succ, share, cnt) in sim.sgs_profile().items():
                print(f"   sgs {self._name(ty):22s} success by level " + " ".join(f"{v:.2f}" for v in succ.tolist())
                      + " | share " + " ".join(f"{v:.2f}" for v in share.tolist())
                      + " | visits/config " + " ".join(f"{v:.0f}" for v in cnt.tolist()), flush=True)

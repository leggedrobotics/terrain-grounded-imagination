"""Gradient check of the imagined policy objective.

Start states come from a fresh simulator collection with the checkpoint's policy (all terrains and curriculum levels).
For each start, the imagined objective J is differentiated with respect to the first action (an offset added to the
policy mean at t = 0), by backpropagation and by central finite differences over all 29 action dimensions:

* cut 0: the H-step return with a critic bootstrap (--pg_event 0);
* cut 1: the return cut at imagined touchdowns with an unsupported sole share above --pg_event (the cut mask is taken
  from the unperturbed rollout, as in training, where it carries no gradient).

The result is the cosine between the gradient vectors per terrain, horizon and objective, for all starts and for starts
whose rollout contains a touchdown with less than 60 % sole support. Each perturbed rollout reuses the observation noise
of the unperturbed one, and TF32 is off.

    python scripts/grad_check.py --ckpt logs/<run>/ckpt_020000.pt --out logs/<run>/grad_check

Writes grad_check.csv (one row per terrain, subset, horizon and cut) and grad_check.json (settings).
"""

from __future__ import annotations

import argparse
import csv
import json
import os

HORIZONS = (8, 12, 16, 20, 24, 28, 32)
POOLED = {"stepping_stones_all": ("stepping_stones", "stepping_stones_flat")}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="training checkpoint (ckpt_<it>.pt or last.pt)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--num_envs", type=int, default=1024, help="simulator environments for the start-state collection")
    p.add_argument("--starts_per_terrain", type=int, default=1536)
    p.add_argument("--chunk", type=int, default=768, help="start states per batch")
    p.add_argument("--fd_eps", type=float, nargs=2, default=(0.05, 0.005), help="finite-difference steps")
    p.add_argument("--pg_event", type=float, default=0.3, help="cut threshold of objective 1")
    p.add_argument("--low_support", type=float, default=0.4, help="unsupported sole share that marks a poor touchdown")
    p.add_argument("--vram_gb", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    import torch

    from tgi.training.config import parse_args

    ck_args = torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"]
    cfg = parse_args(["--log_dir", a.out])
    for k, v in ck_args.items():  # the checkpoint's recipe
        if hasattr(cfg, k) and k not in ("log_dir", "resume"):
            setattr(cfg, k, v)
    cfg.resume, cfg.num_envs, cfg.vram_gb, cfg.seed = a.ckpt, a.num_envs, a.vram_gb, a.seed
    cfg.compile, cfg.async_collect, cfg.eval_every, cfg.quick_every = 0, 0, 0, 0
    if cfg.scan == "full":
        cfg.fpm = 0

    from tgi.mdp import normalize_xi, unflatten_state
    from tgi.mdp.interface import XY_DRIFT, Z_DRIFT
    from tgi.policy_io import bootstrap_footholds, perceive
    from tgi.training.trainer import Trainer

    tr = Trainer(cfg)
    sim, mdp, buf, dev, gamma = tr.sim, tr.mdp, tr.buffer, tr.dev, cfg.gamma
    use_fpm = tr.fpm_active(tr.start_it)
    names = tr.logger.terrain_names
    HZ = max(HORIZONS)

    # start states: a full replay buffer of fresh transitions, with the terrain type and level of each row
    types, levels = [], []
    terrain, sim_step = sim.env.scene.terrain, sim.step

    def step(act):
        types.append(sim.col_type[terrain.terrain_types].clone())
        levels.append(terrain.terrain_levels.clone())
        return sim_step(act)

    sim.step = step
    tr.collector.collect(buf.T, use_fpm)
    sim.step = sim_step
    types, levels = torch.stack(types[-buf.T:]), torch.stack(levels[-buf.T:])  # (age, env), oldest first

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for m in (tr.pi, tr.wm, tr.critic, tr.fpm):
        if m is not None:
            m.requires_grad_(False)

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    ages, envs = [], []
    for t in types.unique().tolist():
        idx = (types[1:] == t).nonzero()  # age >= 1: the previous contact state is in the buffer
        idx = idx[torch.randperm(len(idx), generator=gen, device=dev)[: a.starts_per_terrain]]
        ages.append(idx[:, 0] + 1), envs.append(idx[:, 1])
    ages, envs = torch.cat(ages), torch.cat(envs)
    print(f"[grad_check] {len(ages)} start states", flush=True)

    def starts(age, env):
        row = buf._row(age)
        prev = buf._row(age - (buf.data["ep_step"][row, env] >= 1).long())
        st = {k: buf.data[k][row, env] for k in ("s", "a_prev", "xi", "cmd")}
        st["c0"] = buf.data["c1"][prev, env]
        return st

    def rollout(st, eps0, xy_d, z_d, seed):
        """H imagined steps from st with eps0 added to the first action: rewards, terminations, values, cut signal."""
        torch.manual_seed(seed)  # the same observation noise in every perturbed rollout
        B = eps0.shape[0]
        S = unflatten_state(st["s"])
        xi, cmd, a_prev = st["xi"], st["cmd"], st["a_prev"]
        xi_n = normalize_xi(xi)
        K = mdp.context(S)
        with torch.no_grad():
            o = mdp.proprio(S, a_prev, cmd, xi)
            fh = tr.fpm(mdp.system_state(S, cmd)) if use_fpm else bootstrap_footholds(B, dev)
            extra = perceive(mdp, S, K, fh, xy_d, z_d)
        steps, zero = [], torch.zeros(B, 29, device=dev)
        for t in range(HZ):
            prev_a = a_prev
            S, K, c, a_prev, o, extra, _ = tr.imag_step(S, K, o, extra, a_prev, cmd, xi, xi_n, xy_d, z_d, use_fpm,
                                                        eps0 if t == 0 else zero)
            steps.append((S, K, c, a_prev, prev_a))
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([s[0][i] for s in steps]) for i in range(6))
        K_all = {k: flat([s[1][k] for s in steps]) for k in steps[0][1]}
        C = flat([s[2] for s in steps])
        cmd_all = cmd.unsqueeze(0).expand(HZ, -1, -1).flatten(0, 1)
        xi_all = xi_n.unsqueeze(0).expand(HZ, -1, -1).flatten(0, 1)
        r, d = tr.reward(S_all, K_all, C, flat([s[3] for s in steps]), flat([s[4] for s in steps]), cmd_all)
        V = tr.critic(tr.critic_input(S_all, C, cmd_all, xi_all)).view(HZ, B)
        with torch.no_grad():
            Ct = torch.stack([s[2] for s in steps])
            touchdown = Ct * (1 - torch.cat((st["c0"].unsqueeze(0), Ct[:-1]), 0))
            unsup = torch.stack([s[1]["unsup"] for s in steps])[..., 0]
            event = (touchdown * unsup).amax(-1)  # (HZ, B)
        return r.view(HZ, B), d.view(HZ, B), V, event

    def objectives(r, d, V, event):
        """J for every (horizon, cut)."""
        disc = gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
        J = {}
        for H in HORIZONS:
            J[(H, 0)] = (disc[:H] * c_t[:H] * r[:H]).sum(0) + gamma**H * alive[H - 1] * V[H - 1]
            lam = torch.where(event[:H] > a.pg_event, 0.0, 1.0)
            G = r[H - 1] + gamma * (1 - d[H - 1]) * V[H - 1]
            for t in range(H - 2, -1, -1):
                G = r[t] + gamma * (1 - d[t]) * ((1 - lam[t]) * V[t] + lam[t] * G)
            J[(H, 1)] = G
        return J

    keys = [(H, c) for H in HORIZONS for c in (0, 1)]
    bp = {k: [] for k in keys}
    fd = {(k, e): [] for k in keys for e in a.fd_eps}
    low = {H: [] for H in HORIZONS}
    for i0 in range(0, len(ages), a.chunk):
        st = starts(ages[i0 : i0 + a.chunk], envs[i0 : i0 + a.chunk])
        B = st["s"].shape[0]
        xy_d = (2 * torch.rand(B, 2, device=dev, generator=gen) - 1) * XY_DRIFT
        z_d = (2 * torch.rand(B, device=dev, generator=gen) - 1) * Z_DRIFT
        seed = 1000 + i0
        eps0 = torch.zeros(B, 29, device=dev, requires_grad=True)
        r, d, V, event = rollout(st, eps0, xy_d, z_d, seed)
        J = objectives(r, d, V, event)
        for j, k in enumerate(keys):
            bp[k].append(torch.autograd.grad(J[k].sum(), eps0, retain_graph=j < len(keys) - 1)[0])
        del r, d, V, J
        with torch.no_grad():
            for e in a.fd_eps:
                g = {k: torch.zeros(B, 29, device=dev) for k in keys}
                for dim in range(29):
                    u = torch.zeros(B, 29, device=dev)
                    u[:, dim] = e
                    Jp = objectives(*rollout(st, u, xy_d, z_d, seed)[:3], event)
                    Jm = objectives(*rollout(st, -u, xy_d, z_d, seed)[:3], event)
                    for k in keys:
                        g[k][:, dim] = (Jp[k] - Jm[k]) / (2 * e)
                for k in keys:
                    fd[(k, e)].append(g[k])
        for H in HORIZONS:
            low[H].append(event[:H].amax(0) > a.low_support)
        print(f"[grad_check] {i0 + B}/{len(ages)}", flush=True)

    bp = {k: torch.cat(v) for k, v in bp.items()}
    fd = {k: torch.cat(v) for k, v in fd.items()}
    low = {H: torch.cat(v) for H, v in low.items()}
    ty, lv = types[ages, envs], levels[ages, envs]

    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1, eps=1e-12)  # noqa: E731
    q = lambda x, p: float(torch.quantile(x, p))  # noqa: E731
    groups = [("all", torch.ones_like(ty, dtype=torch.bool))]
    groups += [(names[t], ty == t) for t in sorted(ty.unique().tolist())]
    groups += [(g, sum(ty == names.index(m) for m in ms).bool()) for g, ms in POOLED.items() if all(m in names for m in ms)]
    e0, e1 = a.fd_eps
    rows = []
    for name, gm in groups:
        for H in HORIZONS:
            for subset, sm in (("all", gm), ("pre_touchdown_lt60", gm & low[H])):
                if not bool(sm.any()):
                    continue
                for c in (0, 1):
                    g = bp[(H, c)][sm]
                    own, own_s = cos(g, fd[((H, c), e0)][sm]), cos(g, fd[((H, c), e1)][sm])
                    full, full_s = cos(g, fd[((H, 0), e0)][sm]), cos(g, fd[((H, 0), e1)][sm])
                    norm = g.norm(dim=-1)
                    rows.append({
                        "terrain": name, "subset": subset, "horizon": H, "cut": c, "n": int(sm.sum()),
                        "median_cosine": q(own, 0.5), "q25": q(own, 0.25), "q75": q(own, 0.75),
                        "median_cosine_small_eps": q(own_s, 0.5), "q25_small_eps": q(own_s, 0.25),
                        "q75_small_eps": q(own_s, 0.75),
                        "median_cosine_fd_vs_fd": q(cos(fd[((H, c), e0)][sm], fd[((H, c), e1)][sm]), 0.5),
                        "p99_grad_norm": q(norm, 0.99), "median_grad_norm": q(norm, 0.5), "fd_eps": e0,
                        "fd_eps_small": e1, "mean_level": float(lv[sm].float().mean()),
                        "median_cosine_vs_uncut_fd": q(full, 0.5), "q25_vs_uncut_fd": q(full, 0.25),
                        "q75_vs_uncut_fd": q(full, 0.75), "median_cosine_vs_uncut_fd_small_eps": q(full_s, 0.5),
                        "q25_vs_uncut_fd_small_eps": q(full_s, 0.25), "q75_vs_uncut_fd_small_eps": q(full_s, 0.75),
                    })  # fmt: skip
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "grad_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({**vars(a), "iteration": tr.start_it, "low_support_share": {H: float(v.float().mean()) for H, v in low.items()}},
              open(os.path.join(a.out, "grad_check.json"), "w"), indent=2)
    for r_ in rows:
        if r_["terrain"] == "all":
            print(f"[grad_check] {r_['subset']:18s} H={r_['horizon']:2d} cut={r_['cut']} n={r_['n']:5d} "
                  f"cos {r_['median_cosine']:.3f} vs uncut FD {r_['median_cosine_vs_uncut_fd']:.3f}", flush=True)
    os._exit(0)  # skip the simulator's slow shutdown


if __name__ == "__main__":
    main()

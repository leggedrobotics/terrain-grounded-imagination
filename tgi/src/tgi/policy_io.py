"""Closed-loop policy execution in the simulator and checkpoint loading."""

from __future__ import annotations

import torch

from .foot_pred import FootholdPredictor
from .mdp import MDP, G1Kinematics, RewardCfg, TerrainFeatures
from .mdp.interface import (BOOTSTRAP_FOOTHOLDS, FOOTHOLD_SCAN_DIM, FULL_SCAN_DIM, PROPRIO_DIM, XY_DRIFT, Z_DRIFT,
                             FootholdScan, FullHeightScan)  # fmt: skip

# policy perception: "foothold" (FPM foothold patches) or "full" (whole height scan)
SCAN = {"mode": "foothold"}


def set_scan_mode(mode: str):
    assert mode in ("foothold", "full"), mode
    SCAN["mode"] = mode


def scan_dim() -> int:
    return FULL_SCAN_DIM if SCAN["mode"] == "full" else FOOTHOLD_SCAN_DIM
from .world_model import Policy


def build_mdp(sim, device: str, reward_cfg: RewardCfg | None = None, field_window: float = 0.5,
              field_hole: float = 0.25, hm_res: float = 0.04, hm_exact: bool = False) -> MDP:
    """Terrain mesh -> heightmap (+ support field), MJCF kinematics, MJCF-ordered limits."""
    hm = sim.heightmap(device, res=hm_res)
    hm.exact_support = hm_exact
    hm.build_support_field(window=field_window, hole_depth=field_hole)
    lo, hi = sim.soft_limits
    terrain = TerrainFeatures(hm, device)
    kin = G1Kinematics(sim.kin_model, sim.joint_names, device)
    mdp = MDP(kin, terrain, sim.default_q, sim.joint_names, lo, hi, sim.action_scale, reward_cfg or RewardCfg())
    mdp.scan = FullHeightScan(hm, device) if SCAN["mode"] == "full" else FootholdScan(hm, device)
    return mdp


def bootstrap_footholds(n: int, device: str) -> torch.Tensor:
    return torch.tensor(BOOTSTRAP_FOOTHOLDS, device=device).expand(n, 2, 2)


def perceive(mdp: MDP, S, K, fh, xy_drift=None, z_drift=None, noise: bool = True, return_points: bool = False):
    return mdp.scan(S[0], S[1], K["torso_p"], K["torso_R"], fh, xy_drift, z_drift, noise, return_points)


def make_policy(device: str) -> Policy:
    return Policy(PROPRIO_DIM, hist=1, extra_dim=scan_dim()).to(device)


class SimRunner:
    def __init__(self, pi: Policy, fpm: FootholdPredictor | None, mdp: MDP, sim, noise: bool = True):
        self.pi, self.fpm, self.mdp, self.sim, self.noise = pi, fpm, mdp, sim, noise
        dev = sim.device
        self.use_fpm_model = fpm is not None
        self.xy_drift = torch.zeros(sim.N, 2, device=dev)
        self.z_drift = torch.zeros(sim.N, device=dev)
        self.footholds = bootstrap_footholds(sim.N, dev).clone()
        self.vis = None

    def _resample_drift(self, ids):
        if not self.noise:
            return
        self.xy_drift[ids] = (2 * torch.rand(len(ids), 2, device=self.sim.device) - 1) * XY_DRIFT
        self.z_drift[ids] = (2 * torch.rand(len(ids), device=self.sim.device) - 1) * Z_DRIFT

    @torch.no_grad()
    def observe(self):
        sim, mdp = self.sim, self.mdp
        S = sim.state()
        cmd = sim.command()
        if self.noise:  # new episodes redraw their perception drift (masked: no host sync)
            m = sim.ep_len == 0
            self.xy_drift = torch.where(m.unsqueeze(-1), (2 * torch.rand_like(self.xy_drift) - 1) * XY_DRIFT, self.xy_drift)
            self.z_drift = torch.where(m, (2 * torch.rand_like(self.z_drift) - 1) * Z_DRIFT, self.z_drift)
        o = mdp.proprio(S, sim.a_prev, cmd, sim.xi(), noise=self.noise)
        if self.fpm is not None and self.use_fpm_model:
            fh = self.fpm(mdp.system_state(S, cmd))
        else:
            fh = bootstrap_footholds(sim.N, sim.device)
        self.footholds = fh
        K = mdp.context(S)
        extra, pts, hz = perceive(mdp, S, K, fh, self.xy_drift, self.z_drift, self.noise, return_points=True)
        self.vis = {"pts": pts, "hz": hz}
        return o, extra

    @torch.no_grad()
    def act(self):
        """Returns actions in the simulator's (MJCF) joint order."""
        o, e = self.observe()
        return self.mdp.jmap.to_mj(self.pi(o.unsqueeze(1), e))


class Bundle:
    """Policy and FPM of a training checkpoint (``ckpt_<it>.pt``)."""

    def __init__(self, ck: dict, device: str):
        self.ck, self.args = ck, ck["args"]
        set_scan_mode(self.args.get("scan", "foothold"))
        self.pi = make_policy(device)
        self.pi.load_state_dict(ck["pi"])
        self.pi.eval()
        self.fpm = None
        if ck.get("fpm") is not None:
            self.fpm = FootholdPredictor().to(device)
            self.fpm.load_state_dict(ck["fpm"])
            self.fpm.eval()
        self.device = device

    def make_runner(self, sim, noise: bool = True):
        a = self.args
        mdp = build_mdp(sim, self.device, field_window=a.get("field_window", 0.5), field_hole=a.get("field_hole", 0.25),
                        hm_res=a.get("hm_res", 0.04), hm_exact=bool(a.get("hm_exact", 0)))
        r = SimRunner(self.pi, self.fpm, mdp, sim, noise)
        r.use_fpm_model = self.fpm is not None and self.ck.get("fpm_active", True)
        return r


def load_policy_bundle(path: str, device: str) -> Bundle:
    return Bundle(torch.load(path, map_location=device, weights_only=False), device)

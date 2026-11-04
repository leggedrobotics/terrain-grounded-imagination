"""Terrain features from the heightmap for the world model and the critic (pelvis height, base and sole scans)."""

from __future__ import annotations

import torch

from .heightmap import HeightMap
from .rotations import yaw_mat

NOMINAL_HEIGHT = 0.75


class TerrainFeatures:
    FOOT_X = (-0.1, 0.1, 5)  # sole spans -0.094..0.092 m around the foot site
    FOOT_Y = (-0.04, 0.04, 3)
    BASE_X = (-0.3, 0.9, 7)
    BASE_Y = (-0.4, 0.4, 5)
    CRITIC_X = (-0.5, 0.9, 8)
    CRITIC_Y = (-0.4, 0.4, 5)

    def __init__(self, hm: HeightMap, device: str):
        self.hm = hm
        self.foot_grid = self._rect(self.FOOT_X, self.FOOT_Y, device)
        self.base_grid = self._rect(self.BASE_X, self.BASE_Y, device)
        self.foot_center = (self.FOOT_X[2] // 2) * self.FOOT_Y[2] + self.FOOT_Y[2] // 2
        self.critic_grid = self._rect(self.CRITIC_X, self.CRITIC_Y, device)
        self.hole_grid = self._rect((-0.3, 0.3, 5), (-0.3, 0.3, 5), device)
        nf, nb = self.foot_grid.shape[0], self.base_grid.shape[0]
        self.wm_dim = 1 + nb + 2 * (1 + nf)
        self.critic_dim = self.critic_grid.shape[0]

    @staticmethod
    def _rect(xr, yr, device) -> torch.Tensor:
        xs = torch.linspace(*xr, device=device)
        ys = torch.linspace(*yr, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((gx.flatten(), gy.flatten()), -1)

    @staticmethod
    def _yaw2(R: torch.Tensor) -> torch.Tensor:
        return yaw_mat(R)[..., :2, :2]

    def _points(self, center_xy, R2, grid):
        # center (..., 2), R2 (..., 2, 2), grid (G, 2) -> (..., G, 2) world xy
        return center_xy.unsqueeze(-2) + (R2.unsqueeze(-3) @ grid.unsqueeze(-1)).squeeze(-1)

    def hole_margin(self, p, R) -> torch.Tensor:
        """Pelvis height above the highest terrain within +-0.3 m: small/negative = fallen into a gap."""
        pts = self._points(p[..., :2], self._yaw2(R), self.hole_grid)
        h = self.hm.query(pts)
        # argmax + gather instead of amax (compiled amax gave NaN gradients)
        return p[..., 2] - h.gather(-1, h.detach().argmax(-1, keepdim=True)).squeeze(-1)

    def wm_features(self, p, R, fp, foot_R) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp: foot sites (..., 2, 3); foot_R: foot rotations (..., 2, 3, 3).

        Returns (features, foot clearance (..., 2), terrain roughness under each sole (..., 2), unsupported sole share
        and 1 - support field under each sole (..., 2, 2)).
        """
        R2 = self._yaw2(R)
        h0 = self.hm.query(p[..., :2])
        hb = (p[..., 2] - h0).unsqueeze(-1)
        # relative heights clamped to +-1 m
        base = (self.hm.query(self._points(p[..., :2], R2, self.base_grid)) - h0.unsqueeze(-1)).clamp(-1.0, 1.0)
        pts = self._points(fp[..., :2], self._yaw2(foot_R), self.foot_grid)  # (..., 2, 15, 2)
        h = self.hm.query(pts)
        h_c = h[..., self.foot_center]
        clear = fp[..., 2] - h_c
        rel = (h - h_c.unsqueeze(-1)).clamp(-1.0, 1.0)
        rough = rel.std(-1)
        # unsupported fraction of the sole: grid points more than ~3 cm off the foot's support height
        rel_s = rel
        if getattr(self.hm, "exact_support", False):
            # exact heights forward, bilinear gradient (bilinear values smear stone edges over a cell)
            h_n = self.hm.query_nearest(pts)
            rel_s = rel + ((h_n - h_n[..., self.foot_center].unsqueeze(-1)).clamp(-1.0, 1.0) - rel).detach()
        unsupported = torch.sigmoid((rel_s.abs() - 0.03) / 0.01).mean(-1)
        # smooth support field: its gradient pulls the soles toward stone centres
        unsupported_field = 1.0 - self.hm.query_support(pts).mean(-1)
        unsupported = torch.stack((unsupported, unsupported_field), -1)
        feats = torch.cat((hb, base, clear, rel.flatten(-2)), -1)
        return feats, clear, rough, unsupported

    def critic_scan(self, p, R) -> torch.Tensor:
        R2 = self._yaw2(R)
        pts = self._points(p[..., :2], R2, self.critic_grid)
        # bilinear, so the value gradient reaches the base position
        return (p[..., 2, None] - self.hm.query(pts) - NOMINAL_HEIGHT).clamp(-1.0, 1.0)

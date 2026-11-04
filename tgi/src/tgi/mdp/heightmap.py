"""Global terrain heightmap: raycast once on the GPU, then queried (bilinear, differentiable) anywhere."""

from __future__ import annotations

import numpy as np
import torch
import warp as wp


class HeightMap:
    def __init__(self, heights: torch.Tensor, origin_xy: tuple[float, float], res: float):
        self.h = heights  # (NX, NY), world z
        self.ox, self.oy = origin_xy
        self.res = res
        self.nx, self.ny = heights.shape

    @classmethod
    def from_trimesh(cls, mesh, device: str, res: float = 0.04, z_top: float = 10.0) -> "HeightMap":
        """Raycast a terrain triangle mesh (Isaac Lab terrain generator output) straight down."""
        lo, hi = mesh.bounds[0, :2], mesh.bounds[1, :2]
        nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
        wm_mesh = wp.Mesh(points=wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device=device),
                          indices=wp.array(np.asarray(mesh.faces, np.int32).flatten(), dtype=wp.int32, device=device))
        out = wp.zeros(nx * ny, dtype=float, device=device)
        wp.launch(_ray_down, dim=nx * ny, inputs=[wm_mesh.id, float(lo[0]), float(lo[1]), res, ny, z_top, out],
                  device=device)
        return cls(wp.to_torch(out).view(nx, ny).clone(), (float(lo[0]), float(lo[1])), res)

    def query(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear terrain height at world xy (..., 2); differentiable w.r.t. xy."""
        u = (xy[..., 0] - self.ox) / self.res
        v = (xy[..., 1] - self.oy) / self.res
        u = u.clamp(0, self.nx - 1.001)
        v = v.clamp(0, self.ny - 1.001)
        i0, j0 = u.floor().long(), v.floor().long()
        fu, fv = u - i0, v - j0
        h00, h10 = self.h[i0, j0], self.h[i0 + 1, j0]
        h01, h11 = self.h[i0, j0 + 1], self.h[i0 + 1, j0 + 1]
        out = (h00 * (1 - fu) + h10 * fu) * (1 - fv) + (h01 * (1 - fu) + h11 * fu) * fv
        hb = getattr(self, "h_blur", None)
        if hb is None:
            return out
        # exact heights forward, gradient (w.r.t. xy) from the Gaussian-smoothed terrain
        b = (hb[i0, j0] * (1 - fu) + hb[i0 + 1, j0] * fu) * (1 - fv) + (hb[i0, j0 + 1] * (1 - fu) + hb[i0 + 1, j0 + 1] * fu) * fv
        return b + (out - b).detach()

    def set_grad_blur(self, sigma: float):
        """Take query() gradients from the terrain blurred by a Gaussian of std ``sigma`` (m); 0 disables."""
        import torch.nn.functional as F

        if sigma <= 0:
            self.h_blur = None
            return
        r = max(1, int(round(3 * sigma / self.res)))
        x = torch.arange(-r, r + 1, device=self.h.device, dtype=torch.float32) * self.res
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = (g / g.sum()).to(self.h.dtype)
        s = F.conv2d(F.pad(self.h[None, None], (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
        s = F.conv2d(F.pad(s, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
        self.h_blur = s[0, 0].contiguous()

    def build_support_field(self, window: float = 0.5, hole_depth: float = 0.25, sigma: float = 0.05):
        """Smooth map in [0, 1]: 1 on supportable surface (within ``hole_depth`` of the local top surface in a
        ``window`` neighbourhood, so stair treads count but gaps between stepping stones do not), blurred by a
        Gaussian of std ``sigma`` so its gradient points toward the middle of supports over a useful band."""
        import torch.nn.functional as F

        k = max(3, int(round(window / self.res)) | 1)
        h = self.h[None, None]
        local_top = F.max_pool2d(h, k, stride=1, padding=k // 2)
        solid = (h > local_top - hole_depth).float()
        r = max(1, int(round(3 * sigma / self.res)))
        x = torch.arange(-r, r + 1, device=h.device, dtype=torch.float32) * self.res
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / g.sum()
        s = F.conv2d(F.pad(solid, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
        s = F.conv2d(F.pad(s, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
        self.support = s[0, 0]

    def query_support(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear support-field value at world xy (differentiable)."""
        u = ((xy[..., 0] - self.ox) / self.res).clamp(0, self.nx - 1.001)
        v = ((xy[..., 1] - self.oy) / self.res).clamp(0, self.ny - 1.001)
        i0, j0 = u.floor().long(), v.floor().long()
        fu, fv = u - i0, v - j0
        s = self.support
        return (s[i0, j0] * (1 - fu) + s[i0 + 1, j0] * fu) * (1 - fv) + (s[i0, j0 + 1] * (1 - fu) + s[i0 + 1, j0 + 1] * fu) * fv

    def query_nearest(self, xy: torch.Tensor) -> torch.Tensor:
        i = ((xy[..., 0] - self.ox) / self.res).round().long().clamp(0, self.nx - 1)
        j = ((xy[..., 1] - self.oy) / self.res).round().long().clamp(0, self.ny - 1)
        return self.h[i, j]


@wp.kernel
def _ray_down(mesh: wp.uint64, x0: float, y0: float, res: float, ny: int, z_top: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    p = wp.vec3(x0 + float(tid // ny) * res, y0 + float(tid % ny) * res, z_top)
    q = wp.mesh_query_ray(mesh, p, wp.vec3(0.0, 0.0, -1.0), 2.0 * z_top)
    if q.result:
        out[tid] = z_top - q.t
    else:
        out[tid] = 0.0

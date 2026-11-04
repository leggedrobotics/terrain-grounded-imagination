"""Stepping stones as in the base env, in strips with a random phase each, so neighbouring strips are staggered.

``stone_rects`` is shared by the training terrain and the sim2sim ``fpmstones`` course.
"""

from __future__ import annotations

import numpy as np


def stone_rects(difficulty: float, size: tuple[float, float], rng: np.random.Generator,
                width_range=(0.5, 0.3), gap_range=(0.1, 0.2), height_limit=(-0.05, 0.05), res: float | None = None,
                width_jitter: float = 0.0, gap_jitter: float = 0.0, transpose: bool = False):
    """Stone rectangles (x0, x1, y0, y1, top_z) covering [0, size[0]] x [0, size[1]]."""
    def lerp(r):
        return r[0] + difficulty * (r[1] - r[0])

    w = max(lerp(width_range) + rng.uniform(-width_jitter, width_jitter), min(width_range))  # jitter never below
    g = min(max(lerp(gap_range) + rng.uniform(-gap_jitter, gap_jitter), 0.0), max(gap_range))  # the hardest level
    hmax = max(lerp(height_limit), 0.0)
    if res:  # base env: int(width / horizontal_scale) pixels
        w, g = max(int(w / res), 1) * res, int(g / res) * res
    sx, sy = (size[1], size[0]) if transpose else size
    rects, y = [], 0.0
    while y < sy - 1e-6:
        y1 = min(sy, y + w)
        x = rng.integers(0, round(w / res)) * res if res else rng.uniform(0.0, w)
        if x - g > 1e-6:  # leading partial stone
            rects.append((0.0, x - g, y, y1, rng.uniform(-hmax, hmax)))
        while x < sx - 1e-6:
            rects.append((x, min(sx, x + w), y, y1, rng.uniform(-hmax, hmax)))
            x += w + g
        y += w + g
    if transpose:
        rects = [(y0, y1, x0, x1, z) for x0, x1, y0, y1, z in rects]
    return rects


def isaac_stepping_stones(**kw):
    """Isaac Lab (trimesh) sub-terrain: eased stones over a pit, with a 2 m start platform."""
    import trimesh
    from isaaclab.terrains import SubTerrainBaseCfg
    from isaaclab.utils import configclass

    def fn(difficulty: float, cfg) -> tuple[list, np.ndarray]:
        rng = np.random.default_rng(np.random.randint(2**31))  # the generator seeds numpy per sub-terrain
        difficulty = difficulty**cfg.progression
        # pit depth: random per tile (only perception sees it)
        depth = rng.uniform(*cfg.holes_depth_range)
        cx, cy, a = cfg.size[0] / 2, cfg.size[1] / 2, cfg.platform_width / 2

        def box(x0, x1, y0, y1, top):
            return trimesh.creation.box((x1 - x0, y1 - y0, top + depth),
                                        trimesh.transformations.translation_matrix(((x0 + x1) / 2, (y0 + y1) / 2, (top - depth) / 2)))

        meshes = [trimesh.creation.box((cfg.size[0], cfg.size[1], 0.1),
                                       trimesh.transformations.translation_matrix((cx, cy, -depth - 0.05))),
                  box(cx - a, cx + a, cy - a, cy + a, 0.0)]
        rects = stone_rects(difficulty, cfg.size, rng, cfg.stone_width_range, cfg.stone_distance_range,
                            cfg.stone_height_limit, width_jitter=cfg.width_jitter, gap_jitter=cfg.gap_jitter,
                            transpose=bool(rng.integers(2)))
        # equal stones per tile; the tile border cuts the outermost ones
        for x0, x1, y0, y1, top in rects:
            if x0 < cx + a and x1 > cx - a and y0 < cy + a and y1 > cy - a:
                top = 0.0  # merged into the platform, as the base env's _center_square
            meshes.append(box(x0, x1, y0, y1, top))
        return meshes, np.array([cx, cy, 0.0])

    @configclass
    class IsaacSteppingStonesCfg(SubTerrainBaseCfg):
        function = fn
        stone_width_range: tuple[float, float] = (0.6, 0.2)
        stone_distance_range: tuple[float, float] = (0.02, 0.25)
        stone_height_limit: tuple[float, float] = (0.0, 0.05)
        holes_depth_range: tuple[float, float] = (0.5, 2.0)  # uniform per tile
        progression: float = 1.5
        width_jitter: float = 0.05
        gap_jitter: float = 0.03
        platform_width: float = 2.0

    return IsaacSteppingStonesCfg(**kw)

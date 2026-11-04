"""TGI's continually trained FPM: the FPM network (``fpm.foot_pred``) with a running input normalizer.

Input: the 70-dim ``system_state``; output: the next touchdown (x, y) of both feet in the pelvis frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fpm.foot_pred import FootPredictor, NormalizedFootPredictor, foot_predictor_mlp

from ..mdp.interface import SYSTEM_STATE_DIM
from ..world_model.networks import RunningNorm


class FootholdPredictor(nn.Module):
    def __init__(self, in_dim: int = SYSTEM_STATE_DIM, hidden=(128, 128)):
        super().__init__()
        self.in_dim, self.hidden = in_dim, tuple(hidden)
        self.net = foot_predictor_mlp(in_dim, 4, self.hidden)
        self.norm = RunningNorm(in_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(..., 70) -> (..., 2, 2) footholds in the root frame."""
        return self.net(self.norm(state)).view(*state.shape[:-1], 2, 2)

    def exportable(self) -> NormalizedFootPredictor:
        """The base env's foot-model format: (B, 70) or (B, 1, 70) -> (B, 4), normalization included."""
        fp = FootPredictor(self.in_dim, 4, self.hidden)
        fp.net.load_state_dict(self.net.state_dict())
        return NormalizedFootPredictor(fp, self.norm.mean.clone(), self.norm.std().clone()).eval()

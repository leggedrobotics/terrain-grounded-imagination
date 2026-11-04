"""Optimizers and the learning-rate schedule."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MuonAdam:
    """Muon on the hidden weight matrices (both sides >= 128), fused Adam on all other parameters, as one optimizer.

    Muon's learning rate is RMS-matched to AdamW (``adjust_lr_fn="match_rms_adamw"``), so both share one schedule.
    """

    def __init__(self, module: nn.Module, lr: float, betas=(0.9, 0.999), muon_lr_mult: float = 1.0):
        params = list(module.parameters())
        self.matrices = [p for p in params if p.ndim == 2 and min(p.shape) >= 128]
        ids = {id(p) for p in self.matrices}
        self.others = [p for p in params if id(p) not in ids]
        self.muon_lr_mult = muon_lr_mult
        self.muon = torch.optim.Muon(self.matrices, lr=lr * muon_lr_mult, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.Adam(self.others, lr=lr, betas=betas, fused=True)
        self.found_inf = None

    def describe(self) -> str:
        n_mat = sum(p.numel() for p in self.matrices) / 1e6
        n_rest = sum(p.numel() for p in self.others) / 1e6
        return f"{n_mat:.2f} M parameters on Muon ({len(self.matrices)} matrices), {n_rest:.2f} M on Adam"

    def set_lr(self, lr: float):
        for g in self.muon.param_groups:
            g["lr"] = lr * self.muon_lr_mult
        for g in self.adam.param_groups:
            g["lr"] = lr

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        if self.found_inf is not None:  # no host sync: Adam skips on the device, Muon sees zeroed gradients
            ok = self.found_inf == 0
            for p in self.matrices:
                if p.grad is not None:
                    p.grad.copy_(torch.where(ok, p.grad, torch.zeros_like(p.grad)))
            self.adam.found_inf = self.found_inf
        self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, sd: dict):
        self.muon.load_state_dict(sd["muon"])
        self.adam.load_state_dict(sd["adam"])


Optimizer = torch.optim.Adam | MuonAdam


def make_optimizer(kind: str, module: nn.Module, lr: float, betas=(0.9, 0.999), muon_lr_mult: float = 1.0) -> Optimizer:
    if kind == "muon":
        return MuonAdam(module, lr, betas, muon_lr_mult)
    return torch.optim.Adam(module.parameters(), lr=lr, betas=betas, fused=True)


def set_lr(opt: Optimizer, lr: float):
    if isinstance(opt, MuonAdam):
        opt.set_lr(lr)
    else:
        for g in opt.param_groups:
            g["lr"] = lr


def cosine_lr(base: float, it: int, decay_iters: int) -> float:
    """Cosine decay from base to 0.1 x base over decay_iters, constant afterwards."""
    if it >= decay_iters:
        return base * 0.1
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / decay_iters)))


def skip_if_nonfinite(opt: Optimizer, grad_norm: torch.Tensor):
    """Make the next step a no-op if the gradient norm is not finite, without a host sync (fused Adam's found_inf)."""
    opt.found_inf = (~torch.isfinite(grad_norm)).float()


def load_optimizer_state(opt: Optimizer, sd: dict, device: str) -> bool:
    """Load a checkpoint's optimizer state; False if it belongs to the other optimizer type (the optimizer starts fresh)."""
    if isinstance(opt, MuonAdam) != ("muon" in sd):
        return False
    opt.load_state_dict(sd)
    adam = opt.adam if isinstance(opt, MuonAdam) else opt
    for g in adam.param_groups:  # a checkpoint's param groups would override the constructor's fused=True
        g["fused"], g["foreach"], g["capturable"] = True, None, False
    for st in adam.state.values():
        if "step" in st:
            st["step"] = torch.as_tensor(st["step"], dtype=torch.float32, device=device)
    return True

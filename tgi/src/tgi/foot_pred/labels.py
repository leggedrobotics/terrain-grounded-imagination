"""Foothold labels from the replay buffer's contact sequences and FK foot positions.

For a foot in stance the label is the touchdown *after* its next lift-off; for a foot in swing, its upcoming touchdown.
"""

from __future__ import annotations

import torch

from ..mdp.mdp import unflatten_state


@torch.no_grad()
def foothold_labels(buffer, kin, chunk: int = 50):
    """Labels (T, N, 2, 2) for the post-transition states s1 of every storage row, validity (T, N), row stamps."""
    T, N, dev = buffer.size, buffer.N, buffer.device
    rows = buffer._row(torch.arange(T, device=dev))
    s1 = buffer.data["s1"][rows]
    c1 = buffer.data["c1"][rows] > 0.5
    done = buffer.data["done"][rows]
    foot = torch.empty(T, N, 2, 3, device=dev)
    for i in range(0, T, chunk):
        S = unflatten_state(s1[i : i + chunk])
        foot[i : i + chunk] = kin.forward(*S)[0]
    latch_next = torch.zeros(N, 2, 3, device=dev)
    latch_future = torch.zeros(N, 2, 3, device=dev)
    ok_next = torch.zeros(N, 2, dtype=torch.bool, device=dev)
    ok_future = torch.zeros(N, 2, dtype=torch.bool, device=dev)
    was_air = torch.zeros(N, 2, dtype=torch.bool, device=dev)
    lab = torch.empty(T, N, 2, 3, device=dev)
    valid = torch.empty(T, N, 2, dtype=torch.bool, device=dev)
    for t in range(T - 1, -1, -1):
        r = done[t].unsqueeze(-1).expand(N, 2)  # s1_t is terminal: nothing after it belongs to the episode
        ok_next &= ~r
        ok_future &= ~r
        was_air &= ~r
        con = c1[t]
        lift = con & was_air
        latch_future = torch.where(lift.unsqueeze(-1), latch_next, latch_future)
        ok_future = torch.where(lift, ok_next, ok_future)
        latch_next = torch.where(con.unsqueeze(-1), foot[t], latch_next)
        ok_next = ok_next | con
        lab[t] = torch.where(con.unsqueeze(-1), latch_future, latch_next)
        valid[t] = torch.where(con, ok_future, ok_next)
        was_air = ~con
    p, R = unflatten_state(s1)[:2]
    rel = lab - p.unsqueeze(-2)
    lab_root = (R.transpose(-1, -2).unsqueeze(-3) @ rel.unsqueeze(-1)).squeeze(-1)[..., :2]  # world_to_root, xy
    lab_rows = torch.zeros(buffer.T, N, 2, 2, device=dev)
    valid_rows = torch.zeros(buffer.T, N, dtype=torch.bool, device=dev)
    lab_rows[rows], valid_rows[rows] = lab_root, valid.all(-1)
    return lab_rows, valid_rows, buffer.row_stamp.clone()

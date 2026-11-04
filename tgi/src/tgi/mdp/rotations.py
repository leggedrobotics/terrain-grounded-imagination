"""Rotation utilities (wxyz quaternions, rotation matrices), all batched and differentiable."""

from __future__ import annotations

import torch


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion -> rotation matrix."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> wxyz quaternion (w >= 0)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp(min=1e-10)) / 2
    qx = torch.sqrt((1 + m00 - m11 - m22).clamp(min=1e-10)) / 2
    qy = torch.sqrt((1 - m00 + m11 - m22).clamp(min=1e-10)) / 2
    qz = torch.sqrt((1 - m00 - m11 + m22).clamp(min=1e-10)) / 2
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    q = torch.stack((qw, qx, qy, qz), -1)
    return q / q.norm(dim=-1, keepdim=True)


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues; smooth at zero."""
    th2 = (v * v).sum(-1, keepdim=True).unsqueeze(-1)
    th = torch.sqrt(th2 + 1e-12)
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
    return eye + a * K + b * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(cos)
    vee = torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1)
    s = torch.where(th < 1e-4, 0.5 + th * th / 12, th / (2 * torch.sin(th)))
    return s.unsqueeze(-1) * vee


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Pure-yaw rotation that shares the heading of R."""
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, i = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, i), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    return R[..., :, :2].flatten(-2)


def axis_angle_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation about a fixed unit axis (3,) by angle (...,)."""
    return so3_exp(axis * angle.unsqueeze(-1))

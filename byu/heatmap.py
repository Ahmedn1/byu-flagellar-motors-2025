"""Gaussian heatmap target construction, CenterNet focal loss, peak extraction."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def make_gaussian_heatmap(h: int, w: int, points: list[tuple[float, float, float]],
                          sigma: float) -> np.ndarray:
    """Render multiple 2D Gaussians on a (h, w) canvas. Each point: (y, x, amp).
    Centers are rounded to integer pixels so the peak gets exactly `amp` —
    matches CenterNet's positive-pixel convention (target == 1 at the peak)."""
    if not points:
        return np.zeros((h, w), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    for py, px, amp in points:
        rpy, rpx = int(round(py)), int(round(px))
        out = np.maximum(out, amp * np.exp(-((ys - rpy)**2 + (xs - rpx)**2) / (2 * sigma * sigma)))
    return out


def focal_heatmap_loss(pred: torch.Tensor, target: torch.Tensor,
                       alpha: float = 2.0, beta: float = 4.0,
                       eps: float = 1e-6) -> torch.Tensor:
    """CenterNet's focal-style heatmap loss (Zhou et al., 2019, 'Objects as Points').

    pred:   (B, 1, H, W) **after sigmoid** in [eps, 1-eps]
    target: (B, 1, H, W) Gaussian-rendered heatmap with peak=1 at object centers.
    Pos pixels = target == 1. Neg pixels = target < 1.
    Normalize by N = number of positive pixels in the batch (clamped to >=1).
    """
    p = pred.clamp(eps, 1 - eps)
    pos = (target >= 0.999).float()   # tolerance for float rounding of integer peak
    neg = 1 - pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def weighted_bce_heatmap_loss(pred: torch.Tensor, target: torch.Tensor,
                              pos_weight: float = 100.0, eps: float = 1e-6) -> torch.Tensor:
    """Alternative to focal: standard BCE with a strong pos_weight to counteract
    the ~147k:1 negative:positive ratio per slice. Treats every pixel's target as
    the desired probability (soft labels via Gaussian heatmap)."""
    p = pred.clamp(eps, 1 - eps)
    pos = (target >= 0.5).float()
    bce = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    w = 1.0 + (pos_weight - 1.0) * pos
    return (bce * w).mean()


def mse_heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE between predicted heatmap and target. Simplest baseline."""
    return ((pred - target) ** 2).mean()


def extract_peak_3d(vol: np.ndarray) -> tuple[tuple[int, int, int], float]:
    """Return (z, y, x) of argmax and its value."""
    idx = int(vol.argmax())
    z, y, x = np.unravel_index(idx, vol.shape)
    return (int(z), int(y), int(x)), float(vol[z, y, x])

"""Tomogram volume loader + resampler to a common physical voxel size (A/voxel).

Design:
- A raw tomogram is a directory of `slice_NNNN.jpg` files (one per z).
- Loading: stack all slices into uint8 (Z, Y, X).
- Resampling: zoom by (orig_spacing / target_spacing) on every axis (isotropic
  physical scale). Motor's apparent voxel size becomes constant across tomograms.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from . import TRAIN_DIR


@dataclass
class TomoMeta:
    tomo_id: str
    orig_shape: tuple[int, int, int]  # (Z, Y, X)
    orig_spacing: float                # angstrom / voxel (original)
    target_spacing: Optional[float]    # angstrom / voxel after resample (None = unchanged)
    shape: tuple[int, int, int]        # final (Z, Y, X) after resample
    scale: tuple[float, float, float]  # final_axis / orig_axis (per axis)


def _list_slices(tomo_dir: Path) -> list[Path]:
    return sorted(tomo_dir.glob("slice_*.jpg"))


def load_raw(tomo_dir: Path | str) -> np.ndarray:
    """Load a tomogram as uint8 (Z, Y, X). No resampling."""
    tomo_dir = Path(tomo_dir)
    paths = _list_slices(tomo_dir)
    if not paths:
        raise FileNotFoundError(f"no slice_*.jpg in {tomo_dir}")
    first = np.asarray(Image.open(paths[0]).convert("L"))
    vol = np.empty((len(paths), *first.shape), dtype=np.uint8)
    vol[0] = first
    for i, p in enumerate(paths[1:], 1):
        vol[i] = np.asarray(Image.open(p).convert("L"))
    return vol


def resample_to_spacing(vol: np.ndarray, orig_spacing: float, target_spacing: float,
                        order: int = 1) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Resample isotropically to target physical spacing.
    factor = orig_spacing / target_spacing (>1 upsample, <1 downsample).
    Returns (vol_resampled, scale_per_axis). Uses linear interpolation by default."""
    f = orig_spacing / target_spacing
    out = zoom(vol, (f, f, f), order=order, prefilter=(order > 1))
    sc = tuple(out.shape[i] / vol.shape[i] for i in range(3))
    if out.dtype != vol.dtype:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out, sc  # type: ignore[return-value]


def load_volume(tomo_id: str, orig_spacing: float, target_spacing: Optional[float] = None,
                root: Path = TRAIN_DIR) -> tuple[np.ndarray, TomoMeta]:
    """High-level: load tomogram and (optionally) resample to target_spacing."""
    vol = load_raw(root / tomo_id)
    if target_spacing is None or abs(target_spacing - orig_spacing) < 1e-6:
        meta = TomoMeta(tomo_id, vol.shape, orig_spacing, target_spacing,
                        vol.shape, (1.0, 1.0, 1.0))
        return vol, meta
    vol2, sc = resample_to_spacing(vol, orig_spacing, target_spacing)
    meta = TomoMeta(tomo_id, vol.shape, orig_spacing, target_spacing, vol2.shape, sc)
    return vol2, meta


def transform_point(zyx: tuple[float, float, float], meta: TomoMeta) -> tuple[float, float, float]:
    """Map a motor coord in the *original* voxel frame to the resampled frame."""
    sz, sy, sx = meta.scale
    return zyx[0] * sz, zyx[1] * sy, zyx[2] * sx


def inverse_transform_point(zyx: tuple[float, float, float], meta: TomoMeta) -> tuple[float, float, float]:
    """Map a motor coord from resampled frame back to original (submission) frame."""
    sz, sy, sx = meta.scale
    return zyx[0] / sz, zyx[1] / sy, zyx[2] / sx

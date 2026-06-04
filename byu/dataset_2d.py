"""2D-slice torch Dataset.

Each item: (img CxHxW float in [0,1], heatmap 1xHxW float in [0,1]).
Sampling: per tomogram, draw `pos_per_tomo` slices near each motor's z* and
`neg_per_tomo` empty slices uniformly elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import LABELS_CSV, PREPROCESSED
from .heatmap import make_gaussian_heatmap


@dataclass
class DS2DConfig:
    target_spacing: float = 32.0
    sigma_xy: float = 3.0        # Gaussian sigma in resampled voxels
    sigma_z: float = 4.0         # z-falloff sigma in resampled voxels
    crop: int = 384              # random crop size (HxW)
    pos_per_tomo: int = 4
    neg_per_tomo: int = 4
    in_channels: int = 3         # 1 (replicate) or 3/5/7/... (z stack)
    augment: bool = True


class Slices2D(Dataset):
    def __init__(self, tomo_ids: list[str], cfg: DS2DConfig,
                 labels_csv: Path = LABELS_CSV, cache_dir: Path = PREPROCESSED,
                 mode: str = "train"):
        self.cfg = cfg
        self.mode = mode
        df = pd.read_csv(labels_csv)
        meta = pd.read_csv(cache_dir / "preprocessed_meta.csv")
        # build per-tomo metadata in the RESAMPLED frame
        self.tomos: list[dict] = []
        for tid in tomo_ids:
            mrow = meta[meta.tomo_id == tid]
            if mrow.empty: continue
            mrow = mrow.iloc[0]
            sp = float(mrow.orig_spacing)
            scale = sp / float(mrow.target_spacing)  # orig -> resampled
            rows = df[(df.tomo_id == tid)]
            n_motors = int(rows["Number of motors"].iloc[0])
            motors_resampled = []
            if n_motors > 0:
                for _, r in rows[rows["Number of motors"] > 0].iterrows():
                    motors_resampled.append((float(r["Motor axis 0"]) * scale,
                                             float(r["Motor axis 1"]) * scale,
                                             float(r["Motor axis 2"]) * scale))
            self.tomos.append(dict(tid=tid, motors=motors_resampled, n=n_motors,
                                   shape=(int(mrow.shape_z), int(mrow.shape_y), int(mrow.shape_x)),
                                   orig_spacing=sp, target_spacing=float(mrow.target_spacing),
                                   scale=scale))
        self.cache_dir = cache_dir
        self._index = self._build_index()

    def _build_index(self) -> list[tuple[int, int]]:
        """Return (tomo_idx, slice_z) pairs."""
        rng = np.random.default_rng(0)
        idx: list[tuple[int, int]] = []
        for ti, t in enumerate(self.tomos):
            Z = t["shape"][0]
            zs = set()
            if self.mode == "train":
                # positive picks: ALWAYS include the integer-rounded motor z*
                # (guarantees at least one full-amp positive pixel),
                # then add `pos_per_tomo - 1` random offsets within +/- 1.5 sigma_z.
                for mz, _, _ in t["motors"]:
                    z_exact = int(round(mz))
                    if 0 <= z_exact < Z: zs.add(z_exact)
                    lo, hi = max(0, int(mz - self.cfg.sigma_z * 1.5)), min(Z - 1, int(mz + self.cfg.sigma_z * 1.5))
                    for _ in range(max(0, self.cfg.pos_per_tomo - 1)):
                        zs.add(int(rng.integers(lo, hi + 1)))
                # negative picks: anywhere (will be 0-target if far from any motor)
                for _ in range(self.cfg.neg_per_tomo):
                    zs.add(int(rng.integers(0, Z)))
            else:  # val: every slice (used in prediction)
                zs = set(range(Z))
            for z in zs:
                idx.append((ti, z))
        return idx

    def reshuffle(self, epoch: int):
        """Reroll which slices are sampled this epoch."""
        if self.mode != "train": return
        rng = np.random.default_rng(epoch + 1)
        idx: list[tuple[int, int]] = []
        for ti, t in enumerate(self.tomos):
            Z = t["shape"][0]
            zs = set()
            for mz, _, _ in t["motors"]:
                z_exact = int(round(mz))
                if 0 <= z_exact < Z: zs.add(z_exact)
                lo, hi = max(0, int(mz - self.cfg.sigma_z * 1.5)), min(Z - 1, int(mz + self.cfg.sigma_z * 1.5))
                for _ in range(max(0, self.cfg.pos_per_tomo - 1)):
                    zs.add(int(rng.integers(lo, hi + 1)))
            for _ in range(self.cfg.neg_per_tomo):
                zs.add(int(rng.integers(0, Z)))
            for z in zs:
                idx.append((ti, z))
        rng.shuffle(idx)
        self._index = idx

    def __len__(self):
        return len(self._index)

    def _load_slice(self, vol_mm: np.ndarray, z: int) -> np.ndarray:
        """Return (C, H, W) float32 normalized to [0,1]. C from cfg.in_channels.
        For C>1: stack slices [z - C//2 .. z + C//2] with edge-clipped neighbors."""
        Z = vol_mm.shape[0]
        C = self.cfg.in_channels
        if C == 1:
            sl = vol_mm[z].astype(np.float32) / 255.0
            return sl[None]
        half = C // 2
        zs = [min(max(0, z + d), Z - 1) for d in range(-half, -half + C)]
        return np.stack([vol_mm[zi] for zi in zs]).astype(np.float32) / 255.0

    def __getitem__(self, i: int):
        ti, z = self._index[i]
        t = self.tomos[ti]
        vol = np.load(self.cache_dir / f"{t['tid']}.npy", mmap_mode="r")
        img = self._load_slice(vol, z)        # (C, H, W)
        H, W = img.shape[1], img.shape[2]
        # build target heatmap: each motor whose z* is near z contributes a 2D Gaussian
        amps_pts = []
        for mz, my, mx in t["motors"]:
            amp = float(np.exp(-((z - mz) ** 2) / (2 * self.cfg.sigma_z ** 2)))
            if amp > 0.05:
                amps_pts.append((my, mx, amp))
        tgt = make_gaussian_heatmap(H, W, amps_pts, self.cfg.sigma_xy)  # (H, W)
        # crop / pad to cfg.crop
        ch, cw = self.cfg.crop, self.cfg.crop

        def crop_pad(x: np.ndarray, ch: int, cw: int, top: int, left: int) -> np.ndarray:
            is_2d = x.ndim == 2
            Hh, Ww = x.shape[-2:]
            out_shape = (ch, cw) if is_2d else (x.shape[0], ch, cw)
            out = np.zeros(out_shape, dtype=x.dtype)
            sy0 = max(0, top); sx0 = max(0, left)
            sy1 = min(Hh, top + ch); sx1 = min(Ww, left + cw)
            if sy1 <= sy0 or sx1 <= sx0: return out
            dy0 = max(0, -top); dx0 = max(0, -left)
            dy1 = dy0 + (sy1 - sy0); dx1 = dx0 + (sx1 - sx0)
            if is_2d: out[dy0:dy1, dx0:dx1] = x[sy0:sy1, sx0:sx1]
            else:     out[:, dy0:dy1, dx0:dx1] = x[:, sy0:sy1, sx0:sx1]
            return out

        # random crop centered around motor (if any) or random
        rng = np.random.default_rng((ti * 1000 + z) if self.mode != "train" else None)
        if self.mode == "train" and amps_pts and rng.random() < 0.7:
            cy, cx = amps_pts[0][0], amps_pts[0][1]
            top = int(cy - ch / 2 + rng.integers(-ch // 4, ch // 4 + 1))
            left = int(cx - cw / 2 + rng.integers(-cw // 4, cw // 4 + 1))
        else:
            top = int(rng.integers(0, max(1, H - ch + 1))) if H >= ch else (H - ch) // 2
            left = int(rng.integers(0, max(1, W - cw + 1))) if W >= cw else (W - cw) // 2
        img = crop_pad(img, ch, cw, top, left)
        tgt = crop_pad(tgt, ch, cw, top, left)
        # geometric augment: horizontal/vertical flip (matches the flip TTA used at inference)
        if self.mode == "train" and self.cfg.augment:
            if rng.random() < 0.5: img = img[:, :, ::-1].copy(); tgt = tgt[:, ::-1].copy()
            if rng.random() < 0.5: img = img[:, ::-1, :].copy(); tgt = tgt[::-1, :].copy()
        return (torch.from_numpy(img.astype(np.float32)),
                torch.from_numpy(tgt.astype(np.float32))[None])  # add channel dim

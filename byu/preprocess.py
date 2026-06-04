"""One-time preprocessing: load each tomogram, resample to TARGET_SPACING,
and save as a compact uint8 .npy under data/preprocessed/<tomo_id>.npy.
Subsequent training reads the .npy directly (fast, no JPEG decode).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm

from . import TRAIN_DIR, PREPROCESSED, LABELS_CSV
from .data import load_raw, resample_to_spacing


DEFAULT_TARGET_SPACING = 32.0   # angstrom / voxel  (motor ~16 vox, tau ~31 vox; cache ~14 GB)


def preprocess_one(tomo_id: str, orig_spacing: float, target_spacing: float,
                   src_root: Path = TRAIN_DIR, dst_root: Path = PREPROCESSED,
                   overwrite: bool = False) -> Path:
    dst_root.mkdir(parents=True, exist_ok=True)
    out = dst_root / f"{tomo_id}.npy"
    if out.exists() and not overwrite:
        return out
    vol = load_raw(src_root / tomo_id)
    if abs(target_spacing - orig_spacing) > 1e-6:
        vol, _ = resample_to_spacing(vol, orig_spacing, target_spacing, order=1)
    np.save(out, vol)
    return out


def preprocess_all(target_spacing: float = DEFAULT_TARGET_SPACING,
                   labels_csv: Path = LABELS_CSV,
                   src_root: Path = TRAIN_DIR,
                   dst_root: Path = PREPROCESSED,
                   overwrite: bool = False) -> Path:
    df = pd.read_csv(labels_csv)
    per = (df.groupby("tomo_id").agg({"Voxel spacing": "first"}).reset_index())
    dst_root.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for _, r in tqdm(per.iterrows(), total=len(per), desc=f"preprocess @ {target_spacing}A"):
        tid, sp = r["tomo_id"], float(r["Voxel spacing"])
        p = preprocess_one(tid, sp, target_spacing, src_root, dst_root, overwrite)
        arr = np.load(p, mmap_mode="r")
        meta_rows.append({"tomo_id": tid, "orig_spacing": sp,
                          "target_spacing": target_spacing,
                          "shape_z": arr.shape[0], "shape_y": arr.shape[1], "shape_x": arr.shape[2]})
    meta = pd.DataFrame(meta_rows)
    meta_path = dst_root / "preprocessed_meta.csv"
    meta.to_csv(meta_path, index=False)
    return meta_path

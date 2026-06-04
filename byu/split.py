"""Deterministic stratified train/val split for the BYU competition.
Stratifies on motor-count bucket x voxel-spacing bucket.
Writes data/split_v1.csv with columns: tomo_id, fold ('train'|'val').
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from . import LABELS_CSV, DATA


def _motor_bucket(n: int) -> str:
    return "0" if n == 0 else ("1" if n == 1 else "2+")


def _spacing_bucket(s: float) -> str:
    # group the discrete spacings into a few coarse buckets
    if s < 10: return "lo"          # ~6.5
    if s < 14: return "mid"         # 13.x
    if s < 17: return "hi"          # 15.6/16.x
    return "vhi"                    # 19.x


def build_split(val_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV)
    per = (df.groupby("tomo_id")
             .agg({"Number of motors": "first", "Voxel spacing": "first"})
             .reset_index())
    per["mb"] = per["Number of motors"].apply(_motor_bucket)
    per["sb"] = per["Voxel spacing"].apply(_spacing_bucket)
    per["strat"] = per["mb"] + "|" + per["sb"]
    # train_test_split with stratify; fall back if a bucket has <2
    counts = per["strat"].value_counts()
    rare = set(counts[counts < 2].index)
    if rare:
        per.loc[per["strat"].isin(rare), "strat"] = "rare"
    tr, va = train_test_split(per, test_size=val_frac, random_state=seed,
                              stratify=per["strat"])
    out = pd.DataFrame({"tomo_id": list(tr["tomo_id"]) + list(va["tomo_id"]),
                        "fold": ["train"] * len(tr) + ["val"] * len(va)})
    out = out.sort_values("tomo_id").reset_index(drop=True)
    return out


def save_split(path: Path | None = None) -> Path:
    out = build_split()
    if path is None:
        path = DATA / "split_v1.csv"
    out.to_csv(path, index=False)
    return path

"""Re-implementation of the competition F_beta=2 metric with tau=1000A tolerance.

Per the host's evaluation page:
- Per tomogram, predictions are one (z,y,x) or "no motor" (we encode as None or -1,-1,-1).
- TP if predicted within tau of (any) GT motor.
- FN if GT has a motor and prediction is absent OR predicted wrongly (>tau).
- FP if GT has no motor and a prediction is given.
- F_beta=2: F = (1+b^2)*TP / ((1+b^2)*TP + b^2*FN + FP), b=2 -> 5*TP/(5*TP+4*FN+FP).

Distance is Euclidean in physical units (angstroms). Coordinates given in *voxel*
indices are multiplied by the per-tomogram voxel spacing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd

from . import TAU_ANGSTROM


@dataclass
class MetricCounts:
    tp: int; fp: int; fn: int
    def fbeta(self, beta: float = 2.0) -> float:
        b2 = beta * beta
        num = (1 + b2) * self.tp
        den = (1 + b2) * self.tp + b2 * self.fn + self.fp
        return float(num / den) if den > 0 else 0.0


def _coerce_pred(row) -> Optional[tuple[float, float, float]]:
    """Official spec: abstain iff *any* axis equals exactly -1 (the sentinel).
    A negative-but-not--1 coord is a (bad) prediction, not an abstain."""
    z, y, x = float(row["Motor axis 0"]), float(row["Motor axis 1"]), float(row["Motor axis 2"])
    if z == -1 or y == -1 or x == -1:
        return None
    return z, y, x


def score(pred_df: pd.DataFrame, gt_df: pd.DataFrame, tau_angstrom: float = TAU_ANGSTROM,
          beta: float = 2.0, return_details: bool = False):
    """Score predictions against ground truth.
    pred_df: one row per tomo (tomo_id, Motor axis 0/1/2). -1 means abstain.
    gt_df:   train_labels.csv format; 0+ rows per tomo, with Voxel spacing column.
              Tomograms with 0 motors must still appear (a single row with -1 coords,
              as in the supplied labels file).
    Returns either F_beta (float) or (F_beta, MetricCounts, per_tomo_df)."""
    # group GT by tomo
    gt_by = {tid: g for tid, g in gt_df.groupby("tomo_id")}
    rows = []
    for _, p in pred_df.iterrows():
        tid = p["tomo_id"]
        if tid not in gt_by:
            continue  # tomo not in GT (e.g. real test); skip in offline scoring
        g = gt_by[tid]
        spacing = float(g["Voxel spacing"].iloc[0])
        n_motors = int(g["Number of motors"].iloc[0])
        pred = _coerce_pred(p)
        kind = ""
        dist_min = float("nan")
        if n_motors == 0:
            if pred is None:
                kind = "TN"
            else:
                kind = "FP"
        else:
            gts = g[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].to_numpy(dtype=float)
            if pred is None:
                kind = "FN"
            else:
                d_vox = np.linalg.norm(gts - np.asarray(pred), axis=1)
                d_ang = d_vox * spacing
                dist_min = float(d_ang.min())
                kind = "TP" if dist_min <= tau_angstrom else "FN"
        rows.append({"tomo_id": tid, "n_motors": n_motors, "kind": kind,
                     "dist_min_A": dist_min, "spacing": spacing})

    detail = pd.DataFrame(rows)
    c = MetricCounts(tp=int((detail.kind == "TP").sum()),
                     fp=int((detail.kind == "FP").sum()),
                     fn=int((detail.kind == "FN").sum()))
    fb = c.fbeta(beta)
    return (fb, c, detail) if return_details else fb


def submission_from_predictions(preds: Iterable[tuple[str, Optional[tuple[float, float, float]]]]
                                ) -> pd.DataFrame:
    """Build a Kaggle-style submission DataFrame from (tomo_id, (z,y,x) or None)."""
    rows = []
    for tid, p in preds:
        if p is None:
            rows.append({"tomo_id": tid, "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1})
        else:
            rows.append({"tomo_id": tid, "Motor axis 0": p[0], "Motor axis 1": p[1], "Motor axis 2": p[2]})
    return pd.DataFrame(rows)

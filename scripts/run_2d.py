"""Train the 2D U-Net heatmap detector for the BYU flagellar-motors task.

Per epoch we sample slices near each motor + some random empty slices, build a
per-slice Gaussian heatmap target, and minimize a pos-weighted BCE loss with
the model's sigmoid output. Validation runs the full per-slice prediction on
each val tomogram, stacks the per-slice heatmaps into a 3D volume, smooths,
and takes the global argmax as the predicted motor coordinate (in the
resampled frame, then back-transformed to the original-voxel frame for
scoring).

We keep two checkpoints per run:
  - ckpt_last.pt: weights at the end of training
  - ckpt_best.pt: weights at the epoch with the best tuned-threshold val F_beta

Last stdout line is `F_BETA=<float>`.

Usage:
    PYTHONPATH=. python scripts/run_2d.py --cfg configs/recipe.py --exp_dir runs/run0
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
import segmentation_models_pytorch as smp

from byu import LABELS_CSV, PREPROCESSED
from byu.dataset_2d import Slices2D, DS2DConfig
from byu.heatmap import extract_peak_3d
from byu.metric import score, submission_from_predictions


def load_cfg(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.CFG


def build_model(cfg: dict) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=cfg.get("in_channels", 7),
        classes=1,
    )


@torch.no_grad()
def predict_one(model, vol: np.ndarray, in_channels: int, device: str,
                batch: int = 8) -> np.ndarray:
    """Return per-slice 2D sigmoid heatmaps stacked as (Z, H, W) float32."""
    Z, H, W = vol.shape
    out = np.zeros((Z, H, W), dtype=np.float32)
    for z0 in range(0, Z, batch):
        z1 = min(Z, z0 + batch)
        batch_imgs = []
        for z in range(z0, z1):
            if in_channels == 1:
                batch_imgs.append(vol[z].astype(np.float32)[None] / 255.0)
            else:
                half = in_channels // 2
                zs = [min(max(0, z + d), Z - 1) for d in range(-half, -half + in_channels)]
                batch_imgs.append(np.stack([vol[zi] for zi in zs]).astype(np.float32) / 255.0)
        x = torch.from_numpy(np.stack(batch_imgs)).to(device)
        y = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
        out[z0:z1] = y
    return out


def predict_all_val(model, val_ids: list[str], meta_df: pd.DataFrame, cfg: dict,
                    device: str, peaks_out: list | None = None
                    ) -> list[tuple[str, tuple[float, float, float] | None]]:
    """Run model on every val tomogram. Returns list of (tid, pred_in_ORIGINAL_frame or None).
    If peaks_out is given, append (tid, peak_score, (z_orig, y_orig, x_orig)) for offline
    threshold tuning."""
    thr = cfg.get("presence_threshold", 0.5)
    smooth = cfg.get("peak_smooth_sigma", 1.0)
    in_ch = cfg.get("in_channels", 7)
    preds = []
    for tid in val_ids:
        mrow = meta_df[meta_df.tomo_id == tid].iloc[0]
        scale = float(mrow.orig_spacing) / float(mrow.target_spacing)
        vol = np.load(PREPROCESSED / f"{tid}.npy", mmap_mode="r")
        heat = predict_one(model, vol, in_ch, device)
        if smooth > 0:
            heat = gaussian_filter(heat, (smooth, smooth, smooth))
        (z, y, x), peak = extract_peak_3d(heat)
        z_o, y_o, x_o = z / scale, y / scale, x / scale
        if peaks_out is not None:
            peaks_out.append((tid, float(peak), z_o, y_o, x_o))
        if peak < thr:
            preds.append((tid, None))
        else:
            preds.append((tid, (z_o, y_o, x_o)))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    exp_dir = Path(args.exp_dir); exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "cfg_snapshot.json").write_text(json.dumps(
        {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}, indent=2))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0))

    # data
    labels = pd.read_csv(LABELS_CSV)
    split = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "split_v1.csv")
    train_ids = split[split.fold == "train"].tomo_id.tolist()
    val_ids = split[split.fold == "val"].tomo_id.tolist()
    meta_df = pd.read_csv(PREPROCESSED / "preprocessed_meta.csv")
    ds_cfg = DS2DConfig(
        sigma_xy=cfg.get("sigma_xy", 3.0),
        sigma_z=cfg.get("sigma_z", 4.0),
        crop=cfg.get("crop", 384),
        pos_per_tomo=cfg.get("pos_per_tomo", 4),
        neg_per_tomo=cfg.get("neg_per_tomo", 4),
        in_channels=cfg.get("in_channels", 7),
        augment=cfg.get("augment", True),
    )
    tr_ds = Slices2D(train_ids, ds_cfg, mode="train")
    tr_dl = DataLoader(tr_ds, batch_size=cfg.get("batch_size", 8), shuffle=True,
                       num_workers=cfg.get("num_workers", 2), pin_memory=True, drop_last=True)

    # model + opt
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.get("epochs", 10))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    t0 = time.time()
    best_fb = -1.0
    best_meta = {}
    for epoch in range(cfg.get("epochs", 10)):
        tr_ds.reshuffle(epoch)
        model.train()
        losses = []
        for x, y in tr_dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = model(x)
                # weighted BCE with logits — numerically stable, pos_weight upweights
                # the rare positive pixels (target >= 0.5 considered "positive").
                pos = (y >= 0.5).float()
                w = 1.0 + (cfg.get("bce_pos_weight", 200.0) - 1.0) * pos
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none") * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        print(f"[epoch {epoch}] loss={np.mean(losses):.4f} dt={time.time()-t0:.0f}s", flush=True)

        # eval at end of every epoch
        model.eval()
        peaks = []
        preds = predict_all_val(model, val_ids, meta_df, cfg, device, peaks_out=peaks)
        sub = submission_from_predictions(preds)
        gt_val = labels[labels.tomo_id.isin(val_ids)]
        fb, c, det = score(sub, gt_val, return_details=True)
        # offline threshold sweep — pick the best operating point for THIS epoch's peaks
        best_thr, best_thr_fb, best_thr_counts = 0.0, fb, (c.tp, c.fp, c.fn)
        for thr in np.arange(0.005, 0.95, 0.01):
            sub2 = submission_from_predictions([(t, None if pk < thr else (zo, yo, xo))
                                                 for (t, pk, zo, yo, xo) in peaks])
            fb2, c2, _ = score(sub2, gt_val, return_details=True)
            if fb2 > best_thr_fb:
                best_thr_fb, best_thr, best_thr_counts = fb2, thr, (c2.tp, c2.fp, c2.fn)
        print(f"[epoch {epoch}] F_beta(thr={cfg.get('presence_threshold', 0.05)})={fb:.4f}  "
              f"TP={c.tp} FP={c.fp} FN={c.fn}", flush=True)
        print(f"[epoch {epoch}] best_thr={best_thr:.3f} -> F={best_thr_fb:.4f} "
              f"TP/FP/FN={best_thr_counts[0]}/{best_thr_counts[1]}/{best_thr_counts[2]}", flush=True)

        torch.save(model.state_dict(), exp_dir / "ckpt_last.pt")
        if best_thr_fb > best_fb:
            best_fb = best_thr_fb
            best_meta = dict(epoch=epoch, tp=best_thr_counts[0], fp=best_thr_counts[1],
                              fn=best_thr_counts[2], best_thr=float(best_thr))
            torch.save(model.state_dict(), exp_dir / "ckpt_best.pt")
            sub.to_csv(exp_dir / "predictions_best.csv", index=False)
            det.to_csv(exp_dir / "detail_best.csv", index=False)

    dur = time.time() - t0
    summary = dict(best_fbeta=best_fb, best=best_meta, duration_s=dur,
                   cfg={k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))})
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DURATION_S={dur:.0f}")
    print(f"TP_FP_FN={best_meta.get('tp', 0)}/{best_meta.get('fp', 0)}/{best_meta.get('fn', 0)}")
    print(f"F_BETA={best_fb:.6f}")


if __name__ == "__main__":
    main()

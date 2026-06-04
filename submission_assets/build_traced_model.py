"""Trace the trained 2D U-Net checkpoint into TorchScript.

Why: at submission time we want to load the model without segmentation_models_pytorch
(saves a pip install in a no-internet Kaggle environment). TorchScript bundles the
weights + the forward graph into a single self-contained file.

Output:
    submission_assets/model_a.ts         — TorchScript model (~98 MB)
    submission_assets/model_a.meta.json  — small JSON config for the notebook

Usage:
    PYTHONPATH=. python submission_assets/build_traced_model.py \
        --ckpt runs/run0/ckpt_best.pt --out_dir submission_assets
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import segmentation_models_pytorch as smp


# These values MUST match the training recipe in configs/recipe.py.
ENCODER = "resnet34"
IN_CHANNELS = 7
TARGET_SPACING = 32.0          # Å/voxel
PEAK_SMOOTH_SIGMA = 1.0        # 3D Gaussian smoothing of stacked heatmap before argmax
# Inference threshold for 3-way TTA on the val set (sweep result: F=0.9103 at thr=0.65).
PRESENCE_THRESHOLD_3WAY_TTA = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt_best.pt from training")
    ap.add_argument("--out_dir", default="submission_assets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_ts = out_dir / "model_a.ts"
    out_meta = out_dir / "model_a.meta.json"

    print(f"loading {args.ckpt}")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,            # weights come from the checkpoint
        in_channels=IN_CHANNELS,
        classes=1,
    )
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # The model is fully convolutional, so TorchScript will accept any (B, 7, H, W)
    # at runtime — the example size is just for tracing.
    example = torch.randn(1, IN_CHANNELS, 480, 480)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        # Sanity-check shape-agnostic tracing.
        out = traced(torch.randn(2, IN_CHANNELS, 384, 384))
        assert out.shape == (2, 1, 384, 384), f"unexpected output shape: {out.shape}"
    traced.save(str(out_ts))

    meta = {
        "in_channels": IN_CHANNELS,
        "target_spacing": TARGET_SPACING,
        "presence_threshold": PRESENCE_THRESHOLD_3WAY_TTA,
        "peak_smooth_sigma": PEAK_SMOOTH_SIGMA,
        "tta_augs": ["identity", "hflip", "vflip"],
        "source_ckpt": str(Path(args.ckpt).resolve()),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"saved: {out_ts}  ({out_ts.stat().st_size / 1e6:.1f} MB)")
    print(f"saved: {out_meta}")


if __name__ == "__main__":
    main()

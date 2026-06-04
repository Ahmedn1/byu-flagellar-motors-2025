"""Preprocess all tomograms -> data/preprocessed/<id>.npy at TARGET_SPACING."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from byu.preprocess import preprocess_all, DEFAULT_TARGET_SPACING

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=DEFAULT_TARGET_SPACING,
                    help="target physical spacing in A/voxel")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    p = preprocess_all(target_spacing=args.spacing, overwrite=args.overwrite)
    print("meta ->", p)

"""Training recipe — the configuration that produced the submitted model.

resnet34 U-Net (via segmentation_models_pytorch), ImageNet-pretrained encoder.
Input is a 7-channel z-stack of slices: [z-3, z-2, z-1, z, z+1, z+2, z+3].
Target heatmap is a per-slice 2D Gaussian (sigma_xy=3 voxels in the 32 Å frame)
attenuated by a z-falloff Gaussian (sigma_z=4 voxels) centered on each motor's z*.

Why these numbers:
- BCE with pos_weight=200 — the slice has ~1 positive pixel per ~150k. Standard
  BCE collapses to "predict 0 everywhere"; pos_weight rebalances so the rare
  positive pixel's gradient survives. Focal loss (CenterNet style) collapses
  here for the same reason — focal's downweighting of "easy negatives" reaches
  zero gradient when there's no class imbalance for it to exploit.
- in_channels=7 (z-context ±3) — recovered 4 missed motors over 3-channel z-context.
- pos_per_tomo=8 — denser positive sampling, +0.015 F over pos=4.
- crop=384 — random 384x384 windows. Smaller than the full slice → acts as data
  augmentation; full-slice training (crop=480) was tested and gave -0.020.
- 10 epochs / cosine LR — sweet spot for this batch_size × dataset size.
"""
CFG = dict(
    seed=0,
    encoder="resnet34",
    encoder_weights="imagenet",
    in_channels=7,
    sigma_xy=3.0,
    sigma_z=4.0,
    crop=384,
    pos_per_tomo=8,
    neg_per_tomo=4,
    augment=True,
    batch_size=8,
    num_workers=2,
    lr=3e-4,
    weight_decay=1e-4,
    epochs=10,
    bce_pos_weight=200.0,
    # used during the in-loop validation pass (rough). The actual submission
    # threshold is picked from the val sweep + TTA inference (see notebook).
    presence_threshold=0.05,
    peak_smooth_sigma=1.0,
)

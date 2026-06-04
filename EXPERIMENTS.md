# Experiments

A distilled log of what I tried, what I learned, and what I'd do differently. The full raw log lives in `.lab/log.md` in my working tree; this file is the version I'd want a code reviewer to see.

## Approach exploration (early phase)

Three modeling families were tried before settling on the 2D U-Net heatmap:

| Approach | Best local val F | Why I picked / dropped it |
|---|---|---|
| 2D U-Net heatmap | **0.889** | Fast inference, well-suited to the per-slice z-context formulation. **Winner.** |
| 3D BasicUNet heatmap (96×192×192 crops, sliding window) | 0.842 | Better val precision than 2D (FP=14 vs 18), but sliding-window inference projected to 30+ hours on Kaggle T4 — over the 12 h cap. |
| 3D ResNet classifier + presence head | 0.062 | Per-window presence classifier collapsed to "always yes" because training had 67% positive crops per motor-tomo vs ~2% at inference. Wrong primitive for one-target-per-tomo detection. |

## Knobs that moved the 2D recipe

In rough order of impact:

| Change | Δ val F | Notes |
|---|---|---|
| Loss: focal → **BCE with pos_weight=200** | **+0.77** | Focal collapsed under 1:150k pos/neg ratio (its negative-loss term goes to ~0 gradient). Pos-weighted BCE keeps the positive pixel's gradient alive. |
| In-channels: 3 → **7 (z-context ±3 slices)** | +0.018 | Recovered 4 missed motors. First conv mixes z-neighbors; subsequent layers are 2D. |
| pos_per_tomo: 4 → **8** | +0.015 | Denser positive supervision per epoch. Compositional with pos_weight. |
| Backbone: resnet18 → **resnet34** | +0.011 | Modest. resnet50 hurt (worse precision). |
| **3-way TTA at inference** | +0.011 over single-pass | Identity + h-flip + v-flip. 4-way (adding h+v) was *worse* — both-flip is too close to identity. |
| Crop: 384 → 480 (full slice) | −0.020 | 384 random crops act as augmentation. |

## Knobs that didn't move it

| Change | Δ val F | Diagnosed cause |
|---|---|---|
| EMA (decay 0.999) | −0.006 (noise) | Cosine LR decays late-stage updates to ~zero; EMA shadow doesn't gain wider-minimum benefit. |
| SWA over last 3/5/7 epochs | −0.010 to −0.040 | Same as EMA: late checkpoints are nearly the same point in weight space. |
| Greedy soup (Wortsman 2022) over per-epoch checkpoints | −0.000 (every addition rejected) | Same cause. The straight line between any two of our late checkpoints passes through a small ridge instead of a shared basin — i.e. no linear mode connectivity in this trained model. |
| Heavy aug (gamma+intensity+noise+zoom, p=0.5-0.7) | −0.044 | Aug too strong. Submitted to private: scored 0.451 (−0.116 from baseline). |
| Zoom-only aug (p=0.7, range 0.7-1.2) | −0.022 | Redundant with multi-scale at inference. |
| Light aug (combined, p=0.3) | −0.055 | Even mild combined aug hurt more than zoom alone. |
| MixUp (α=0.2) | −0.030 | Softens decision boundary → FPs ~doubled. |
| resnet50 backbone | −0.026 | Better recall but much worse precision (FP 18 → 34). |
| Multi-task aux head (λ=1.0) | −0.118 | Presence loss (~0.5) dwarfed heatmap loss (~0.02); heatmap head starved. |
| Multi-task aux head (λ=0.05, balanced) | −0.005 (noise) | Mechanism works (FP drops on val with joint abstain) but recall trade matches val-optimal threshold; no net F gain. |
| BN train-mode at inference | −0.034 | Per-batch BN stats too noisy at 8 slices/batch. |
| Multi-scale inference (28+32 Å) | −0.001 | Val F unchanged, FPs halved. **Timed out at submission** because the 28 Å scale resamples to 1.49× more voxels. |
| Multi-scale inference (32+36 Å) | −0.030 | 36 Å downsamples too aggressively. |
| Unsupervised threshold (Otsu, GMM on peak distribution) | −0.06 to −0.10 | Peak distributions aren't bimodal enough — TP and FP confidences overlap. |
| Shape-feature calibrator (LR on peak / mean / entropy / etc.) | +0.010 on val | Submitted to private: −0.014. Overfit val/public, didn't transfer to private. |

## Test submissions and what I learned from each

| Submission | Private | Public | Wall time | Lesson |
|---|---|---|---|---|
| Raw exp_10 single-pass (thr=0.675) | 0.5672 | 0.5513 | unknown | Baseline. Val 0.889 → private 0.567. |
| Shape calibrator | 0.5529 | 0.5844 | 8 h | val-fit calibration → overfits public (+0.033), under-performs private (−0.014). |
| Aug-trained model (single-pass, thr=0.725) | 0.4511 | 0.4924 | 7 h | Confirms aug hurts both val and private. Don't aug-retrain. |
| 4-way TTA single-GPU | timed out | — | >12 h | Inference budget is binding. |
| **3-way TTA + 2-GPU threading (winner)** | **0.6069** | **0.5974** | ~8 h | **+0.040 private. Submitted as final.** |
| 3-way TTA + 2-scale (28+32) | timed out | — | >12 h | 28 Å is 1.49× more expensive per pass — pushed past the cap. |
| 3D cascade (2D for confident, 3D for uncertain band) | 0.3850 | 0.3200 | ~9 h | The 3D model's val→private gap is huge; replacing 2D's hardest cases with 3D's *worst* cases backfired. |

## Closing observations

- **Val-to-private gap was ~0.30** for the winning model. It's smaller than my distillation/cascade attempts suggested (the 3D model had a much bigger gap, ~0.46), but it's still big. The top teams reported the same and gave up on local CV beyond F≈0.93, switching to LB tuning. With 5 submissions/day this is hard.
- **Anything tuned on val tends to overfit public more than private.** The shape calibrator showed this most clearly: +0.033 public, −0.014 private. Universal lesson: prefer mechanistically-motivated changes (TTA averaging, multi-GPU parallelism) over val-fit ones (calibration heads, threshold lowering).
- **The biggest mistake** was not adopting Bartley's external dataset. That's ~0.05-0.10 of free F for everyone in the gold zone. I had set a "no extra data" rule for myself for learning purposes — defensible but it left a lot of points on the table.
- **The second-biggest** was sticking with 32 Å resampling. I picked it for disk budget; the top teams resampled at 16 Å. Probably another ~0.05.
- **The third was the val/cv strategy.** A 80/20 stratified split was reasonable but it left a confidence-distribution gap to LB that I never bridged. The top teams used 4- or 5-fold CV and used public LB above val F ≈ 0.93 — that's the pragmatic choice in retrospect.

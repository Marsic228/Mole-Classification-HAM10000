# Fold 5 ROI geometry diagnostics — 2026-11-05

## Scope and evidence

- Source: `roi_geometry_diagnostic_fold5.csv` and its summary JSON, **1,706 validation images**.
- Fixed ConvNeXt classifier, reference-mask versus U-Net-predicted-mask crops; **no retraining**.
- Geometry measured on original 600×450 images. Per-image mask Dice is recomputed at original resolution and need not equal the model's 256×192 validation Dice.
- Reference masks are the project’s reviewed/U-Net-assisted masks, **not independent clinical ground truth**.

## Confirmed results

| Outcome | Images |
|---|---:|
| Regression (reference correct → predicted wrong) | 44 |
| Correction (reference wrong → predicted correct) | 28 |
| Wrong → other wrong | 22 |
| Prediction unchanged | 1,612 |

- **36/44 regressions** happened when both masks produced `roi` (not fallbacks).
- **All 36 of those ROI→ROI regressions retained 100%** of the reference-mask pixels in the predicted crop; **43/44** regressions across all transitions retained 100%.
- Therefore, the main observed regression pattern is **not obvious loss of the annotated lesion**. Scale, position, visible background, and classifier sensitivity remain plausible contributors.
- Among these 36 ROI→ROI regressions, predicted crop area increased by more than 1 percentage point for **16**, decreased by more than 1 pp for **15**, and changed by at most 1 pp for **5**. A universal crop-tightening rule is therefore unsupported.
- **15/36** ROI→ROI regressions still had mask Dice ≥0.95.

## Geometry association — restrict to baseline-correct, ROI→ROI images

This restriction is essential: a regression is only possible when the reference prediction was correct, and comparing all unchanged predictions would mix correct and incorrect baselines. Population: **1,014** images; **36** regressions, **978** unchanged and correct.

| Measure | Regressions (n=36) | No regression (n=978) |
|---|---:|---:|
| Median absolute crop-area difference, image fraction | 0.0884 | 0.0178 |
| Median crop rectangle IoU | 0.8622 | 0.9528 |
| Median mask Dice | 0.9364 | 0.9682 |

- Absolute crop-area difference **≥5 percentage points:** **22/282** regressions (**7.80%**).
- Absolute crop-area difference **<5 pp:** **14/732** regressions (**1.91%**).
- Exploratory unadjusted risk ratio **4.08**. Stratifying by diagnosis class gives Mantel–Haenszel odds ratio **2.77** (approximate 95% CI 1.38–5.56); exploratory p≈0.0015.
- **Do not treat this as causal or a validated clinical prediction threshold**: the 5 pp split was selected post hoc, data include related images within a fold, and class groups can be small.

### Candidate geometry warning observable at inference

A sparse mask occupying only a small fraction of its bounding box can signal distant components or an irregular shape:

`bbox_fill = predicted_foreground_area / predicted_bounding_box_area`.

Among baseline-correct ROI→ROI images, `bbox_fill < 0.40` appears in **67** images with **8** regressions (11.9%), versus **28/947** (3.0%) with `bbox_fill >= 0.40`. This is **exploratory only**: it does **not prove** that disconnected false-positive islands exist, and the 0.40 threshold is not recommended as an inference rule.

## Representative cases to inspect

| ID | Class | Original → predicted class | Reference crop fraction → predicted | Mask Dice |
|---|---|---|---|---:|
| `ISIC_0033820` | mel | mel → bcc | 0.137 → 0.882 | 0.565 |
| `ISIC_0031372` | df | df → mel | 0.137 → 0.662 | 0.690 |
| `ISIC_0026313` | df | df → mel | 0.321 → 0.771 | 0.935 |
| `ISIC_0029343` | mel | mel → nv | 0.127 → 0.280 | 0.871 |
| `ISIC_0030015` | df | df → mel | 0.826 → 0.662 | 0.936 |

These illustrate **both widening and tightening** among regressions. The accompanying ranked CSV contains **all 44** regressions.

## Recommended next controlled test (no retraining first)

**Hypothesis:** small disconnected predicted-mask components enlarge bounding boxes and create crop variations; **largest-connected-component (LCC) postprocessing** might reduce some harmful changes.

1. First run `mask_component_audit.py` to count 8-connected components, quantify how many predicted crops would actually change if only the largest component were retained, and see whether those cases overlap the **44 regressions**. This is an audit only: no new crops and no changes to trained models.
2. **Only if there is evidence of meaningful detached components**, freeze a single LCC-only postprocessing rule and evaluate it on **all five existing held-out folds**, with the same U-Net and ConvNeXt checkpoints, sigmoid threshold 0.5, and ROI margin 0.15. Evaluate fold-paired macro F1, class F1/recall (`df`, `mel`), crop transitions, corrections/regressions, and computational overhead.
3. If few relevant images change under LCC, **do not pursue LCC**. Prioritize the alternative hypothesis: classifier sensitivity to normal reference-vs-predicted crop geometry (e.g., controlled crop-robustness training with leakage-safe segmenter predictions).

**Do not adopt** LCC or an ad-hoc bounding-box-fill threshold based on Fold 5 alone. Improvements in one development fold require controlled five-fold comparison and, ultimately, independent validation.

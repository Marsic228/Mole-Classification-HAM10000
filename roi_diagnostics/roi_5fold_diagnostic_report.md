# HAM10000 — Five-fold automatic ROI diagnostic audit

**Date:** 2026-11-05  
**Input:** Five `classifier_predicted_vs_reference_foldN.csv` files supplied by user, each containing `image_id`, `dx`, `true_label`, `reference_prediction`, `predicted_mask_prediction`.  
**Scope:** Identical fold-specific ConvNeXt checkpoints; reference-mask ROI vs U-Net-predicted-mask ROI. Same 8,530 development images; this is **not independent test-set evaluation**.

## Validation and headline result

- All five CSVs parsed; fold validation sizes **1,707 / 1,706 / 1,706 / 1,705 / 1,706**, total **8,530** distinct image IDs with **no duplicates**.
- All `dx` strings agree with `true_label` indexes under class order `akiec, bcc, bkl, df, mel, nv, vasc`. No missing labels or out-of-range predictions.
- Fold-mean macro F1: reference ROI **0.751194**, predicted ROI **0.742623**; paired difference **−0.008571**, fold-level sample SD of differences **0.015487**. The five folds have two positive and three negative differences, including a **−0.030244** Fold 5 drop.
- Pooled OOF macro F1, computed on concatenated predictions: reference **0.752872** vs predicted **0.743661**. **This is a different aggregation from the average of fold macro F1 scores.**
- Pooled accuracy: **0.854865** vs **0.850879** (net **34** fewer correct predictions).

## Per-fold prediction changes

| Fold | Images | Changed | Corrected errors | New errors | Wrong→wrong | Reference macro F1 | Predicted macro F1 | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1707 | 77 | 34 | 30 | 13 | 0.777217 | 0.788229 | +0.011012 |
| 2 | 1706 | 77 | 27 | 38 | 12 | 0.734088 | 0.724227 | −0.009862 |
| 3 | 1706 | 89 | 33 | 44 | 12 | 0.756871 | 0.742961 | −0.013910 |
| 4 | 1705 | 88 | 37 | 37 | 14 | 0.736785 | 0.736932 | +0.000147 |
| 5 | 1706 | 94 | 28 | 44 | 22 | 0.751009 | 0.720765 | −0.030244 |
| **Total** | **8530** | **425** | **159** | **193** | **73** | — | — | — |

Predictions changed for **425 / 8,530 (4.98%)** images; **8,105** remained unchanged. There is a **net loss of 34 correct predictions** despite the small fraction of changed predictions. A changed prediction is not necessarily a correction or regression: 73 instances moved between two incorrect classes.

## Pooled per-class classification quality

| True class | Support | Reference F1 | Predicted F1 | Δ F1 | Reference recall | Predicted recall | Corrected / newly wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 286 | 0.6811 | 0.6823 | +0.0012 | 0.6608 | 0.6608 | 13 / 13 |
| bcc | 428 | 0.7817 | 0.7865 | +0.0048 | 0.7991 | 0.8178 | 23 / 15 |
| bkl | 929 | 0.7190 | 0.7130 | −0.0060 | 0.6900 | 0.6846 | 36 / 41 |
| df | 98 | 0.6901 | 0.6380 | **−0.0520** | 0.6020 | 0.5306 | 3 / 10 |
| mel | 973 | 0.6149 | 0.6080 | −0.0068 | 0.5612 | 0.5683 | 37 / 30 |
| nv | 5686 | 0.9285 | 0.9262 | −0.0023 | 0.9508 | 0.9442 | 47 / 84 |
| vasc | 130 | 0.8549 | 0.8516 | −0.0033 | 0.8385 | 0.8385 | 0 / 0 |

**Important precision/recall nuance:** For true `mel`, correctly identified images increased from **546 to 553** across five folds, but false `mel` predictions increased from **257 to 293**. Melanoma precision therefore decreased **0.6800 → 0.6537** while recall increased **0.5612 → 0.5683**. Pooled melanoma F1 still decreased slightly.

**Dermatofibroma (`df`):** Across 98 images, true-positive `df` predictions decreased **59 → 52**, resulting in the largest pooled F1 decline. Fold 5 accounts for a major part: among **20 true `df` images**, true positives declined **13 → 10**. There were five regressions and two corrections. The five regressions were three `df→mel`, one `df→bkl`, and one `df→nv`.

## Main label transition patterns

Here, a transition is `reference-mask classifier prediction → predicted-mask classifier prediction`, conditioned on the **actual** class. These are class-prediction changes, not image changes.

- **Actual `nv`**: correctly predicted `nv`→wrong `mel` in **41** cases; wrong `mel`→correct `nv` in **26** cases. Other `nv` regressions include `nv→bkl` (**22**) and `nv→bcc` (**15**).
- **Actual `mel`**: wrong `nv`→correct `mel` in **26** cases; correct `mel`→wrong `nv` in **20** cases. Hence, the effect is mixed, not uniformly negative for melanoma recall.
- **Actual `bkl`**: correct `bkl`→wrong `nv` in **20** cases, wrong `nv`→correct `bkl` in **18** cases.
- **Actual `df`, Fold 5**: five newly incorrect cases; `df` recall declined from 13/20 to 10/20. This class has low support, and a handful of cases meaningfully affects F1.

## Priority visual review: Fold 5

The companion `roi_fold5_priority_cases.csv` contains **25 concrete image IDs**:

- `df`: **5 regressions + 2 corrections** (all such Fold 5 cases).
- `mel`: **10 regressions + 4 corrections** (all such Fold 5 cases).
- `nv`: **4 correctly classified nv → wrong mel** regression examples.

Five `df` regression images:
`ISIC_0031372`, `ISIC_0026313`, `ISIC_0029962`, `ISIC_0030015`, `ISIC_0033810`.

Ten `mel` regression images:
`ISIC_0034048`, `ISIC_0024701`, `ISIC_0033820`, `ISIC_0031957`, `ISIC_0024333`, `ISIC_0029343`, `ISIC_0031915`, `ISIC_0032981`, `ISIC_0028410`, `ISIC_0033302`.

Correction controls:
`df`: `ISIC_0032410`, `ISIC_0028790`; `mel`: `ISIC_0032098`, `ISIC_0031377`, `ISIC_0028065`, `ISIC_0034172`.

### What the five-column CSVs cannot establish

1. There are **no per-image crop statuses** (`roi` vs fallback). Group-level export counts were reported separately, but they cannot be joined to these exact errors.
2. There are **no mask images or crop bounding boxes**, so it is not yet possible to identify segmentation false positives, lost lesion pixels, enlarged crop regions, or shifts in lesion framing in these specific cases.
3. No image-level probabilities, margins, or calibration scores are present.
4. No runtime measurements are present. No clinical safety, statistical-significance, or external generalization claim follows from these development observations.

### Proposed visual comparison

Compare **original input**, **reference-mask ROI**, and **predicted-mask ROI** for the 25 priority images; label the outcome and true class. Specifically record:

- Was each input truly cropped or was it a full-image fallback?
- Did the ROI-to-fallback state change between masks?
- Did the lesion change apparent size, location, or surrounding-skin context?
- Did the predicted crop exclude a visible lesion part or include distracting structures?
- For cases that improved, was the crop closer to what the classifier was trained on?

The companion `prepare_roi_visual_review.py` produces contact sheets and an uploadable ZIP from local images; it **does not** retrain U-Net/ConvNeXt or change archived files.

## Working interpretation and next controlled experiment

**Supported by the comparisons:** Five-fold mean macro F1 falls by **0.00857** when only mask source for validation ROI changes; Fold 5 declines most; `df` performance and melanoma precision deserve attention. Automatic ROI gives usable classification performance but provides no clear improvement over reference-mask ROI under this setup.

**Hypotheses needing image-level evidence:** altered crop size/context, small disconnected mask components expanding boxes, or changed ROI/fallback state. These are untested causes; high Dice alone does not distinguish them.

**Next gate:** inspect the 25 image triplets, then join crop status/geometry to the full 425 changes if available. Only after diagnosis choose **one** controlled improvement (for example removing tiny disconnected components if distant false positives explain a substantial fraction of regressions, or building leakage-safe predicted-ROI classifier training if systematic framing shift dominates). Do not change sigmoid threshold, margin, fallback rules and model simultaneously. Preserve the original outputs as the baseline.

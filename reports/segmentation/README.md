# Segmentation and automatic ROI evidence

The [canonical history, D23–D25](../../docs/research/Diploma_Experiment_History.md#d23-automatic-smallunet-to-roi-to-convnext--local-evidence-verified) records the research decisions. This guide maps the preserved evidence; it is not a second experiment journal.

| File or location (relative to this directory) | Meaning |
|---|---|
| [automatic_roi_5fold_summary.json](automatic_roi_5fold_summary.json) | Recalculated five-fold scores, sample SDs, paired changes, separate pooled OOF, coverage, Fold 5 counts, evidence paths and limitations |
| [phase1_artifact_manifest.json](phase1_artifact_manifest.json) | Pre-edit artifact sizes/SHA-256/availability, original Git status/diff, and preservation-verification result |
| small_unet_foldN_seed123/history.json | Epoch validation summaries; selected checkpoints require Dice improvement >0.0001 |
| small_unet_foldN_seed123/best_validation_per_image.csv | Selected validation Dice/IoU per image, measured at 256×192 |
| small_unet_foldN_seed123/classifier_predicted_vs_reference_foldN.csv | Same fold-specific D22 ConvNeXt, reference versus unfiltered predicted ROI inputs |
| small_unet_foldN_seed123/largest_component_roi_comparison_foldN.csv and corresponding summary JSON | Post-hoc LCC versus original predicted ROI; includes original crop statuses used to verify coverage |
| [largest_component_roi_5fold_summary.json](largest_component_roi_5fold_summary.json) | Original aggregate LCC result; retained unchanged |
| [Fold 5 geometry CSV](small_unet_fold5_seed123/roi_geometry_diagnostic_fold5.csv) and [summary](small_unet_fold5_seed123/roi_geometry_diagnostic_fold5_summary.json) | Geometry and mask overlap at original resolution; not the same Dice aggregation/resolution as training validation |
| [Fold 5 component summary](small_unet_fold5_seed123/mask_component_audit_fold5_summary.json) | Hypothetical crop effects before downstream LCC evaluation; its outcome labels refer to the original input comparison |

N denotes canonical folds 1–5. Fold 1 also retains roi_retention_audit.csv (256×192) and roi_retention_audit_original_resolution.csv; these distinct historical diagnostics must not be merged. Additional saved analysis and visual-review preparation source lives in [roi_diagnostics](../../roi_diagnostics/); the older [geometry narrative](roi_fold5_geometry_analysis_report.md) is preserved, with its pending LCC recommendation superseded by D25. The adjusted odds-ratio/CI/p-value in that older narrative were not independently reproduced here.

## Results and evidence boundaries

Five-fold mean ± sample SD: segmentation Dice **0.913393 ± 0.003422**; reference ROI macro F1 **0.751194 ± 0.017390**; predicted ROI **0.742623 ± 0.027058**; paired predicted-minus-reference **−0.008571 ± 0.015487**. Predicted ROI covers **6,103/8,530 (71.5%)**; the other images use whole-image fallback. Separately, pooled development OOF macro F1 is **0.752872 reference / 0.743661 predicted**. The JSON defines each aggregation explicitly.

LCC gives **+0.000752 ± 0.008047** versus unfiltered predicted ROI, improving 2/5 folds. **Reject as default preprocessing.** Geometry differences are associated with regressions on Fold 5; causation is not established. Foreground retention does not guarantee retention of all useful context. Crop-scale/position variation in classifier training remains an untested hypothesis, not an approved training experiment.

These are selected-checkpoint development results, not independent external or clinical validation. Reference-mask provenance is incomplete; Dice describes agreement with available masks. Geometry diagnostics concentrate on Fold 5; LCC was investigated post hoc. Inference cost and classifier confidence were not measured.

## Preprocessing and reproduction requirements

- Use the frozen [split CSV](../../splits/ham10000_dev_cv_v1.csv) and [manifest](../../splits/ham10000_dev_cv_v1.json), seed 123 and class order akiec, bcc, bkl, df, mel, nv, vasc. Preserve historical model-specific folds for unrelated older experiments.
- Segmentation input is RGB /255, 256×192 (width×height); bilinear image resizing and nearest-neighbor reference-mask resizing. Predict with sigmoid ≥0.5 and upsample binary masks by nearest neighbor to original resolution.
- Use [roi_crop.py](../../preprocessing/roi_crop.py): margin 0.15 per side, minimum dimension fraction 0.35, original aspect ratio, maximum foreground fraction 0.60; full-image fallback for empty/large/edge masks or crops too large to fit. Export 224×224 JPEGs using the original encoding behavior.
- Reference images are under data/processed_roi_margin15_v1/SOURCE_SPLIT/CLASS/IMAGE_224x224.jpg. Predicted crops are under data/processed_roi_predicted_foldN_v1/val/CLASS/IMAGE_224x224.jpg regardless of original source_split. Raw images are under data/images/.
- Reuse the corresponding segmentation and D22 ConvNeXt checkpoints listed in the JSON and manifest. Preserve classifier transforms, AMP and batching. LCC uses 8-connectivity, JPEG-roundtrips altered crops, replays original predictions for changed cases and reuses saved predictions for unchanged crops.
- Recalculate macro F1 with all seven labels, then unweighted fold mean and sample SD (ddof=1); compute paired deltas within folds. Compute pooled OOF separately after checking unique matched image IDs and labels. Geometry comparisons must restrict to reference-correct ROI→ROI images and use absolute crop-area difference divided by original image area.
- Model replay/export scripts write reports or crops; their non-destructive/read-only descriptions do not mean they perform no filesystem writes. Never rerun them over original outputs. Phase 1 only recalculated existing CSV/JSON evidence and hashed bytes; it did not execute model inference or retraining.

Masks.zip, the separate expert-mask archive, raw/processed data, crops, checkpoints, environments, caches and image-bearing/transport ZIPs stay local with separate backups. Source and numerical CSV/JSON/Markdown evidence should remain visible for a later Git inclusion review. [Source provenance](../../docs/research/SOURCE_PROVENANCE.md#automatic-roi-source-and-artifact-chain--2026-11-05) explains remaining attribution and mask limitations. The tracked-only reports index and its generator remain unchanged pending separate approval.

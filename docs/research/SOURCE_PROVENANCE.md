# Research source and reproducibility provenance

**Updated:** 2026-11-05 (automatic ROI evidence supplement; historical provenance retained)
**Repository:** `Marsic228/Mole-Classification-HAM10000`
**Evidence boundary:** checkpoint *inference* is reproduced within stated tolerances; training has **not** been reproduced from scratch. These are development-set evaluations, not clinical or independent external validation.

## Current tracked source provenance

| Run / model | Source | Evidence |
|---|---|---|
| ConvNeXt v17 EMA, warmup | `experiments/convnext/convnext_tiny_cv_train_reference.py` | Tracked training implementation; inference replay 5/5 PASS |
| ConvNeXt natural sampling + Cross-Entropy | `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` | Tracked training implementation; inference replay 5/5 PASS (saved metrics exact) |
| ConvNeXt natural sampling + Balanced Softmax | `experiments/convnext/convnext_tiny_cv_train_natural_balsoftmax.py` | Controlled CE comparison; five best-checkpoint replay checks PASS, but lower macro F1 (0.7295 vs 0.7386); rejected as CE replacement |
| Paired natural CE / Balanced Softmax evaluator | `evaluation/convnext_natural_ce_vs_balsoftmax.py` | Loads 10 fold-specific checkpoints with canonical held-out folds; summary in `reports/convnext_natural_ce_vs_balsoftmax_summary.json` |
| PanDerm Base partial4 EMA | `experiments/panderm/panderm_cv_train.py` | Tracked training implementation; inference replay 5/5 PASS |
| ConvNeXt v13 / v14 SAM | `archive/legacy_experiments/convnext_tiny_cv_train_v13.py` and `...v14_sam.py` | Historical source preserved, not part of the 15 replays |

The **recovered sources** below are historical VS Code editor snapshots, not executable project modules. Their `.py.txt` wrappers preserve the original contents; the hashes and timestamps are recorded in `archive/recovered_sources/provenance.json`.

| Experiment | Selected Fold 1 snapshot | Selected Folds 2–5 snapshot | Historical clues |
|---|---|---|---|
| ConvNeXt v18 LLRD | `convnext_tiny_cv_train__18sU__d61e29bdf7db.py.txt` | `convnext_tiny_cv_train__87P5__d3f3908e0434.py.txt` | LLRD decay 0.8; warmup; EMA; Adam groups; matching output tag |
| PanDerm partial8 EMA | `panderm_cv_train__RqQr__fa1ad5637c49.py.txt` | `panderm_cv_train__uWQC__092e667bce5b.py.txt` | 8 trainable blocks; EMA; Adam; matching tag |
| PanDerm partial4 AdamW | `panderm_cv_train__H6pT__6bd5f3de975d.py.txt` | `panderm_cv_train__ZmWB__29652336298f.py.txt` | 4 trainable blocks; AdamW; matching tag |
| ResNet18 Focal aux02 | `resnet18_focal_cv_train__vVeV__f4be7be19424.py.txt` | `resnet18_focal_cv_train__Q4NT__a77a7b30e004.py.txt` | 7-class head + 3-class auxiliary head; weight 0.2; matching tag |

**Not an exact-original-training guarantee:** Tags, dates and compatible configurations support historical attribution but cannot prove the checkpoint was trained by a specific snapshot. All 22 distinct snapshots parse and their byte hashes match the archive manifest. Some intermediate PanDerm versions have misleading output names, so use the selected pairs above and retain transitional snapshots only as evidence.

## Inference reproducibility evidence

- The original 15/15 priority checkpoint replays **PASS** at macro F1 and accuracy tolerance 0.003, with strict state-dict loading and best-epoch match. All 15 checkpoint SHA-256 hashes are recorded separately.
- On 2026-10-19, an additional paired evaluation of the five natural CE and five Balanced Softmax selected checkpoints passed 10/10 replay checks (macro F1 differences at most 0.000026 against saved best summaries). The new report files do not establish end-to-end retraining reproducibility, independent testing or SHA-256 verification for the additional Balanced Softmax checkpoints.
- ConvNeXt v17 five-fold macro F1 from replay: **0.743064 ± 0.014585** (sample SD); Fold 3 differed from its independent evaluation by one image prediction, though within tolerance.
- PanDerm partial4 five-fold macro F1 from replay: **0.736796 ± 0.023202**.
- ConvNeXt natural CE five-fold macro F1 from replay: **0.738647 ± 0.017843**; matches saved best-summary and epoch accuracy metrics exactly.
- The underlying fold assignments are model-specific. The CV development pool has 8,530 image IDs, and no overlap of lesion IDs between train and validation within each original fold.
- See `Checkpoint_Replay_Verification_2026-10-14.md` and `Checkpoint_Replay_Checkpoint_Manifest_2026-10-14.csv`; corresponding reproducible commands are in the research `README.md` and source script `evaluation/replay_checkpoint_fold.py`.

## Environment and immutable fold evidence

| Evidence | Identifier |
|---|---|
| Canonical new-experiment and saved ConvNeXt folds | `splits/ham10000_dev_cv_v1.csv` and `reports/repro_audit/convnext_folds.csv`, SHA-256 `a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248` |
| Historical PanDerm folds | `reports/repro_audit/panderm_folds.csv`, SHA-256 `08ffb6dbf84565c41f098b843c299cf00e9a0d80f4a8e7651ec568e1501b8be6` |
| ConvNeXt evaluation environment | Python 3.13.7, torch 2.12.0+cu126, torchvision 0.27.0+cu126, sklearn 1.9.0 |
| PanDerm evaluation environment | Python 3.10.11, torch 2.4.1+cu118, torchvision 0.19.1+cu118, sklearn 1.7.2 |
| Upstream PanDerm checkout | `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` |

Full *observed* environment inventories exist in `reports/repro_audit/{convnext,panderm}_environment.json`. They are not confirmed clean-build lockfiles. `data/`, `checkpoints/`, `venv/` and `PanDerm/` are local/ignored; a remote checkout without protected backups cannot rerun these validations.

## Invalid / unresolved historical evidence

- **INVALID:** `reports/convnext_v17_panderm_partial4_ensemble_summary.json` and its fold1 companion have explicit failed reference checks. Do **not** claim macro F1 0.9323 as a valid result. Correct matched OOF *development* ensemble result is 0.762088 in `reports/oof_convnext_panderm_ensemble_results.json`.
- **UNRESOLVED:** independent evaluation versus training best-summary differences for ConvNeXt v11 CutMix folds 1, 2, and 5. Preserve both records until explained.
- **RUN NAME SAFETY:** the former generic ConvNeXt trainer was removed from active experiments as a duplicate of the archived v14 SAM source. Future experiments must use distinct, versioned scripts and unique `RUN_NAME` values; never overwrite the natural-CE or Balanced Softmax outputs.
- **NOT VERIFIED:** deterministic full retraining, exact original checkpoint-source attribution for the recovered historical snapshots, restored backup images/weights, rebuilt locked environments, and independent or clinical generalization.

## Automatic ROI source and artifact chain — 2026-11-05

This supplement records **currently available local evidence**. The segmentation code and reports are uncommitted; current fingerprints do not prove which exact bytes were executed during training. The two pre-existing tracked source modifications remain intact. Full original status and tracked diffs are preserved in the [Phase 1 manifest](../../reports/segmentation/phase1_artifact_manifest.json).

| Stage | Source relationship | Saved artifact relationship |
|---|---|---|
| Frozen development folds | core/canonical_cv.py; splits/ham10000_dev_cv_v1.csv and .json | Same canonical held-out image and lesion memberships for segmentation and ROI classifiers |
| Reference masks and input pairs | core/cross_validation_dataset.py (local segmentation additions); Masks_to_IsicId_mapping.csv; Masks.zip | RGB /255 at 256×192, bilinear image resize; nearest-neighbor mask resize and >127 binarization |
| Segmentation training | experiments/segmentation/train_fold1.py through train_fold5.py; core/segmentation_unet.py; core/segmentation_losses.py | checkpoints/segmentation/small_unet_foldN_seed123_best.pt; reports/segmentation/small_unet_foldN_seed123/history.json and best_validation_per_image.csv |
| Predicted ROI export | evaluation/export_predicted_roi_fold1.py through export_predicted_roi_fold5.py; preprocessing/roi_crop.py | data/processed_roi_predicted_foldN_v1/val/CLASS/IMAGE_224x224.jpg; all held-out images stored under val/ |
| Fixed reference-ROI classifier | experiments/convnext/convnext_tiny_cv_train_roi_margin15.py | checkpoints/convnext_tiny_natural_ce_roi_margin15_seed123_foldN_best.pt and reports/convnext_tiny_natural_ce_roi_margin15_seed123_foldN_evaluation.json |
| Reference/predicted comparison | evaluation/compare_predicted_roi_fold1.py through compare_predicted_roi_fold5.py | classifier_predicted_vs_reference_foldN.csv in each segmentation run directory; reference crops use original source_split folders |
| Five-fold and visual diagnostics | roi_diagnostics/analyze_roi_5fold.py; roi_diagnostics/prepare_roi_visual_review.py; evaluation/preview_segmentation_failures.py | roi_diagnostics CSV/Markdown evidence; image-bearing visual-review ZIP remains local |
| Original-resolution geometry | evaluation/roi_geometry_diagnostics.py | Fold 5 roi_geometry_diagnostic_fold5.csv and summary JSON; its Dice resolution differs from training-validation Dice |
| Components and downstream LCC | evaluation/mask_component_audit.py; evaluation/evaluate_largest_component_roi.py | Fold 5 component audit; five largest_component_roi_comparison_foldN.csv/summary JSON files and largest_component_roi_5fold_summary.json |

Here N means the corresponding canonical fold 1–5. Exact paths, sizes, availability and SHA-256 values are in the manifest, including all ten checkpoints, original project Python files, critical reports, metadata/mapping and both mask archives. The preserved original Python and report collections also have aggregate baseline fingerprints. No large assets were copied into Git.

**Mask-provenance boundary:** the available reference masks are described in project diagnostics as reviewed/U-Net-assisted; their complete upstream generation/training provenance is not established by the local files. Dice **0.913393 ± 0.003422** measures agreement with those available masks. The separate ISIC2018_Task1_Training_GroundTruth.zip archive is present and fingerprinted, but its availability does not establish that this experiment was validated against independent expert ground truth. Independent annotation provenance, overlap and upstream leakage remain questions for stronger claims.

**Evidence and decisions:** [D23–D25](Diploma_Experiment_History.md#d23-automatic-smallunet-to-roi-to-convnext--local-evidence-verified) contain the canonical interpretation; [verified results](../../reports/segmentation/automatic_roi_5fold_summary.json) separate fold means, sample SDs and pooled OOF. Predicted ROI classification is slightly worse than reference ROI; geometry associations are exploratory; post-hoc LCC filtering is rejected as default. Inference cost and classifier confidence were not measured. The older geometry narrative's adjusted odds ratio/CI/p-value were not independently reproduced in Phase 1.

**Reproduction boundaries:** preprocessing resolution, threshold, ROI rules, JPEG encoding, classifier transforms, class order, AMP/batching and saved fold memberships must be preserved. The LCC evaluator reuses saved predictions for unchanged crops and replays original crops for changed cases before evaluating altered crops. No new inference or retraining was performed in Phase 1. The historical environment manifests remain historical observations, not a newly verified session environment or clean-build lockfile.

**Next hypothesis:** controlled crop-scale/position variation in ConvNeXt training might improve predicted-input robustness; untested and not approved as a training experiment. Any future prediction-generation design must prevent held-out-lesion leakage. Historical baselines and model-selection decisions remain unchanged.

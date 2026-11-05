# Phase 2 — Script consolidation

Implementation and verification record, 5 November 2026. This records software consolidation, not a new scientific experiment or an alternative experiment journal.

Status: implementation complete. Final preservation is evaluated with two external relocations explicitly accepted by the user. No original script was retired.

## Mapping and scope

| Preserved originals, N = 1–5 | New module |
|---|---|
| evaluation/compare_predicted_roi_foldN.py | evaluation/compare_predicted_roi.py |
| evaluation/export_predicted_roi_foldN.py | evaluation/export_predicted_roi.py |
| experiments/segmentation/train_foldN.py | experiments/segmentation/train.py |

Exactly five project files were created by this task: these three modules, tests/test_phase2_consolidation.py, and this document. Implementation order was comparison, export, training; each family passed its current checks before the next module was created.

The original 15 runners and shared modules were not edited. No dependency installation, model architecture/loss/preprocessing change, Git mutation, Obsidian access, new scientific experiment or Phase 3 work occurred.

## CLI commands

Use the existing venv. All relative paths resolve against the repository root. The canonical helper's relative reads run inside a temporary repository-root working-directory context, restoring the prior working directory.

```powershell
.\venv\Scripts\python.exe -B -m evaluation.compare_predicted_roi --fold 1 --output-root C:\path\to\fresh-comparison
.\venv\Scripts\python.exe -B -m evaluation.export_predicted_roi --fold 1 --output-root C:\path\to\fresh-crops
.\venv\Scripts\python.exe -B -m experiments.segmentation.train --fold 1 --output-root C:\path\to\fresh-training --dry-run
.\venv\Scripts\python.exe -B -m unittest discover -s tests -p test_phase2_consolidation.py -v
```

The training example is deliberately a dry-run. Real training was not performed. Full retraining was not authorized.

Common arguments: required --fold {1,2,3,4,5}; --device {auto,cpu,cuda}, default auto; --dry-run; --help. There is no --fold all, overwrite/force option, new scientific hyperparameter, resume mode or deterministic mode.

Dry-run validates paths, canonical metadata, relevant checkpoint identity and collisions. It constructs no model, deserializes no checkpoint, performs no inference and creates no output. Importing a new module performs no experiment. Help exits without importing the model stack. Explicit unavailable CUDA errors.

Historical default outputs remain intact and are refused on collision. Verification always supplied fresh scratch paths.

### Classification comparison

Let RUN = convnext_tiny_natural_ce_roi_margin15_seed123.

| Argument | Default |
|---|---|
| --checkpoint | checkpoints/RUN_foldN_best.pt |
| --reference-root | data/processed_roi_margin15_v1 |
| --predicted-root | data/processed_roi_predicted_foldN_v1 |
| --reference-report | reports/RUN_foldN_evaluation.json |
| --output-root | reports/segmentation/small_unet_foldN_seed123 |

Output: OUTPUT_ROOT/classifier_predicted_vs_reference_foldN.csv. Schema remains image_id,dx,true_label,reference_prediction,predicted_mask_prediction.

Reference lookup preserves source_split; a copy of predicted metadata is forced to source_split=val. Both preserve the canonical helper's row order. Class order remains akiec,bcc,bkl,df,mel,nv,vasc.

Model construction remains convnext_tiny(weights=None), classifier[2] replaced with a seven-output Linear layer, strict model_state_dict loading from the corresponding D22 reference-ROI-trained checkpoint. Loading remains weights_only=True and map_location=cpu before moving the model to the device.

ConvNeXt_Tiny_Weights.DEFAULT.transforms() is retained. In the installed version this means resize shortest side 236, bilinear/antialias, center crop 224, float conversion, mean (0.485,0.456,0.406), std (0.229,0.224,0.225). Batch 8, shuffle=False, zero workers, inference_mode and CUDA-only autocast are retained. Both input streams use the same model sequentially.

Seven-label macro F1 retains zero_division=0 and the absolute reference macro-F1 replay gate <=0.003. Accuracy, per-class F1, changes/corrections/regressions remain printed. The CSV remains label-only. Optional probability capture is a Python verification hook; it is not a new CLI output or scientific setting.

### Predicted ROI export

| Argument | Default |
|---|---|
| --checkpoint | checkpoints/segmentation/small_unet_foldN_seed123_best.pt |
| --image-root | data/images |
| --output-root | data/processed_roi_predicted_foldN_v1 |

Output: OUTPUT_ROOT/val/CLASS/IMAGE_ID_224x224.jpg, regardless of original source_split. A pre-existing output root, even empty, is refused.

SmallUNet loading/device behavior is retained. Inference is one image at a time, eval/inference_mode, without AMP. RGB is bilinearly resized to width 256, height 192; float32 CHW /255 conversion and the NumPy copy retain the original sequence. Sigmoid >=0.5 produces a binary mask, nearest-neighbor resized to original dimensions.

The existing crop_lesion_with_margin helper is reused unchanged: margin 0.15, minimum dimension fraction 0.35, maximum foreground fraction 0.60, original aspect ratio, exclusive bounding-box maxima, ceil/round/clamping and fallback precedence empty -> large area -> edge -> large crop. No LCC filtering is added.

The exact final call remains cropped.resize((224,224)).save(destination), retaining implicit Pillow JPEG behavior and the installed RGB bicubic resize default. No explicit encoder option is introduced.

### Segmentation training

| Argument | Default |
|---|---|
| --image-root | data/images |
| --mask-mapping | Masks_to_IsicId_mapping.csv |
| --mask-archive | Masks.zip |
| --output-root | repository root |

OUTPUT_ROOT prefixes checkpoints/segmentation/small_unet_foldN_seed123_best.pt and reports/segmentation/small_unet_foldN_seed123/{history.json,best_validation_per_image.csv}.

The original initialization sequence is retained: dependency imports -> torch.manual_seed(123) -> canonical metadata -> mapping -> train dataset -> validation dataset -> shuffled train loader -> sequential validation loader -> device -> SmallUNet -> Adam. Both loaders retain batch 4, zero workers, default generator/RNG and drop_last=False. Fresh-process checks include the new preflight before initialization.

SegmentationDataset, prepare_segmentation_pair, SmallUNet, segmentation_loss and evaluate_segmentation are reused without changes. RGB uses bilinear 256x192 and /255; masks use nearest resize and >127. Adam LR is 0.001. Loss remains BCE-with-logits plus per-image soft Dice, eps=1e-6. No scheduler, warmup, AMP, augmentation, accumulation, clipping or resume behavior is added.

Epochs remain range(1,13). Full train then full validation; improvement current_dice > best_dice + 1e-4; patience 4; history is saved before the stop check. Best-checkpoint and selected per-image CSV replacement within a new run retain their historical timing.

Checkpoint fields remain model_state_dict, optimizer_state_dict, epoch, fold, seed, mean_dice, config. Config remains input_size [192,256], batch_size 4, learning_rate 0.001, loss 'BCE + soft Dice', threshold 0.5. RNG/resume fields were not added.

## Deliberate safety improvements

- Check collisions before inference/model construction. Comparison uses exclusive CSV creation; export reserves a fresh root; training reserves a fresh report directory and rechecks the checkpoint.
- Use explicit exceptions instead of optimization-removable assertions in the new modules.
- Require checkpoint overrides to match the selected fold's original Phase 1 size/SHA-256. Copied identical bytes are allowed; other models/folds are refused.
- Validate paired IDs, labels, classes, lengths, ordering and image existence before classifier inference.
- Validate the complete export file set: IDs, class paths, counts and duplicates. The historical count expressions (1707,1706,1706,len(val),len(val)) become canonical and exact-file-set checks.
- Training preflight checks mapping columns, duplicate/blank mapping IDs, missing masks/raw images and ambiguous PNG basenames in the ZIP.

No shared model or preprocessing code was refactored. Console grouping changed slightly; numerical behavior and historical CSV/checkpoint schemas remain intact.

## Preservation baseline and provenance

Initial HEAD: fc02311fc1cc4ef85f1435d059bef950c50d674d. Initial status had seven modified tracked files, 74 untracked files and no staged diff. Full initial status/diffs are in preservation_baseline.json.

All 210 Phase 1 size/SHA-256 entries passed before implementation. The separate additional baseline covers 29,626 existing files and 2,900,630,923 bytes: original Python, reports/docs/splits, relevant raw/reference/predicted image trees and mask inputs. Every original untracked file is covered directly or by the Phase 1 entries. This is not an exhaustive environment authentication.

All original source and relevant checkpoint identities are grounded in reports/segmentation/phase1_artifact_manifest.json. Present bytes do not authenticate exact historical training-source attribution. Canonical CSV SHA-256 remains a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248. Frozen membership is selected by ID; returned ordering still comes from the unchanged metadata construction. There are 8,530 development images, 6,349 lesions and zero train/validation lesion overlap.

| Fold | Train | Validation | Selected segmentation epoch | D22 classifier checkpoint epoch |
|---|---:|---:|---:|---:|
| 1 | 6823 | 1707 | 9 | 14 |
| 2 | 6824 | 1706 | 12 | 5 |
| 3 | 6824 | 1706 | 9 | 7 |
| 4 | 6825 | 1705 | 12 | 8 |
| 5 | 6824 | 1706 | 8 | 12 |

### Accepted external relocations

During this task two original root files disappeared and byte-identical copies appeared under reports/segmentation/. This task issued no move/delete operation. Work stopped on discovery. The user explicitly approved continuing: 'Yes—accept the relocations and finish'.

| Phase 1 original path | Accepted current path | Size | SHA-256 |
|---|---|---:|---|
| fold5_regressions_ranked_by_geometry.csv | reports/segmentation/fold5_regressions_ranked_by_geometry.csv | 8179 | 03622d9320da86a6a9cfce970c32a2225c3da0d2bbcbd859a82c5ce3466d82d8 |
| roi_fold5_geometry_analysis_report.md | reports/segmentation/roi_fold5_geometry_analysis_report.md | 5281 | ab1a7edb62d51d9918692ef716f18b56f536e0d5feafbd67d928e92317736f48 |

The Phase 1 manifest remains unchanged and retains the historical paths. Thus final strict original-path verification is 208/210, with the remaining two exact at their approved new paths. The additional baseline has 29,624 same-path matches plus these two approved relocation matches. Do not describe this as 210 unchanged paths. The separate relocation record resolves the old names without editing Phase 1.

## Verification outcomes

| Level | Result | Scope/limit |
|---|---|---|
| Static equivalence — Verified | All five originals per family match after documented substitutions | Fold/path/count/comment/whitespace differences only; training epoch function AST matches after removing lazy imports |
| CLI/unit/smoke — Verified | 20/20 final tests pass; all three real module dry-runs exit 0 | Parsing, exact argument surface, paths, collisions, missing inputs/checkpoints, classes/labels/order, import/help safety, fallbacks and devices |
| Classification replay — Verified | All 8,530 rows and five CSVs byte-identical | Exact IDs, order, labels and both predictions; metrics within 1e-12; confusion matrices exact |
| Probabilities — Verified on samples | 20 fold/device/input combinations x 9 images; maximum absolute error 0 | Both ROI streams, CPU FP32 and CUDA AMP; original function extracted without executing its top-level module |
| ROI replay — Verified | All 8,530 JPEGs byte-identical and all statuses exact | Complete output sets checked; no encoder exception or relaxed tolerance |
| ROI masks/pixels/geometry — Partially verified | 57 real image samples plus fixtures match original expressions exactly | Includes first batch/tail and all observed fallback statuses; full historical masks were never stored |
| Training initialization/order — Verified | All five folds match in separate original/consolidated CPU and CUDA processes | Exact parameter hashes, RNG states, optimizer state, first two training batches and all ordered validation IDs |
| Training steps — Verified on bounded CPU fixtures | Two synthetic optimizer updates per implementation match exactly | Batch 4 plus incomplete batch 1; loss, parameters and Adam state exact |
| Epoch/selection/schema — Verified mechanically | Mocked validation fixtures pass boundaries, reset, patience and 12-epoch cap | Checkpoint payload captured; no real multi-epoch training |
| Historical full retraining — Not verified | Not run or authorized | Checkpoint inference replay is not end-to-end retraining reproducibility |

Probability tolerances were FP32 atol=1e-6, rtol=1e-5; CUDA AMP atol=1e-4, rtol=1e-3. Metrics require absolute error <=1e-12. IDs, labels, predictions, statuses, sampled masks/pixels and historical JPEG bytes require exact equality. No tolerance was loosened. The original reference-score <=0.003 sanity gate also passed, but exact historical CSV identity was the stronger criterion.

Full classification replay used the production run function once per fold with a reused loaded model for both streams; constructor substitution avoided loading the checkpoint twice. Full export used the production loop with an observation wrapper. AST-extracted original functions/expressions were used for direct controlled comparisons. Original top-level inference/training runners were never imported or executed in full.

The final tests include 11-item classifier fixtures, canonical order against saved CSVs, all crop fallbacks and the strict >0.60 boundary, sigmoid >=0.5, dry-runs with model/checkpoint loading patched to fail, synthetic optimizer updates and exact 0.0001/patience boundaries.

Initial harness issues were corrected without original-file or dependency changes: a minimal subprocess environment lacked USERNAME; a scratch script initially lacked PYTHONPATH. Complete-environment reruns passed. The first unit harness used TemporaryDirectory cleanup for its own transient fixtures; it was changed to retained mkdtemp fixtures before subsequent suites. Named logs, baseline files, replay outputs and later fixtures remain retained. No pre-existing file was deleted by this task.

## Runtime and resources

Python 3.13.7; Torch 2.12.0+cu126; torchvision 0.27.0+cu126; NumPy 2.4.6; pandas 3.0.5; Pillow 12.2.0; scikit-learn 1.9.0; NVIDIA GeForce RTX 2060. These match relevant recorded ConvNeXt environment versions, without proving every historical detail or a clean-machine rebuild. PanDerm's distinct environment/folds were not used.

About 208 GB was free before inference. Export used a conservative scratch budget below 1 GB and checked at least 1 GB remained available. Full classification replay took 70.40 seconds; export took 108.47 seconds. No full retraining occurred.

## Retained verification evidence

Phase 3A retention update (5 November 2026): the complete verification bundle is now preserved locally at `data/verification/phase2-2026-11-05/` (repository-relative; ignored by Git). The [compact verification summary](../../reports/segmentation/phase2_verification/README.md) and [SHA-256 retention inventory](../../reports/segmentation/phase2_verification/retention_manifest.json) record the permanent location and checks. All 8,583 copied files (59,359,920 bytes) passed exact relative-file-set, size and SHA-256 verification. Embedded absolute paths and the original Temp evidence remain unchanged. The scratch location below is retained as historical provenance. Phase 3A did not rerun inference, training or the 20 automated tests; full historical retraining reproducibility remains unverified.

Scratch root: <LOCAL_TEMP>/ham10000-phase2-bab3dcea-483f-4382-b247-d07b0ac76b60

| Artifact | Meaning |
|---|---|
| preservation_baseline.json | Start HEAD/status/diffs, additional fingerprints and Phase 1 manifest hash |
| original_family_static_comparison.json; environment.json | Source-family comparison and runtime |
| final_tests.txt; cli_dry_runs.json | Final 20-test suite and actual module dry-runs |
| comparison_foldN/ | Five byte-identical comparison CSVs |
| comparison_foldN_verification.json; comparison_replay_summary.json | Per-fold metrics, matrices and acceptance |
| comparison_csv_byte_verification.json; probability_verification.json | CSV-byte checks and sampled probabilities |
| export_foldN/ | Fresh JPEG trees |
| export_foldN_per_image.json; export_foldN_verification.json; export_replay_summary.json | Statuses, dimensions, mask/pixel hashes and acceptance |
| initialization_original_DEVICE.json; initialization_consolidated_DEVICE.json; initialization_verification.json | Fresh-process initialization and batch-order checks |
| verify_comparison.py; verify_export.py; verify_initialization.py; verify_probabilities.py | Retained scratch verification harnesses |
| *_tests*.txt; *_replay.log; phase2-fixture-*/ | Test/replay logs and retained later fixtures |
| preservation_discrepancy.json; relocated_evidence_check.json; accepted_external_relocations.json | Stop evidence, relocation fingerprints and approval |
| final_preservation_verification.json; final_new_file_fingerprints.json | Final audit and hashes of the five new files |
| final_git_status.txt; final_git_diff.txt; final_git_diff_stat.txt | Exact final Git evidence |

No verification artifacts will be automatically removed. Cleanup requires separate approval.

## Final Git state and future limits

HEAD remains fc02311fc1cc4ef85f1435d059bef950c50d674d. Nothing was staged, committed or pushed. All seven tracked user modifications match this task's start bytes and exact diff. The existing tracked diff remains 383 insertions and 11 deletions across seven files. Ordinary git diff --stat excludes the five new untracked files.

Final status: seven modified tracked files, 79 untracked files. Relative to the baseline this reflects the five task-created files plus two accepted external relocations. Exact full status follows and is also retained separately.

```text
 M .gitignore
 M README.md
 M core/cross_validation_dataset.py
 M docs/research/Diploma_Experiment_History.md
 M docs/research/README.md
 M docs/research/SOURCE_PROVENANCE.md
 M preprocessing/preview_roi.py
?? core/segmentation_losses.py
?? core/segmentation_unet.py
?? docs/research/Phase2_Script_Consolidation.md
?? evaluation/analyze_predicted_roi_classification_fold1.py
?? evaluation/analyze_segmentation_fold1.py
?? evaluation/audit_segmentation_roi_fold1.py.py
?? evaluation/audit_segmentation_roi_fold1_original.py
?? evaluation/compare_predicted_roi.py
?? evaluation/compare_predicted_roi_fold1.py
?? evaluation/compare_predicted_roi_fold2.py
?? evaluation/compare_predicted_roi_fold3.py
?? evaluation/compare_predicted_roi_fold4.py
?? evaluation/compare_predicted_roi_fold5.py
?? evaluation/evaluate_largest_component_roi.py
?? evaluation/export_predicted_roi.py
?? evaluation/export_predicted_roi_fold1.py
?? evaluation/export_predicted_roi_fold2.py
?? evaluation/export_predicted_roi_fold3.py
?? evaluation/export_predicted_roi_fold4.py
?? evaluation/export_predicted_roi_fold5.py
?? evaluation/mask_component_audit.py
?? evaluation/preview_segmentation_failures.py
?? evaluation/roi_geometry_diagnostics.py
?? evaluation/segmentation_metrics.py
?? evaluation/segmentation_validation.py
?? experiments/segmentation/sanity_overfit.py
?? experiments/segmentation/train.py
?? experiments/segmentation/train_fold1.py
?? experiments/segmentation/train_fold2.py
?? experiments/segmentation/train_fold3.py
?? experiments/segmentation/train_fold4.py
?? experiments/segmentation/train_fold5.py
?? preprocessing/check_segmentation_data.py
?? reports/segmentation/README.md
?? reports/segmentation/automatic_roi_5fold_summary.json
?? reports/segmentation/fold5_regressions_ranked_by_geometry.csv
?? reports/segmentation/largest_component_roi_5fold_summary.json
?? reports/segmentation/phase1_artifact_manifest.json
?? reports/segmentation/roi_fold5_geometry_analysis_report.md
?? reports/segmentation/small_unet_fold1_seed123/best_validation_per_image.csv
?? reports/segmentation/small_unet_fold1_seed123/classifier_predicted_vs_reference_fold1.csv
?? reports/segmentation/small_unet_fold1_seed123/history.json
?? reports/segmentation/small_unet_fold1_seed123/largest_component_roi_comparison_fold1.csv
?? reports/segmentation/small_unet_fold1_seed123/largest_component_roi_comparison_fold1_summary.json
?? reports/segmentation/small_unet_fold1_seed123/roi_retention_audit.csv
?? reports/segmentation/small_unet_fold1_seed123/roi_retention_audit_original_resolution.csv
?? reports/segmentation/small_unet_fold2_seed123/best_validation_per_image.csv
?? reports/segmentation/small_unet_fold2_seed123/classifier_predicted_vs_reference_fold2.csv
?? reports/segmentation/small_unet_fold2_seed123/history.json
?? reports/segmentation/small_unet_fold2_seed123/largest_component_roi_comparison_fold2.csv
?? reports/segmentation/small_unet_fold2_seed123/largest_component_roi_comparison_fold2_summary.json
?? reports/segmentation/small_unet_fold3_seed123/best_validation_per_image.csv
?? reports/segmentation/small_unet_fold3_seed123/classifier_predicted_vs_reference_fold3.csv
?? reports/segmentation/small_unet_fold3_seed123/history.json
?? reports/segmentation/small_unet_fold3_seed123/largest_component_roi_comparison_fold3.csv
?? reports/segmentation/small_unet_fold3_seed123/largest_component_roi_comparison_fold3_summary.json
?? reports/segmentation/small_unet_fold4_seed123/best_validation_per_image.csv
?? reports/segmentation/small_unet_fold4_seed123/classifier_predicted_vs_reference_fold4.csv
?? reports/segmentation/small_unet_fold4_seed123/history.json
?? reports/segmentation/small_unet_fold4_seed123/largest_component_roi_comparison_fold4.csv
?? reports/segmentation/small_unet_fold4_seed123/largest_component_roi_comparison_fold4_summary.json
?? reports/segmentation/small_unet_fold5_seed123/best_validation_per_image.csv
?? reports/segmentation/small_unet_fold5_seed123/classifier_predicted_vs_reference_fold5.csv
?? reports/segmentation/small_unet_fold5_seed123/history.json
?? reports/segmentation/small_unet_fold5_seed123/largest_component_roi_comparison_fold5.csv
?? reports/segmentation/small_unet_fold5_seed123/largest_component_roi_comparison_fold5_summary.json
?? reports/segmentation/small_unet_fold5_seed123/mask_component_audit_fold5.csv
?? reports/segmentation/small_unet_fold5_seed123/mask_component_audit_fold5_summary.json
?? reports/segmentation/small_unet_fold5_seed123/roi_geometry_diagnostic_fold5.csv
?? reports/segmentation/small_unet_fold5_seed123/roi_geometry_diagnostic_fold5_summary.json
?? roi_diagnostics/analyze_roi_5fold.py
?? roi_diagnostics/prepare_roi_visual_review.py
?? roi_diagnostics/roi_5fold_changed_predictions.csv
?? roi_diagnostics/roi_5fold_diagnostic_report.md
?? roi_diagnostics/roi_changed_transitions.csv
?? roi_diagnostics/roi_fold5_priority_cases.csv
?? roi_diagnostics/roi_fold_summary.csv
?? roi_diagnostics/roi_per_class_metrics.csv
?? tests/test_phase2_consolidation.py
```

Keep all 15 original scripts at their current locations. Any later retirement, archival, dependency/scientific change, Git operation or Obsidian edit requires separate review. Phase 3 was not started.

Not established: full historical retraining reproducibility, clean-machine rebuild, cross-hardware numerical equivalence, historical source authentication, full historical probabilities/masks, or new scientific conclusions.

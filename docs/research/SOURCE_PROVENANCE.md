# Research source and reproducibility provenance

**Updated:** 2026-10-15  
**Repository:** `Marsic228/Mole-Classification`  
**Evidence boundary:** checkpoint *inference* is reproduced within stated tolerances; training has **not** been reproduced from scratch. These are development-set evaluations, not clinical or independent external validation.

## Current tracked source provenance

| Run / model | Source | Evidence |
|---|---|---|
| ConvNeXt v17 EMA, warmup | `experiments/convnext/convnext_tiny_cv_train_reference.py` | Tracked training implementation; inference replay 5/5 PASS |
| ConvNeXt natural sampling + Cross-Entropy | `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` | Tracked training implementation; inference replay 5/5 PASS (saved metrics exact) |
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

- 15/15 checkpoint replays **PASS** at macro F1 and accuracy tolerance 0.003, with strict state-dict loading and best-epoch match. All 15 checkpoint SHA-256 hashes are recorded separately.
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
- **RUN NAME SAFETY:** verify the editable `experiments/convnext/convnext_tiny_cv_train.py` has a unique nonempty `RUN_NAME` before executing it. Never overwrite natural-CE outputs.
- **NOT VERIFIED:** deterministic full retraining, exact original checkpoint-source attribution for the recovered historical snapshots, restored backup images/weights, rebuilt locked environments, and independent or clinical generalization.

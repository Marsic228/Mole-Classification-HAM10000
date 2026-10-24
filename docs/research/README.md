# Research documentation

This folder keeps the diploma experiment history together with its primary audit evidence. **This is not a clinical validation package.** Most reported metrics were calculated on the HAM10000 8,530-image development pool using lesion-safe cross-validation and selected checkpoints.

| Document | What it contains |
|---|---|
| [Diploma Experiment History](Diploma_Experiment_History.md) | Chronological experiment decisions, results, methodology, and unresolved limitations; **D22 documents the 24 October 2026 mask-assisted ROI experiment** |
| [Source Provenance](SOURCE_PROVENANCE.md) | Authentic tracked sources versus candidate editor-history snapshots, fold IDs, environments, and evidence status |
| [Checkpoint Replay Verification](Checkpoint_Replay_Verification_2026-10-14.md) | Fold-by-fold 15/15 historical inference replay results, tolerance and known differences |
| [Checkpoint SHA-256 Manifest](Checkpoint_Replay_Checkpoint_Manifest_2026-10-14.csv) | Fingerprints of the 15 historical reloaded checkpoints |
| [Historical Source Recovery Audit](Historical_Source_Recovery_Audit_2026-10-15.md) | Detailed recovered-snapshot attribution and misleading transitional PanDerm versions |
| [Research Evidence Audit — Second Pass](Research_Audit_Second_Pass_2026-10-14.md) | Basic report/source integrity and outstanding v11 CutMix disagreements |

**Stage 3 result (2026-10-24):** The mask-assisted ROI + natural-sampling CE experiment obtained **0.7512 ± 0.0174** five-fold development macro F1, versus **0.7386 ± 0.0178** for the paired full-image CE control; paired mean gain **+0.01255 ± 0.00888**. All **5/5** ROI selected checkpoints passed independent inference replay. Full results: [ROI-versus-CE summary](../../reports/convnext_tiny_natural_ce_roi_margin15_seed123_vs_natural_ce_5fold_summary.json) and [experiment D22](Diploma_Experiment_History.md). Precomputed lesion masks are required and a deployable automatic segmentation system has **not** yet been evaluated.

**Other source locations:** [raw report index](../../reports/README.md), [historical source snapshots](../../archive/recovered_sources/README.md), [per-fold CSV exports](../../reports/repro_audit/), [canonical folds](../../splits/ham10000_dev_cv_v1.csv).

## Check a representative checkpoint (repository root)

These commands perform *inference only*. They require the ignored local image data, checkpoint weights, and model-specific Python environment. The script lives at `evaluation/replay_checkpoint_fold.py`; commit it separately if not tracked yet.

```powershell
.\venv\Scripts\python.exe -m evaluation.replay_checkpoint_fold --model convnext_v17 --fold 1 --hash-checkpoint
.\venv\Scripts\python.exe -m evaluation.replay_checkpoint_fold --model convnext_natural_ce --fold 1 --hash-checkpoint
.\PanDerm\.venv-panderm\Scripts\python.exe -m evaluation.replay_checkpoint_fold --model panderm_partial4 --fold 1 --batch-size 4 --hash-checkpoint
```

For the five canonical ROI folds, use `python -m evaluation.roi_fold_check <fold>` where `<fold>` is 1–5, after supplying the local ROI images and the saved ROI checkpoints. That evaluation updates per-fold reports and the paired five-fold summary, without retraining.

Outputs of the historical replay commands go to `local_audit/replays/`. Do **not** commit the entire `local_audit/` folder; it contains generated local files. Preserve only reviewed summary documents and manifests.

## Safe interpretation

- All 15 historically audited priority checkpoint replays **passed within tolerance 0.003**; this does not mean training was reproduced from scratch. Separately, all **5/5** 2026-10-24 ROI selected checkpoints exactly matched the training summary F1 on replay.
- Fold identities from ConvNeXt and PanDerm historical environments **differ**. Do not cross-apply numeric fold IDs across the two exports.
- Raw source snapshots under `archive/recovered_sources/` are **historical candidate originals**. They retain old imports and should not be executed directly.
- The historical ~0.9323 ensemble report explicitly failed its reference checks. Use the corrected 0.762088 pooled **development OOF** score when discussing the ensemble.
- The v11 CutMix disagreements remain open; documentation is not a substitute for rerunning the original evaluation.
- Mask-assisted ROI development results are not external or clinical validation and do not establish inference costs or reliability with automatically predicted masks.

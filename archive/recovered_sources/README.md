# Recovered historical training sources (VS Code Local History)

**2026-10-15 recovery checkpoint.** These are immutable byte-for-byte copies of
VS Code Local History snapshots found by the user's local source recovery scan.
Files deliberately end in `.py.txt` so they **cannot accidentally be imported or
executed as Python modules**. The `.txt` suffix does not alter the original bytes.

**Evidence status:** strong historical candidate source, **not authenticated as
exact checkpoint-generating code**. Matching output tags, settings, timestamps,
and report/checkpoint names raises confidence; it does not prove which exact
editor revision was executed during model training. No retraining was performed.

## Selected candidates (as-found; do not execute)

| Experiment | Fold 1 training snapshot | Folds 2–5 training snapshot | Evaluation snapshot | Observed configuration |
|---|---|---|---|---|
| ConvNeXt-Tiny v18 LLRD | [`18sU.py`](snapshots/convnext_tiny_cv_train__18sU__d61e29bdf7db.py.txt) | [`87P5.py`](snapshots/convnext_tiny_cv_train__87P5__d3f3908e0434.py.txt) | [`3tVZ.py`](snapshots/convnext_tiny_cv_evaluation__3tVZ__1554fca06fec.py.txt) | LR_DECAY=0.8; Adam parameter groups; one-epoch warmup; EMA; 15 epochs maximum. Raw snapshot imports use old root-level module names. |
| PanDerm partial8 EMA | [`RqQr.py`](snapshots/panderm_cv_train__RqQr__fa1ad5637c49.py.txt) | [`uWQC.py`](snapshots/panderm_cv_train__uWQC__092e667bce5b.py.txt) | not recovered | TRAINABLE_BLOCKS=8; Adam; EMA; WRS+focal; 15 epochs maximum. Intermediate DZZV.py has TRAINABLE_BLOCKS=4 with partial8 filenames: do not use it as partial8 source. |
| PanDerm partial4 AdamW | [`H6pT.py`](snapshots/panderm_cv_train__H6pT__6bd5f3de975d.py.txt) | [`ZmWB.py`](snapshots/panderm_cv_train__ZmWB__29652336298f.py.txt) | [`r2i1.py`](snapshots/panderm_cv_evaluation__r2i1__ef8d9ac831ed.py.txt) | TRAINABLE_BLOCKS=4; AdamW(lr=1e-4,weight_decay=1e-2); EMA; WRS+focal; 15 epochs maximum. 8DSI.py uses AdamW with stale partial8 filenames and is not the selected source. |
| ResNet18 focal auxiliary-loss 0.2 | [`vVeV.py`](snapshots/resnet18_focal_cv_train__vVeV__f4be7be19424.py.txt) | [`Q4NT.py`](snapshots/resnet18_focal_cv_train__Q4NT__a77a7b30e004.py.txt) | [`QA8Q.py`](snapshots/resnet18_focal_cv_evaluation__QA8Q__a55fedd0adbb.py.txt) | 7-class main head + 3-class auxiliary head for {bkl,mel,nv}; auxiliary loss weight=0.2; Adam; WRS+focal; 3 epochs maximum. Historical root imports are not updated. |

## Important interpretation and hazards

- The **fold 1 vs. folds 2–5** pairs differ primarily in the `__main__` fold loop;
  they are preserved as separate real snapshots. Compare timestamps and full
  code before assigning an exact run provenance.
- **Misleading transitional PanDerm snapshots:** `DZZV.py` says partial8 in its
  output paths but `TRAINABLE_BLOCKS=4`. `8DSI.py` retains partial8 paths but
  switches to AdamW and four blocks. `ofRo.py` switches to eight blocks before
  correcting its partial4 output tag. All are preserved as forensic evidence,
  **not recommended historical representatives**.
- Raw scripts still reference original flat root imports (`from train import`,
  `from cross_validation_split import`, etc.) and original filesystem layout.
  **Do not run or rename them to active `.py` files**. Create a separately
  documented compatibility copy if training replay is ever needed.
- `convnext_tiny_cv_train.py` in `experiments/convnext/` is an editable workbench;
  do not replace it with any recovered snapshot. Ensure its future experiment
  run name is unique and does not overwrite existing checkpoints/reports.
- The already completed **15/15 checkpoint inference replays** verify ConvNeXt
  v17, PanDerm partial4, and natural CE checkpoints within stated tolerance.
  They do **not** authenticate these four additional training sources or
  establish deterministic training reproducibility.
- Historical ConvNeXt and PanDerm **numeric CV folds differ by environment**:
  always use the original exported image-ID assignments when comparing checkpoints.
  Git-tracked reports marked `reference_check_passed: false` are invalid metrics.
- All **22** distinct candidate snapshots and their original SHA-256 digests are recorded
  in `provenance.json`, including rejected transitional versions; nothing was
  silently discarded or "fixed". The exact original bytes are preserved.

## Citable evidence

- Run results: `reports/README.md` and individual fold summaries in `reports/`.
- Original folds / package provenance: `reports/repro_audit/`.
- Checkpoint replay reports / hashes: the separately prepared
  `Checkpoint_Replay_Verification_2026-10-14.md` and
  `Checkpoint_Replay_Checkpoint_Manifest_2026-10-14.csv` (not committed by this package).
- Source recovery method: local `evaluation/locate_historical_sources.py`
  when retained by the user, and the original private recovery ZIP.

# Recovery audit: historical training sources — 2026-10-15

## Finding

VS Code Local History retained **22 distinct source snapshots**, all with matching
SHA-256 hashes to the source ZIP manifest and valid Python syntax.
Four previously missing experiment variants now have strong, period-authentic
**candidate** source pairs for Fold 1 and Folds 2–5. This does **not** prove which
revision was actually used to train any saved checkpoint.

| Experiment | Fold 1 snapshot | Folds 2–5 snapshot | Evidence |
|---|---|---|---|
| ConvNeXt-Tiny v18 LLRD | `18sU.py` | `87P5.py` | Matching exact report/checkpoint tag, relevant hyperparameters, historical timestamp and fold loop |
| PanDerm partial8 EMA | `RqQr.py` | `uWQC.py` | Matching exact report/checkpoint tag, relevant hyperparameters, historical timestamp and fold loop |
| PanDerm partial4 AdamW | `H6pT.py` | `ZmWB.py` | Matching exact report/checkpoint tag, relevant hyperparameters, historical timestamp and fold loop |
| ResNet18 focal auxiliary-loss 0.2 | `vVeV.py` | `Q4NT.py` | Matching exact report/checkpoint tag, relevant hyperparameters, historical timestamp and fold loop |

## Verified configuration from source content

- **ConvNeXt v18 LLRD:** source defines `LR_DECAY=0.8`, five late trainable
  feature modules, per-module Adam parameter groups with `target_lr`, warmup,
  CosineAnnealingLR, EMA and focal loss. The Fold 1 and Folds 2–5 selected
  files differ only in the executed fold range. Model results must be compared
  against the original ConvNeXt fold CSV, not PanDerm's numerically different folds.
- **PanDerm partial8 EMA:** eight trainable blocks, Adam optimizer, focal loss,
  WeightedRandomSampler, EMA, cosine scheduler. Selected files differ in fold loop;
  other minor debug output changes exist among snapshots.
- **PanDerm partial4 AdamW:** four trainable blocks, AdamW (`lr=1e-4`,
  `weight_decay=1e-2`), WeightedRandomSampler, focal loss, EMA, cosine scheduler.
  Selected fold loop is Fold 1 vs. Folds 2–5.
- **ResNet18 aux02:** 7-class main head plus three-way {bkl,mel,nv} auxiliary head,
  combined loss using auxiliary coefficient 0.2, Adam, WeightedRandomSampler,
  three epochs. Fold 1 and Folds 2–5 selected files differ by the executed fold
  range. Its training procedure/model differs from the archived ResNet hard-sampling
  script and must never be silently substituted with it.

## Misleading intermediate revisions

- `ofRo.py`: eight trainable PanDerm blocks but still names partial4 report files.
- `DZZV.py`: only **four** PanDerm trainable blocks but retains partial8 report files.
- `8DSI.py`: switches optimizer to AdamW and four blocks but still uses partial8
  names; do not use as a genuine partial8 training source.
- `3VYo.py`: PanDerm AdamW evaluation tag ends in an extra underscore;
  `r2i1.py` corrects the tag. The docstring is still stale about partial8.

## Remaining evidence boundaries

- These are *unmodified original code bytes* in `.py.txt` wrappers. Root-level
  imports and original `__file__` assumptions still require adaptation for the
  reorganized repository. Do **not** run these files directly: historical
  training might overwrite existing checkpoint or report artifacts.
- Candidate matching demonstrates historical provenance, **not exact training
  replay**. Save final checkpoint/data hashes and verify package-environment
  locks/backup locations separately. Selected earlier checkpoint inference
  replays (15/15) do not prove retraining reproducibility of these four variants.
- Old v11 CutMix metric discrepancies and invalid reference-check ensemble files
  remain unresolved. Keep them documented; do not use failed-reference metrics.

## Deliverable

`Recovered_Historical_Experiment_Sources_2026-10-15.zip`: relative tree rooted
at `archive/recovered_sources/` with 22 byte-identical source snapshots,
`README.md` and `provenance.json`. Archive has **24 entries**. Each source has
its raw-content SHA256 recorded and verified on output.

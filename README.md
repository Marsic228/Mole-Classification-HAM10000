# Mole Classification — HAM10000

Diploma research project for seven-class classification of dermatoscopic skin-lesion images. The work covers data preparation, class imbalance, transfer learning, group-safe cross-validation and complementary-model ensembles.

> **Research only:** Models are not clinically validated, cannot diagnose skin cancer, and must not replace a dermatologist. Smartphone photos or unrelated inputs may be outside the training distribution.

## Dataset and evaluation

- **HAM10000:** 10,015 records; classes `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- **Development pool:** 8,530 images covering 6,349 distinct `lesion_id` values (original train + validation partitions).
- **Validation:** five stratified, lesion-group-safe folds; zero overlapping lesions between training and validation for every fold.
- **Primary metric:** macro F1. Also report accuracy, per-class precision/recall/F1, confusion matrices and fold variability.
- **Test policy:** the old exposed test split is historical evidence only; do not use it to select models or claim independent final validation.

### Development results (as of 24 October 2026)

| System | Metric type | Macro F1 | Accuracy |
|---|---|---:|---:|
| ConvNeXt-Tiny v17, warmup + CutMix + EMA | Mean of five folds ± sample SD | **0.7430 ± 0.0144** | 0.8450 ± 0.0091 |
| PanDerm Base partial4 | Mean of five folds ± sample SD | **0.7368 ± 0.0232** | 0.8350 ± 0.0087 |
| Natural-sampling ordinary CE (loss-ablation control) | Selected best checkpoints: five-fold mean ± sample SD | **0.7386 ± 0.0178** | 0.8478 (mean) |
| Mask-assisted ROI (15% margin) + natural-sampling CE | Same canonical folds; selected best checkpoints: five-fold mean ± sample SD | **0.7512 ± 0.0174** | 0.8548 ± 0.0057 |
| Natural-sampling Balanced Softmax (rejected vs CE) | Same canonical folds; best-checkpoint mean ± sample SD | **0.7295 ± 0.0175** | 0.8408 (mean) |
| Natural-sampling LDAM (rejected vs CE) | Same canonical folds; best-checkpoint mean ± sample SD | **0.7308 ± 0.0202** | 0.8423 (mean) |
| ConvNeXt v17 | Pooled OOF, 8,530 images | **0.7444** | 0.8451 |
| PanDerm partial4 | Pooled OOF, 8,530 images | **0.7379** | 0.8351 |
| ConvNeXt v17 + PanDerm partial4, 50/50 probability averaging | Pooled matched OOF, 8,530 images | **0.7621** | 0.8570 |

**Mean fold F1 and pooled out-of-fold (OOF) F1 are different measures.** The ensemble is a promising development result, **not external or clinical validation**. The natural-sampling loss comparisons are complete: ordinary CE achieved mean macro F1 **0.7386 ± 0.0178**, Balanced Softmax **0.7295 ± 0.0175**, and LDAM **0.7308 ± 0.0202**. Both class-aware losses were rejected as replacements for CE under their tested configurations. The local paired CE/LDAM evaluation passed **10/10 selected-checkpoint inference replay checks**; this does not establish retraining reproducibility or external validity.

## Approved seven-stage research roadmap (15 October 2026)

The following is the agreed **execution order**, not a claim that later methods have been tested. Collect comparable evidence throughout; perform the systematic synthesis at Stage 5.

| Stage | Scope | Status and evidence boundary |
|---|---|---|
| **1. Reproducibility audit** | Freeze folds, environments, model/preprocessing details and evaluation | **Checkpoint-replay milestone reached:** 15/15 saved fold checkpoints reproduced metrics within the declared 0.003 tolerance. End-to-end retraining, a clean-machine rebuild and external validation remain unverified. |
| **2. Long-tail methods** | Natural CE, Balanced Softmax, LDAM | **Initial comparisons complete.** CE remains the control. Balanced Softmax and LDAM were rejected as replacements under their tested configurations. See D19–D21 in the experiment history. |
| **3. ROI / segmentation** | Test lesion-centered preprocessing | **Initial controlled ROI experiment complete (D22):** mask-assisted cropping improved canonical-fold mean macro F1 from 0.7386 to 0.7512. Automatic segmentation, mask reliability, deployment cost and external validation remain untested. |
| **4. Supervised Contrastive Learning** | Test improvements in learned representations | **Planned.** Account for positive pairs and RTX 2060 6-GB / batch-8 limits. |
| **5. Systematic experiment analysis** | Compare comparable experiments, errors and complementarity | **Planned synthesis.** Gather per-fold, per-class and matched-ID OOF evidence throughout earlier stages; do the comprehensive analysis before final selection. Do not mix pooled OOF macro F1 with five-fold means. |
| **6. Final ensemble selection** | Choose a defensible standalone or ensemble model | **Planned.** ConvNeXt v17 + PanDerm partial4, 50/50, has development-only pooled matched OOF macro F1 **0.7621**; it is a candidate, not an independently validated final model. Consider error diversity, minority classes and inference cost. |
| **7. Independent evaluation and application** | Assess generalization and build a research prototype | **Planned.** Seek non-overlapping external data, study domain shift, calibration and unsupported/OOD inputs. No clinical diagnosis claim. |

**Next executable step:** Continue Stage 3 — assess mask quality and failure modes, then study automatic segmentation, Dice/IoU, confidence-aware fallback, and additional inference cost. Preserve the completed mask-assisted ROI experiment as a development-stage candidate and retain the natural-CE control. See [D22 in the experiment history](docs/research/Diploma_Experiment_History.md).

## Stage 3 — Mask-assisted ROI feasibility (24 October 2026)

**Hypothesis:** lesion-centered cropping may improve macro F1 by increasing the lesion's effective resolution and reducing irrelevant background. Cropping could also remove useful context. **This is a controlled input-preprocessing experiment, not automatic segmentation.**

- **Masks:** external ISIC lesion masks with a `Masks_to_IsicId_mapping.csv` lookup; all 8,530 development images matched to existing mask files. The masks are supplied to preprocessing, including for validation images. Their real-world availability and mask-generation provenance are separate deployment/validity questions.
- **Crop:** mask bounding box, 15% margin per side; each crop dimension at least 35% of the original; preserve the original image aspect ratio; resize result to 224×224.
- **Fallback:** keep the whole image for empty masks, mask area >60%, masks touching an image edge, or crops too large to fit. Of 8,530 images, **6,232 (73.1%)** were ROI-cropped; **2,298 (26.9%)** used fallback (1,749 large crop; 359 large area; 190 edge; zero empty masks).
- **Controlled model comparison:** ConvNeXt-Tiny + natural-sampling CE, CutMix, EMA, first-epoch warmup, cosine scheduler, seed 123, and identical frozen `lesion_id`-safe canonical folds (`splits/ham10000_dev_cv_v1.csv`). Only the input images were changed.
- **Five-fold results:** ordinary CE **0.7386 ± 0.0178** macro F1; ROI + CE **0.7512 ± 0.0174**; paired mean difference **+0.01255 ± 0.00888** (sample SD). Mean ROI accuracy **0.8548 ± 0.0057**. Four of five folds improved; the remaining fold was essentially unchanged.
- **Per-class:** mean F1 improved in six of seven classes; `vasc` declined (0.8676 → 0.8539). Melanoma F1 improved from **0.5962 → 0.6145** and recall from **0.5427 → 0.5612**. All five ROI selected checkpoints passed independent inference replay against saved best-epoch scores.

**Code:** `preprocessing/roi_crop.py`, `preprocessing/preview_roi.py`, `experiments/convnext/convnext_tiny_cv_train_roi_margin15.py`, `evaluation/roi_fold_check.py`.

**Data paths:** raw JPGs in `data/images/`; local mask archive `Masks.zip`; tracked mapping `Masks_to_IsicId_mapping.csv`; generated 224×224 crops in `data/processed_roi_margin15_v1/` (ignored by Git). The local mask archive and generated crops are not committed. The results are in [`reports/convnext_tiny_natural_ce_roi_margin15_seed123_vs_natural_ce_5fold_summary.json`](reports/convnext_tiny_natural_ce_roi_margin15_seed123_vs_natural_ce_5fold_summary.json), with five per-fold evaluation JSON files and saved training histories.

**Evidence boundary:** scores are **selected-best-checkpoint development CV fold means**, not pooled OOF scores or independent external validation. Improved classification with provided masks does **not** prove an end-to-end automatically segmented workflow improves performance; predicted masks, segmentation errors and inference cost are untested. Historical v17 folds are not a directly paired comparison.

## Project structure

Python source code is organized into packages. Run module commands from the repository root.

```text
Mole-Classification/
├── core/                  # CV protocols, dataset and shared utilities
│   ├── canonical_cv.py
│   ├── cross_validation_split.py
│   ├── cross_validation_dataset.py
│   ├── train.py
│   └── utils.py
├── docs/
│   └── research/         # Experiment history and reproducibility evidence
├── experiments/
│   ├── convnext/          # Reference and versioned training experiments
│   ├── panderm/           # PanDerm research scripts
│   └── metadata/          # Metadata-fusion experiments
├── evaluation/            # Evaluations, OOF ensemble and CV audit
├── preprocessing/            # Dataset preparation and validation
├── archive/
│   ├── diagnostics/
│   ├── legacy_experiments/
│   └── recovered_sources/ # Preserved historical snapshots and provenance
├── splits/                # Frozen canonical fold assignments
├── reports/               # Experiment results
├── HAM10000_metadata.csv
├── README.md
├── requirements.txt
├── data/                  # Local dataset; ignored by Git
├── checkpoints/           # Local model weights; ignored
├── cache/                 # Local generated cache; ignored
├── venv/                  # Main Python environment; ignored
└── PanDerm/               # External repository; ignored
```

Historical scripts are retained for research traceability. Some require their original Python environments, checkpoints and fold assignments.

## Installation (Windows / PowerShell)

From a new workspace:

```powershell
git clone https://github.com/Marsic228/Mole-Classification-HAM10000.git
cd Mole-Classification-HAM10000
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Important:** `requirements.txt` is an incomplete reproducibility specification for the full project. In particular, CUDA-enabled PyTorch and the independent PanDerm environment may need additional installation steps. Use historical environment manifests when reproducing historical runs. Do not assume that one environment reproduces both backbones.

Get the HAM10000 images separately and place the original JPG files in `data/images/`. The repository contains `HAM10000_metadata.csv`, but not the full image dataset. The preprocessing scripts (`preprocessing/preprocessing_config.py`, `preprocessing/preprocess_full.py`) produce processed train/validation/test folders using the original group-safe partition. The dataset loader currently expects:

```text
data/processed/{source_split}/{class_name}/{image_id}_224x224.jpg
```

For development folds, `source_split` is `train` or `val`. Check that preprocessing configuration and generated files exist before running training; the repository alone is not sufficient to run models.

## Frozen folds: new experiments only

The October 2026 audit found that the main ConvNeXt and original PanDerm Python environments produced different numeric folds, despite the same declared splitting settings. All 8,530 images and labels matched, but only **1,741 / 8,530** images had the same fold number across the environments. Package versions differ; their exact causal role has not been established.

### Canonical five-fold structure (new experiments)

Use **StratifiedGroupKFold (n_splits=5, shuffle=True, random_state=42)** with diagnosis label **dx** for stratification and **lesion_id** for grouping. Every development image is validated exactly once; train and validation have **zero shared lesions** within each fold.

| Fold | Training images | Validation images |
|---|---:|---:|
| 1 | 6,823 | 1,707 |
| 2 | 6,824 | 1,706 |
| 3 | 6,824 | 1,706 |
| 4 | 6,825 | 1,705 |
| 5 | 6,824 | 1,706 |

The frozen new-experiment development pool totals **8,530 images from 6,349 distinct lesions**. These are **canonical fold memberships**, not interchangeable with historical numeric PanDerm folds: PanDerm original validation sizes are 1,699 / 1,695 / 1,713 / 1,702 / 1,721. Preserve the saved original folds when replaying historical checkpoints.

For **future** experiments, use the audited ConvNeXt fold assignment recorded in:

- `splits/ham10000_dev_cv_v1.csv`
- `splits/ham10000_dev_cv_v1.json`

CSV SHA256:

```text
a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248
```

Validate it in the project root:

```powershell
.\venv\Scripts\python.exe -m core.canonical_cv verify
```

For **new** training/evaluation scripts:

```python
from core.canonical_cv import get_fold_metadata
train_metadata, val_metadata = get_fold_metadata(1)
```

**Do not change historic checkpoint folds.** Scripts used to reproduce existing ConvNeXt/PanDerm checkpoints may need their historical `core/cross_validation_split.py` and original Python environment. A matching fold number across environments does not imply a matching validation image set. Save and compare image/lesion IDs, rather than assuming numeric fold identity.

## PanDerm (third-party dependency)

PanDerm is kept as its **own local Git checkout**, not committed as a folder or submodule of this repository.

- Upstream: https://github.com/SiyuanYan1/PanDerm
- Historical source commit: `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`
- Local source: `PanDerm/classification/`
- Historical Python environment: `PanDerm/.venv-panderm/`
- Pretrained weights expected by `experiments/panderm/panderm_cv_train.py`: `PanDerm/classification/pretrain_weight/panderm_bb_data6_checkpoint-499.pth`

Obtain the pinned source:

```powershell
git clone https://github.com/SiyuanYan1/PanDerm.git PanDerm
git -C PanDerm checkout fd7a80748ba7fc3e203fed88f909f4689d0d6f24
```

Follow upstream installation and weight-acquisition instructions. The dedicated historical environment used different PyTorch, torchvision, scikit-learn and timm versions from the main environment. Pretrained weights are not part of this repository.

## Historical ConvNeXt + PanDerm OOF ensemble

Only run these commands when the original compatible model checkpoints, processed data and historical environments are available. They reproduce the historical evaluation, **not the new canonical-fold policy**:

```powershell
.\venv\Scripts\python.exe -m evaluation.export_model_oof_predictions --model convnext
.\PanDerm\.venv-panderm\Scripts\python.exe -m evaluation.export_model_oof_predictions --model panderm
.\venv\Scripts\python.exe -m evaluation.merge_model_oof_ensemble
```

The exporters generate each model's held-out probabilities using its **own original folds**. The merger aligns images by `image_id`, checks labels/class ordering, and averages corresponding class probabilities. Pooled results are saved in `reports/oof_convnext_panderm_ensemble_results.json` along with prediction arrays. Do not tune ensemble weights on a set and label the same-set result as unbiased independent performance.

## Experimental workflow and reproducibility rules

1. State a testable hypothesis and a controlled change.
2. Verify canonical folds for **new** experiments, and retain historical fold assignments for old checkpoints.
3. Run a sanity check/one-fold pilot before a full five-fold training run.
4. Preserve training configurations, seeds, environment versions, selected checkpoints, evaluation reports, per-class metrics and failed experiments.
5. Compare equivalent metrics on matched held-out images; separate fold-mean and pooled OOF summaries.
6. Keep raw images, private configuration, model weights and generated caches outside ordinary Git. Back them up separately.
7. Perform disjoint independent evaluation, input/domain-shift screening, confidence calibration and unsupported-input handling before considering a research application ready.

The intended application is a **research prototype**, not a medical diagnostic device.

## Code organization and research workflow

### Active research

- `experiments/convnext/convnext_tiny_cv_train_reference.py` — stable ConvNeXt v17 reference using the original historical CV assignments.
- `experiments/convnext/convnext_tiny_cv_train_natural_balsoftmax.py` — completed controlled Balanced Softmax experiment (rejected versus natural CE).
- `evaluation/convnext_natural_ce_vs_balsoftmax.py` — paired 10-checkpoint evaluation on the canonical development folds.
- The former generic `convnext_tiny_cv_train.py` workspace was removed; its v14 SAM contents are already preserved as `archive/legacy_experiments/convnext_tiny_cv_train_v14_sam.py`.
- `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` — completed natural-sampling Cross-Entropy control (mean fold macro F1: 0.7386 ± 0.0178).
- `experiments/convnext/convnext_tiny_cv_train_roi_margin15.py` — completed mask-assisted ROI experiment (canonical mean macro F1: 0.7512 ± 0.0174).
- `preprocessing/roi_crop.py` and `preprocessing/preview_roi.py` — conservative mask-based cropping and visual/quality preview.
- `evaluation/roi_fold_check.py` — independently replay ROI fold checkpoints and build the paired CE–ROI five-fold summary.
- `core/canonical_cv.py` and `splits/` — frozen lesion-safe folds for new experiments.
- `experiments/panderm/` and `experiments/metadata/` — model research scripts.
- `evaluation/` — independent evaluation, OOF aggregation and CV audit.
- `archive/` — historical experiments and diagnostic exercises.

See `archive/README.md` for archive details.

### Commands (Windows / PowerShell)

Run from the repository root:

```powershell
# Verify frozen development folds; does not train
.\venv\Scripts\python.exe -m core.canonical_cv verify

# Replay the CE vs Balanced Softmax comparison (requires saved checkpoints and data; inference only)
.\venv\Scripts\python.exe -m evaluation.convnext_natural_ce_vs_balsoftmax
```

The comparison command requires local images and all ten saved checkpoints; it does not retrain the models.

### Experiment management

1. Use canonical lesion-safe folds for **new** experiments, preserving historical fold assignments for existing checkpoints.
2. Change one major experimental factor at a time and give every experiment a unique `RUN_NAME`.
3. Preserve completed experiment source code, validation reports and unsuccessful results.
4. Compare metrics only under compatible evaluation protocols, separating fold means from pooled OOF results.
5. Update the stable reference only after properly validated improvements.

Do not overwrite historical experiment artifacts. Keep the local weights and image data backed up separately.

For detailed experiment results, see [Research Reports Index](reports/README.md).

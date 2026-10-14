# Mole Classification — HAM10000

Diploma research project for seven-class classification of dermatoscopic skin-lesion images. The work covers data preparation, class imbalance, transfer learning, group-safe cross-validation and complementary-model ensembles.

> **Research only:** Models are not clinically validated, cannot diagnose skin cancer, and must not replace a dermatologist. Smartphone photos or unrelated inputs may be outside the training distribution.

## Dataset and evaluation

- **HAM10000:** 10,015 records; classes `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- **Development pool:** 8,530 images covering 6,349 distinct `lesion_id` values (original train + validation partitions).
- **Validation:** five stratified, lesion-group-safe folds; zero overlapping lesions between training and validation for every fold.
- **Primary metric:** macro F1. Also report accuracy, per-class precision/recall/F1, confusion matrices and fold variability.
- **Test policy:** the old exposed test split is historical evidence only; do not use it to select models or claim independent final validation.

### Development results (as of 14 October 2026)

| System | Metric type | Macro F1 | Accuracy |
|---|---|---:|---:|
| ConvNeXt-Tiny v17, warmup + CutMix + EMA | Mean of five folds ± sample SD | **0.7430 ± 0.0144** | 0.8450 ± 0.0091 |
| PanDerm Base partial4 | Mean of five folds ± sample SD | **0.7368 ± 0.0232** | 0.8350 ± 0.0087 |
| ConvNeXt v17 | Pooled OOF, 8,530 images | **0.7444** | 0.8451 |
| PanDerm partial4 | Pooled OOF, 8,530 images | **0.7379** | 0.8351 |
| ConvNeXt v17 + PanDerm partial4, 50/50 probability averaging | Pooled matched OOF, 8,530 images | **0.7621** | 0.8570 |

**Mean fold F1 and pooled out-of-fold (OOF) F1 are different measures.** The ensemble is a promising development result, **not external or clinical validation**. Balanced Softmax/LDAM investigations are ongoing and are not included as established improvements.

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
├── experiments/
│   ├── convnext/          # Reference, workspace and completed experiments
│   ├── panderm/           # PanDerm research scripts
│   └── metadata/          # Metadata-fusion experiments
├── evaluation/            # Evaluations, OOF ensemble and CV audit
├── preprocessing/         # Dataset preparation and validation
├── archive/
│   ├── diagnostics/
│   └── legacy_experiments/
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
git clone https://github.com/Marsic228/Mole-Classification.git
cd Mole-Classification
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
- `experiments/convnext/convnext_tiny_cv_train.py` — editable experimental workspace (local, uncommitted experiment changes are deliberately kept outside documentation commits).
- `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` — completed natural-sampling Cross-Entropy control (mean fold macro F1: 0.7386 ± 0.0178).
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

# Launch the experimental ConvNeXt workspace (starts training)
.\venv\Scripts\python.exe -m experiments.convnext.convnext_tiny_cv_train
```

Do not run the second command until an experiment is configured and ready.

### Experiment management

1. Use canonical lesion-safe folds for **new** experiments, preserving historical fold assignments for existing checkpoints.
2. Change one major experimental factor at a time and give every experiment a unique `RUN_NAME`.
3. Preserve completed experiment source code, validation reports and unsuccessful results.
4. Compare metrics only under compatible evaluation protocols, separating fold means from pooled OOF results.
5. Update the stable reference only after properly validated improvements.

Do not overwrite historical experiment artifacts. Keep the local weights and image data backed up separately.

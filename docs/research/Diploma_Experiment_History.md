# Diploma Experiment History — Mole Classification / HAM10000

**Canonical experiment archive**  
**Reconstructed:** 2026-10-07; **evidence reconciliation + local static and preflight audits:** 2026-10-14 (GitHub `4088b9d`; bulk audit 16:50:35Z; reproducibility preflight 17:14:34Z)  
**Most recent experiment entry:** 2026-10-24 (mask-assisted ROI vs natural CE; D22).
**Repository:** `Marsic228/Mole-Classification-HAM10000`

## Quick navigation

These links point to entries in this registry; they do not duplicate model status or metrics. For current decisions, check dated revisions and later supplements in the linked sections.

- [[#D17. ConvNeXt v17 — one-epoch LR warmup — GITHUB-VERIFIED INDEPENDENT FIVE-FOLD EVALUATION — ACCEPTED STANDALONE REFERENCE|Standalone reference decision]]
- [[#D20. Natural sampling + Balanced Softmax — CHAT-VERIFIED FIVE-FOLD TRAINING AND PAIRED REPLAY — REJECTED AS CE REPLACEMENT|2026-10-19 controlled Balanced Softmax vs CE result]]
- [[#D21. Natural sampling + LDAM — CHAT-VERIFIED FIVE-FOLD TRAINING AND PAIRED REPLAY — REJECTED AS CE REPLACEMENT|2026-10-21 controlled LDAM vs CE result]]
- [[#D22. Mask-assisted ROI cropping — CHAT-VERIFIED FIVE-FOLD DEVELOPMENT CV — PROMISING|2026-10-24 ROI cropping vs CE; five-fold checkpoint replay]]
- [[#E3. ConvNeXt v17 + PanDerm partial4 equal-probability soft voting — GITHUB-VERIFIED OOF DEVELOPMENT RESULT — LEADING ENSEMBLE CANDIDATE|Ensemble candidate and fold-identity correction]]
- [[#D9. ConvNeXt v9 — full unfreezing, Adam lr=1e-4 — GITHUB-VERIFIED — REJECTED|Full-unfreezing counterexample (v9; v10 follows)]]
- [[#F2. PanDerm Base partial8 — GITHUB-VERIFIED REPORTS — COMPLETE / REJECTED AS NEW REFERENCE|Deeper PanDerm fine-tuning]]
- [[#B4. Confusion-aware Focal Loss, λ = 0.10 — GITHUB-VERIFIED — REJECTED|Recall versus global balance]]
- [[#Confirmed seven-stage research roadmap — 2026-10-15 (current execution order)|Current research-roadmap section]]
- [[#12c. 2026-10-14 complete five-fold checkpoint replays: 15/15 PASS|Checkpoint replay and remaining reproducibility limits]]
- [[#12. Items still to verify before diploma finalization|Items still to verify]]

## Purpose

This file is the permanent experiment registry for the diploma project. Unlike `Memory.md`, it should preserve **successful, rejected, diagnostic, and superseded experiments** so that later diploma writing does not depend on remembering old chats.

### Evidence labels

- **GITHUB-VERIFIED** — exact values recovered from committed JSON reports/code in `Marsic228/Mole-Classification`.
- **CHAT-VERIFIED** — exact values/results recovered from AI + English Project conversations or terminal output. This label describes the source status *at the time of the original entry*; later GitHub archiving is recorded in the 2026-10-14 evidence reconciliation.
- **LEGACY-HISTORICAL** — real project experiment, but it used the older train/validation/test workflow and is **not directly comparable** to the current lesion-safe 5-fold protocol.
- **PARTIAL** — experiment is known, but some exact values/configuration details still need recovery.
- **DIAGNOSTIC** — useful engineering/evaluation check, not a candidate model.

## Source hierarchy

For exact numerical values, use sources in this order:

1. committed GitHub JSON evaluation/CV reports;
2. terminal output captured in Project conversations for experiments not yet committed;
3. Project conversation summaries for rationale, configuration, and decisions;
4. legacy Claude memory for project origin / pre-modeling state.

If two sources disagree, preserve both until the reason is established. Do not silently overwrite historical results.

---

# 1. Evaluation-protocol eras

## Era A — early historical split

Early experiments used the original processed `train/val/test` folders and were useful for learning and model development. They are retained for the diploma research narrative, but **must not be compared numerically as if they were current CV results**.

The old held-out test was later exposed during development. Its results are historical only and the test set is excluded from current model selection.

## Era B — current controlled development protocol

From 2026-09-05 onward:

- development pool: **8530 images** (old train + validation);
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`;
- stratify by `dx`;
- group by `lesion_id`;
- verified lesion overlap between train/validation folds: `0`;
- primary metric: **mean macro F1 across 5 folds**;
- old exposed test set excluded from tuning/model selection.

This is the protocol used for the ResNet18 CV cycle, architecture screening, ConvNeXt lineage, metadata fusion, and PanDerm.

---

# 2. Project origin / pre-modeling foundation

The legacy Claude phase established the repository, HAM10000 metadata, reproducible environment, data-pipeline classes, and leakage awareness before serious modeling.

Important inherited facts:
- HAM10000 has 10,015 records and 7 classes.
- `lesion_id` can repeat across images, creating leakage risk.
- the earlier MATLAB diploma/slides reported an 8-class CNN and ~97.92% validation accuracy, but that historical claim was explicitly treated as **unverified** and not comparable to the current Python pipeline.

---

# 3. Era A — early CNN / imbalance / ResNet18 experiments

## A0. Limited custom-CNN debug baselines — LEGACY-HISTORICAL

### A0a. 100-batch debug run
**Source:** Project chat, 2026-06-29.  
- `train_max_batches=100`
- full model not yet treated as final evidence
- train loss: **1.188454**
- validation loss: **1.390114**
- validation accuracy: **0.4300**
- issue discovered: partial validation could be class-order biased when `shuffle=False`.

### A0b. corrected limited baseline with full validation
**Source:** `reports/baseline_experiment_report.json`  
- custom CNN
- 100 training batches
- full 1,512-image validation
- train loss: **1.193917**
- validation loss: **1.014911**
- validation accuracy: **0.680556**
- majority-class baseline accuracy: **0.680556**
- behavior: essentially collapsed to `nv`; only `bkl` had tiny nonzero F1 besides `nv`.
- status: diagnostic baseline, not final model quality.

---

## A1. Aggressive class-weighted CrossEntropy — LEGACY-HISTORICAL / CHAT-VERIFIED

**Goal:** counter severe imbalance using inverse-frequency-style class weighting.

Recovered historical result:
- final validation accuracy: **~0.5165**
- below majority baseline;
- minority recall sometimes increased, but precision/general performance collapsed;
- `df` remained unsolved.

Selected class metrics captured in chat:
- `mel`: P/R/F1 ≈ **0.3376 / 0.4463 / 0.3844**
- `bcc`: **0.1254 / 0.7818 / 0.2161**
- `vasc`: **0.1895 / 0.6667 / 0.2951**
- `nv`: **0.9094 / 0.5948 / 0.7192**

**Decision:** rejected; weighting was too aggressive.

---

## A2. Softer sqrt class-weighted CrossEntropy — LEGACY-HISTORICAL / CHAT-VERIFIED

**Change:** use square-root inverse-frequency class weights to reduce overcorrection.

Training:
- train loss: `1.485743 → 1.346105 → 1.231323`
- validation loss: `1.062002 → 0.936402 → 1.044920`
- validation accuracy: `0.674603 → 0.675926 → 0.593915`

Final selected class recall/F1 examples:
- `akiec`: R **0.0233**, F1 **0.0435**
- `bcc`: R **0.7091**, F1 **0.3071**
- `bkl`: R **0.5220**, F1 **0.3162**
- `df`: R/F1 **0 / 0**
- `mel`: R **0.1017**, F1 **0.1552**
- `nv`: R **0.7250**, F1 **0.7882**
- `vasc`: R **0.4074**, F1 **0.4231**

**Decision:** rejected.

---

## A3. WeightedRandomSampler + ordinary CrossEntropy — LEGACY-HISTORICAL / GITHUB-VERIFIED

**Sources:** `weighted_train.py`, `dataset_loader_check.py`, `reports/weighted_model_evaluation_report.json`

Configuration:
- baseline CNN;
- batch size `8`;
- `WeightedRandomSampler`;
- ordinary CrossEntropy;
- Adam-like project optimizer, LR `0.001`;
- 3 epochs.

Training trend:
- train/val loss: `1.4160/1.1263 → 1.1513/1.0095 → 0.9589/0.8901`
- validation accuracy reached **0.676587**

Final:
- accuracy: **0.676587**
- majority baseline: **0.680556**
- delta: **−0.003968**

Per-class F1:
- akiec **0.2857**
- bcc **0.3465**
- bkl **0.2713**
- df **0.0000**
- mel **0.2980**
- nv **0.8366**
- vasc **0.6222**

**Decision:** not sufficient, but sampler became an important later component.

---

## A4. Targeted augmentation for `df` + `mel` with sampler — LEGACY-HISTORICAL / GITHUB-VERIFIED

**Sources:** `augmented_sampler_train.py`, `dataset_loader_check.py`

Special transform for `df`/`mel`:
- random horizontal flip;
- random rotation ±10°;
- training sampler still weighted.

### Epoch 2 evaluation
**Source:** `reports/augmented_sampler_epoch_2_evaluation_report.json`
- accuracy: **0.697751**
- majority baseline: **0.680556**
- delta: **+0.017196**
- `df` F1: **0.0000**
- `mel` P/R/F1: **1.0000 / 0.0113 / 0.0223**

### Epoch 3 evaluation
**Source:** `reports/augmented_sampler_model_evaluation_report.json`
- accuracy: **0.682540**
- `df` F1: **0.0000**
- `mel` F1: **0.0000**

**Decision:** rejected. Accuracy briefly improved, but the target minority classes did not.

---

## A5. Pretrained ResNet18, mostly frozen — LEGACY-HISTORICAL / GITHUB-VERIFIED

**Source:** `reports/resnet18_model_evaluation_report.json`

- ImageNet-pretrained ResNet18;
- transfer-learning baseline;
- final validation accuracy: **0.714286**.

Per-class F1:
- akiec **0.4122**
- bcc **0.3623**
- bkl **0.3853**
- df **0.0625**
- mel **0.4488**
- nv **0.8619**
- vasc **0.5909**

**Decision:** transfer learning clearly better than the custom CNN path.

---

## A6. ResNet18 deeper fine-tuning — LEGACY-HISTORICAL / GITHUB-VERIFIED

### A6a. partial fine-tuning, intermediate configuration
**Source:** `reports/resnet18_finetune_epoch_2_model_evaluation_report.json`
- accuracy: **0.777116**
- F1: akiec **0.5106**, bcc **0.4860**, bkl **0.5890**, df **0.3871**, mel **0.5440**, nv **0.8821**, vasc **0.7234**

Epoch 3:
**Source:** `reports/resnet18_finetune_epoch_3_model_evaluation_report.json`
- accuracy: **0.778439**

### A6b. `layer3 + layer4 + fc` trainable
**Source:** `reports/resnet18_layer3_layer4_epoch_3_evaluation_report.json`
- validation accuracy: **0.792989**
- F1: akiec **0.4717**, bcc **0.5345**, bkl **0.5862**, df **0.4375**, mel **0.4983**, nv **0.8988**, vasc **0.7778**
- selected architecture for the next modeling cycle.

### Historical old test evaluation
**Source:** `reports/resnet18_layer3_layer4_epoch_3_test_report.json`
- accuracy: **0.808081**
- macro precision: **0.600457**
- macro recall: **0.643243**
- macro F1: **0.613486**
- F1: akiec **0.4884**, bcc **0.7030**, bkl **0.6201**, df **0.5556**, mel **0.4048**, nv **0.9164**, vasc **0.6061**

**Important:** this old test result is historical only; the test was later excluded from tuning/model selection.

### A6c. `layer2 + layer3 + layer4 + fc`
**Source:** `reports/resnet18_layer2_layer3_layer4_epoch_3_evaluation_report.json`
- accuracy: **0.796296**
- macro F1: **0.566843**
- F1: akiec **0.3871**, bcc **0.5546**, bkl **0.6044**, df **0.2667**, mel **0.5209**, nv **0.9069**, vasc **0.7273**

**Decision:** deeper unfreezing did not justify replacing the `layer3+layer4` setup.

---

## A7. ResNet18 discriminative learning rates — LEGACY-HISTORICAL / GITHUB-VERIFIED

**Source:** `resnet18_train.py`, `reports/resnet18_discriminative_lr_epoch_3_evaluation_report.json`

Learning rates:
- layer3: `1e-5`
- layer4: `3e-5`
- fc: `1e-4`

Epoch 3:
- accuracy: **0.787698**
- macro precision: **0.651857**
- macro recall: **0.542543**
- macro F1: **0.579057**
- mel F1: **0.5217**

**Decision:** rejected.

---

# 4. Era B — lesion-safe 5-fold ResNet18 cycle

## B0. ResNet18 + WRS + CrossEntropy reference — GITHUB-VERIFIED

Configuration:
- pretrained ResNet18;
- `layer3 + layer4 + fc` trainable;
- `WeightedRandomSampler`;
- CrossEntropy;
- Adam `lr=1e-4`;
- batch `8`;
- lesion-safe 5-fold SGKFold;
- 3 epochs.

CV:
- epoch 1 macro F1: **0.6080 ± 0.0238**
- epoch 2 macro F1: **0.6100 ± 0.0305** ← selected CE reference
- epoch 3 macro F1: **0.6087 ± 0.0201**

Epoch 2:
- accuracy: **0.7742 ± 0.0260**
- macro precision: **0.6375 ± 0.0382**
- macro recall: **0.6118 ± 0.0308**
- macro F1: **0.6100 ± 0.0305**

---

## B1. Focal Loss `gamma=2` — GITHUB-VERIFIED — ACCEPTED RESNET REFERENCE

Same setup as B0 except Focal Loss.

Verified epoch-3 recheck:
- accuracy: **0.7734 ± 0.0161**
- macro precision: **0.6436 ± 0.0246**
- macro recall: **0.6201 ± 0.0284**
- macro F1: **0.6229 ± 0.0152**

Per-class F1:
- akiec **0.4993 ± 0.0523**
- bcc **0.6316 ± 0.0403**
- bkl **0.5893 ± 0.0227**
- df **0.4970 ± 0.0459**
- mel **0.4769 ± 0.0556**
- nv **0.8824 ± 0.0135**
- vasc **0.7841 ± 0.0509**

**Decision:** became the ResNet18 controlled reference.

---

## B2. Mel-only targeted augmentation — GITHUB-VERIFIED — REJECTED

- accuracy: **0.7623 ± 0.0161**
- macro F1: **0.6090 ± 0.0196**
- mel F1: **0.4433 ± 0.0469**

**Decision:** rejected; target class worsened.

---

## B3. Focal alpha for mel = 1.25 — GITHUB-VERIFIED — REJECTED

- accuracy: **0.7716 ± 0.0302**
- macro F1: **0.6150 ± 0.0283**
- mel P/R/F1: **0.4935 / 0.4675 / 0.4669**

**Decision:** rejected.

---

## B4. Confusion-aware Focal Loss, λ = 0.10 — GITHUB-VERIFIED — REJECTED

Target: penalize true-mel probability assigned to `bkl`/`nv`.

- accuracy: **0.7600 ± 0.0275**
- macro F1: **0.6052 ± 0.0341**
- mel P/R/F1: **0.4384 / 0.5736 / 0.4876**

**Interpretation:** mel recall increased, but global balance worsened.

---

## B5. Confusion-aware Focal Loss, λ = 0.05 — GITHUB-VERIFIED — REJECTED

- accuracy: **0.7729 ± 0.0255**
- macro F1: **0.6112 ± 0.0307**
- mel F1: **0.4793 ± 0.0245**

**Decision:** rejected.

---

## B6. 3-class specialist (`bkl/mel/nv`) — GITHUB-VERIFIED — DIAGNOSTIC

Epoch 2:
- 3-class accuracy: **0.8218 ± 0.0116**
- 3-class macro F1: **0.6844 ± 0.0160**
- bkl F1: **0.6318 ± 0.0191**
- mel F1: **0.5202 ± 0.0483**
- nv F1: **0.9012 ± 0.0052**

The specialist was stronger for its restricted 3-class task, especially mel.

**2026-10-14 audit clarification:** All 15 specialist evaluation JSONs (`5 folds × 3 epochs`) intentionally have a 3×3 confusion matrix and `class_names=["bkl","mel","nv"]`. Their 30 static-audit notices are expected protocol differences, **not corrupt 7-class reports**. The archived specialist training source explicitly selects these three classes and maps `bkl:0`, `mel:1`, `nv:2`. These restricted-task scores must never be compared numerically as 7-class macro F1.

---

## B7. 7-class reference + specialist cascade — GITHUB-VERIFIED — REJECTED

- accuracy: **0.7795 ± 0.0137**
- macro F1: **0.6236 ± 0.0215**
- mel F1: **0.4817 ± 0.0425**

**Decision:** only tiny overall improvement over the ResNet reference; complexity not justified.

---

## B8. Auxiliary-loss experiment (`aux02`) — GITHUB-VERIFIED RESULT / PARTIAL CONFIG — REJECTED

**Source:** `reports/resnet18_focal_aux02_epoch_3_cv_summary.json`

- accuracy: **0.7817 ± 0.0157**
- macro precision: **0.6470 ± 0.0196**
- macro recall: **0.5911 ± 0.0362**
- macro F1: **0.5937 ± 0.0326**
- mel F1: **0.4501 ± 0.0815**

**Decision:** rejected.

**Archive note:** exact implementation details for the `aux02` term should be recovered from the corresponding historical commit before using it in diploma prose.

---

## B9. Hard-example sampling (`hard15`) — GITHUB-VERIFIED — REJECTED

Epoch 3 report:
- accuracy: **0.7725 ± 0.0232**
- macro precision: **0.6762 ± 0.0362**
- macro recall: **0.5772 ± 0.0504**
- macro F1: **0.5893 ± 0.0403**
- mel F1: **0.4448 ± 0.0915**

Historical configuration:
- hard examples mined from OOF behavior, especially `bkl/mel/nv`;
- top hard subset upweighted (project notes: hard ×1.5).

**Decision:** rejected.

**Archive note:** some earlier chat summaries referenced a different hard-example epoch/selection around macro F1 ~0.60. The committed epoch-3 report above is exact; if the diploma needs “best selected hard-example checkpoint,” reconstruct it from all epoch summaries rather than assuming epoch 3.

---

# 5. Architecture screening under lesion-safe CV

## C1. EfficientNet-B0 — GITHUB-VERIFIED — REJECTED

Best screening checkpoint: epoch 2.

- accuracy: **0.7675 ± 0.0150**
- macro F1: **0.6072 ± 0.0265**
- mel F1: **0.4633 ± 0.0239**

---

## C2. MobileNetV3-Large — GITHUB-VERIFIED — REJECTED

Best screening checkpoint: epoch 3.

- accuracy: **0.7682 ± 0.0331**
- macro F1: **0.6157 ± 0.0413**
- mel F1: **0.4892 ± 0.0431**

---

## C3. ConvNeXt-Tiny screening — GITHUB-VERIFIED — ACCEPTED SCREENING WINNER

Best screening checkpoint: epoch 2.

- accuracy: **0.8020 ± 0.0200**
- macro precision: **0.7134 ± 0.0424**
- macro recall: **0.6587 ± 0.0472**
- macro F1: **0.6601 ± 0.0281**

Per-class F1:
- akiec **0.5277**
- bcc **0.7050**
- bkl **0.6117**
- df **0.4963**
- mel **0.5413**
- nv **0.9031**
- vasc **0.8357**

**Decision:** move ConvNeXt to proper fine-tuning.

---

# 6. ConvNeXt controlled fine-tuning lineage

Common controlled protocol unless explicitly changed:
- pretrained ConvNeXt-Tiny;
- lesion-safe 5-fold SGKFold;
- WRS;
- Focal Loss `gamma=2`;
- batch `8`;
- Adam `lr=1e-4` unless changed;
- 224 pretrained transforms unless changed;
- best checkpoint by validation macro F1;
- early stopping patience `4`.

## D1. ConvNeXt v1 — longer fine-tuning + best checkpoint — GITHUB-VERIFIED — ACCEPTED

Major change vs screening:
- max `15` epochs/fold;
- best checkpoint by macro F1;
- early stopping.

Best epochs: `11, 4, 4, 7, 3`.

Aggregate:
- accuracy: **0.8161 ± 0.0049**
- macro precision: **0.7121 ± 0.0275**
- macro recall: **0.6842 ± 0.0517**
- macro F1: **0.6880 ± 0.0273**

Per-class F1:
- akiec **0.5803**
- bcc **0.7205**
- bkl **0.6262**
- df **0.5769**
- mel **0.5537**
- nv **0.9108**
- vasc **0.8476**

---

## D2. ConvNeXt v2 — AdamW — GITHUB-VERIFIED — REJECTED

Major change:
- Adam → `AdamW(lr=1e-4, weight_decay=1e-2)`.

Aggregate:
- accuracy: **0.8086 ± 0.0252**
- macro precision: **0.7316 ± 0.0425**
- macro recall: **0.6698 ± 0.0291**
- macro F1: **0.6876 ± 0.0376**

**Decision:** no improvement over v1.

---

## D3. ConvNeXt v3 — cosine scheduler — GITHUB-VERIFIED — ACCEPTED

Major change:
- `CosineAnnealingLR`.

Best fold F1 / epoch:
- F1 `0.7474` / e8
- F2 `0.6955` / e3
- F3 `0.6976` / e3
- F4 `0.6618` / e4
- F5 `0.6965` / e3

Aggregate:
- accuracy: **0.8127 ± 0.0264**
- macro precision: **0.7419 ± 0.0298**
- macro recall: **0.6858 ± 0.0406**
- macro F1: **0.6998 ± 0.0306**

---

## D4. ConvNeXt v4 — deeper partial fine-tuning (last 5 feature modules) — GITHUB-VERIFIED — ACCEPTED

Major change:
- trainable ConvNeXt feature modules increased to last `5` + classifier.

### Original CV summary
- macro F1: **0.7093 ± 0.0099**

### GPU + AMP validated rerun / conference value
- accuracy: **0.8238 ± 0.0101**
- macro precision: **0.7415 ± 0.0247**
- macro recall: **0.6943 ± 0.0182**
- macro F1: **0.7115 ± 0.0108**

Per-class F1:
- akiec **0.5920**
- bcc **0.7503**
- bkl **0.6556**
- df **0.6245**
- mel **0.5642**
- nv **0.9121**
- vasc **0.8818**

**Decision:** became reference at that stage.  
**Conference article remains frozen at this result.**

---

## D5. ConvNeXt v5 — differential learning rates — GITHUB-VERIFIED — REJECTED

Major change:
- earlier trainable feature modules: `2e-5`;
- later feature modules: `1e-4`;
- classifier: `1e-4`.

Aggregate:
- accuracy: **0.8241 ± 0.0123**
- macro F1: **0.7018 ± 0.0282**

**Decision:** rejected.

---

## D6. ConvNeXt v6 — mild global augmentation — GITHUB-VERIFIED — REJECTED

- accuracy: **0.8197 ± 0.0190**
- macro F1: **0.7088 ± 0.0303**
- mel F1: **0.5675 ± 0.0298**

**Decision:** rejected vs v4.

---

## D7. ConvNeXt v7 — higher resolution 288×288 — GITHUB-VERIFIED — REJECTED

- accuracy: **0.8190 ± 0.0277**
- macro F1: **0.6949 ± 0.0219**
- mel F1: **0.5661 ± 0.0111**

**Decision:** rejected; higher resolution did not help.

---

## D8. ConvNeXt v8 — MixUp α=0.2 — GITHUB-VERIFIED — REJECTED

Reconstructed from five committed fold evaluation files:
- accuracy: **0.8197 ± 0.0283**
- macro F1: **0.7013 ± 0.0269**

Fold macro F1:
- `0.7219, 0.7078, 0.7133, 0.6541, 0.7092`

**Decision:** rejected.

---

## D9. ConvNeXt v9 — full unfreezing, Adam lr=1e-4 — GITHUB-VERIFIED — REJECTED

- accuracy: **0.8145 ± 0.0189**
- macro F1: **0.7048 ± 0.0196**
- mel F1: **0.5567 ± 0.0389**

**Interpretation:** full unfreezing did not beat partial fine-tuning.

---

## D10. ConvNeXt v10 — full unfreezing, Adam lr=1e-5 — GITHUB-VERIFIED — REJECTED

- accuracy: **0.8183 ± 0.0068**
- macro F1: **0.6993 ± 0.0236**
- mel F1: **0.5705 ± 0.0346**

**Decision:** rejected.

---

## D11. ConvNeXt v11 — CutMix — GITHUB-VERIFIED — ACCEPTED

Back to partial last-5 fine-tuning.

Major change:
- CutMix `alpha=1.0`, probability `0.5`.

Aggregate:
- accuracy: **0.8328 ± 0.0063**
- macro precision: **0.7561 ± 0.0379**
- macro recall: **0.6959 ± 0.0397**
- macro F1: **0.7176 ± 0.0244**

Per-class F1:
- akiec **0.6134**
- bcc **0.7521**
- bkl **0.6646**
- df **0.6652**
- mel **0.5705**
- nv **0.9170**
- vasc **0.8408**

**2026-10-14 audit qualification:** Selected-best-validation summaries and separate fold evaluations for `convnext_tiny_finetune_v11_cutmix` disagree materially in **folds 1, 2 and 5** (full 7-class evaluation with 1,707/1,706 validation images). Best-summary vs evaluation macro F1: fold 1 `0.751912 → 0.736414` (−0.015498); fold 2 `0.720694 → 0.704486` (−0.016209); fold 5 `0.692059 → 0.711129` (+0.019070). The saved fold-1 epoch-6 report agrees with the best summary, so the difference is not just an erroneous copied best-epoch number. **Cause remains unknown.** Preserve both evaluations, do not silently reconcile them, and do not claim v11 exact checkpoint-reload reproducibility until original checkpoint IDs, evaluator transforms, data membership and precision are checked. Historical acceptance refers to its development-stage result and does not remove this caveat.

### Seed diagnostic
A seed-123 rerun produced a very similar overall mean but noticeable fold movement. Conclusion: both fold composition and stochastic training contribute to variability.

---

## D12. ConvNeXt v12 — CutMix probability 0.25 — GITHUB-VERIFIED — REJECTED

Major change:
- CutMix probability `0.5 → 0.25`.

- accuracy: **0.8132 ± 0.0180**
- macro F1: **0.7096 ± 0.0335**

**Decision:** rejected; `p=0.5` remained better.

---

## D13. ConvNeXt v13 — CutMix + EMA — GITHUB/PROJECT-VERIFIED — SUPERSEDED STANDALONE REFERENCE

Configuration:
- pretrained ConvNeXt-Tiny;
- last 5 feature modules + classifier trainable;
- WRS;
- Focal Loss γ=2;
- Adam `1e-4`;
- cosine scheduler `T_max=15`, `eta_min=1e-6`;
- batch 8;
- CutMix α=1, p=0.5;
- seed 123;
- EMA decay `0.999`, updated after every optimizer step;
- validation and checkpoint selection use EMA weights;
- max 15 epochs;
- patience 4.

Aggregate:
- accuracy: **0.8387 ± 0.0087**
- macro precision: **0.7602 ± 0.0272**
- macro recall: **0.7226 ± 0.0443**
- macro F1: **0.7365 ± 0.0343**

Per-class F1:
- akiec **0.6454 ± 0.0246**
- bcc **0.7722 ± 0.0156**
- bkl **0.6954 ± 0.0272**
- df **0.6699 ± 0.1253**
- mel **0.6030 ± 0.0342**
- nv **0.9181 ± 0.0035**
- vasc **0.8516 ± 0.0771**

Fold macro F1:
- **0.7803, 0.7173, 0.7640, 0.6981, 0.7229**

**Decision:** historical standalone image-only reference; superseded by v17 on 2026-10-10. Retained for original-fold comparisons.

---

## D14. ConvNeXt v14 — SAM + CutMix + EMA — GITHUB-VERIFIED — REJECTED AS STANDALONE

Configuration:
- v13-style pipeline;
- SAM `rho=0.05` with Adam base optimizer;
- first custom SAM+AMP attempt produced NaNs;
- successful run used FP32;
- computational cost was very high (~219 minutes for full work).

Aggregate:
- accuracy: **~0.8414 ± 0.0140**
- macro precision: **~0.8015 ± 0.0292**
- macro recall: **~0.6992 ± 0.0287**
- macro F1: **0.7370 ± 0.0310**

Fold macro F1:
- **0.7581, 0.7463, 0.7677, 0.6907, 0.7220**

**Decision:** essentially tied v13 but much more expensive; not a standalone replacement.

---

## D15. ConvNeXt v15 — v13 + v14 soft-voting ensemble — GITHUB/PROJECT-VERIFIED — HISTORICAL IMAGE-ENSEMBLE REFERENCE

Method:
- fold-matched v13 and v14;
- average class probabilities.

Aggregate:
- accuracy: **0.8470 ± 0.0119**
- macro precision: **0.7906 ± 0.0321**
- macro recall: **0.7156 ± 0.0339**
- macro F1: **0.7444 ± 0.0305**

Fold macro F1:
- **0.7685, 0.7443, 0.7760, 0.6995, 0.7336**

Per-class F1:
- akiec **0.6382**
- bcc **0.7886**
- bkl **0.7056**
- df **0.6648**
- mel **0.6096**
- nv **0.9232**
- vasc **0.8808**

**Decision:** best raw development score, but roughly doubles image inference cost.

---

## D16. Metadata fusion family — CHAT-VERIFIED

### D16a / D16b — retrained image-feature fusion heads — REJECTED
- attempts using image representation + metadata retraining;
- Fold-1 epoch-0 image reference macro F1 ≈ **0.7803**;
- trained best values around **0.7524 / 0.7556**;
- conclusion: retraining disturbed a strong image model and overfit.

### D16c — residual metadata correction — ACCEPTED LIGHTWEIGHT MULTIMODAL RESULT

Design:
- freeze v13 completely;
- use v13 image logits;
- metadata: age + sex + localization, 20-D processed vector;
- train-fold-only preprocessing;
- bias-free `Linear(20, 7)`;
- zero initialization;
- final logits = image logits + metadata correction;
- only **140 trainable parameters**.

Fold macro F1:
- **0.7852, 0.7240, 0.7688, 0.6981, 0.7364**

Delta vs v13:
- `+0.0049, +0.0068, +0.0048, +0.0000, +0.0135`

Aggregate:
- accuracy: **0.8435 ± 0.0105**
- macro precision: **0.7707 ± 0.0302**
- macro recall: **0.7248 ± 0.0450**
- macro F1: **0.7425 ± 0.0349**

Recovered per-class means:
- akiec **0.6504**
- bcc **0.7728**
- bkl **0.7002**
- df **0.6847**
- mel **0.6066**
- nv **0.9215**
- vasc: **0.8612** (mean of five F1 values: 0.925926, 0.782609, 0.916667, 0.833333, 0.847458). **Recovered 2026-10-14** from committed `reports/convnext_tiny_v16c_metadata_residual_evaluation.json`.

**Decision:** best lightweight multimodal result; improved macro F1 on 4/5 folds.

---


## D17. ConvNeXt v17 — one-epoch LR warmup — GITHUB-VERIFIED INDEPENDENT FIVE-FOLD EVALUATION — ACCEPTED STANDALONE REFERENCE

**Date:** 2026-10-10. **Evidence:** `convnext_tiny_cv_train_v17_warmup.py` and the locally run v17 training/evaluation terminal logs in this AI + English lesson. Run/report prefix: `convnext_tiny_finetune_v17_warmup_seed123`; best EMA checkpoints: `checkpoints/convnext_tiny_finetune_v17_warmup_seed123_fold{fold}_best.pt`. Independent evaluation reloaded the five selected checkpoints; **this was evaluation on the same five validation folds, not an external test**. Historical entry initially labelled CHAT-VERIFIED. **GitHub confirmation, 2026-10-14:** all five independent evaluation JSON reports and all five best-summary JSON reports are tracked under `reports/convnext_tiny_finetune_v17_warmup_seed123_fold{fold}_evaluation.json` / `_best_summary.json`. Reorganized reference code is committed at `experiments/convnext/convnext_tiny_cv_train_reference.py`. This establishes archival existence of code and metrics, not end-to-end GPU rerun reproducibility.

**Hypothesis:** a gradual first-epoch LR warmup may reduce abrupt early fine-tuning updates, improving macro F1 and/or fold stability relative to v13.

**Single intentional change vs v13:** linearly increase Adam LR `1e-5 → 1e-4` through epoch 1 (853 batches for folds 1/2/3/5, 854 for fold 4). Continue the existing epoch-stepped `CosineAnnealingLR(T_max=15, eta_min=1e-6)` thereafter. Keep ConvNeXt-Tiny pretrained weights, last five feature modules+classifier trainable, transforms, lesion-safe five folds, WRS, Focal Loss gamma 2, batch 8, CutMix alpha 1 / p .5, seed 123, AMP, EMA decay .999 updated each optimizer step, max 15 epochs, patience 4, best EMA checkpoint by validation macro F1.

| Fold | Best epoch | v17 training macro F1 | v17 independent macro F1 | v13 macro F1 | Independent v17 accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 3 | 0.7481 | 0.7481 | 0.7803 | 0.8459 |
| 2 | 4 | 0.7451 | 0.7451 | 0.7173 | 0.8476 |
| 3 | 8 | 0.7591 | 0.7586 | 0.7640 | 0.8517 |
| 4 | 5 | 0.7193 | 0.7195 | 0.6981 | 0.8293 |
| 5 | 10 | 0.7437 | 0.7437 | 0.7229 | 0.8505 |

**Independent five-fold aggregate (mean ± sample SD):**
- Accuracy **0.8450 ± 0.0091** (v13 **0.8387 ± 0.0087**)
- Macro precision **0.7787 ± 0.0115** (v13 **0.7602 ± 0.0272**)
- Macro recall **0.7217 ± 0.0228** (v13 **0.7226 ± 0.0443**)
- Macro F1 **0.7430 ± 0.0144** (v13 **0.7365 ± 0.0343**; approx **+0.0065** mean)
- Training-logs best-fold aggregate, prior to independent reload: **0.7431 ± 0.0146**. Do not conflate these rounded summaries.

**Per-class F1** (mean ± sample SD calculated from the independently evaluated, four-decimal per-fold class scores):

| Class | v13 mean F1 | v17 F1 (mean ± SD) | Delta mean F1 |
|---|---:|---:|---:|
| akiec | 0.6454 | 0.6411 ± 0.0574 | -0.0043 |
| bcc | 0.7722 | 0.7716 ± 0.0179 | -0.0006 |
| bkl | 0.6954 | 0.7005 ± 0.0396 | +0.0051 |
| df | 0.6699 | 0.6710 ± 0.0686 | +0.0011 |
| mel | 0.6030 | 0.6192 ± 0.0175 | +0.0162 |
| nv | 0.9181 | 0.9219 ± 0.0073 | +0.0038 |
| vasc | 0.8516 | 0.8757 ± 0.0390 | +0.0241 |

Mean melanoma recall from this evaluation ≈ **0.6044**; the system is **not clinically validated**. `df` F1 variability improved numerically (SD 0.1253 in v13 vs 0.0686 in v17). v17 improved mean F1 on 3/5 folds (fold 1 fell ≈0.0322), so the mean improvement and lower fold SD must not be described as proven statistical significance or external generalization.

**Precision/reproducibility note:** the best-epoch training scores in Fold 3 (`0.7591`) and Fold 4 (`0.7193`) differ slightly from independently reloaded evaluation (`0.7586`, `0.7195`). AMP/numerical precision differences could cause borderline argmax changes; the training script verifies autocast for validation, but the separate evaluation script was not supplied for comparison, so **do not claim AMP as the verified cause**. For future comparisons, standardize the evaluation precision and preprocessing. Use independent values above as the selected final results.

**Time:** Fold 1 7.1 minutes + folds 2–5 42.1 minutes ≈ 49.2 minutes on an RTX 2060.

**Decision (lesson closeout): ACCEPT v17 as the new image-only *development* reference.** This is a pragmatic decision under current macro-F1/per-class/cost criteria. v13 becomes historical standalone reference; v15 (v13+v14) retains the highest previous image-ensemble mean (0.7444 ± 0.0305), although v17 is close with single-model inference. v16c remains the reported lightweight multimodal result (0.7425 ± 0.0349). Next: only pursue layer-wise LR decay if methodologically distinct from rejected v5; then conditional checkpoint averaging and ConvNeXt+PanDerm complementary-error ensemble.

---

## D18. ConvNeXt v18 — layer-wise learning-rate decay (LLRD) — GITHUB-VERIFIED FIVE-FOLD REPORTS / SOURCE STILL TO PIN — REJECTED

**Date:** 2026-10-12. **Evidence:** uploaded `convnext_tiny_cv_train(20261012-073726).py`; five-fold training logs followed by independent per-class EMA checkpoint evaluation terminal output in AI + English conversation. Run prefix: `convnext_tiny_finetune_v18_llrd_seed123`. Historical note was written before the repository audit. **GitHub confirmation, 2026-10-14:** five independent `_evaluation.json` and five `_best_summary.json` reports exist under `reports/convnext_tiny_finetune_v18_llrd_seed123_*`. The exact dedicated v18 train script is **not present as a named tracked source file** in the reorganized repository; inspect saved local copies and preserve the precise script/version before claiming code-level replay. Checkpoint weights remain outside Git.

**Hypothesis:** gradual smaller LRs for earlier trainable ConvNeXt feature modules might preserve pretrained features and improve generalization over v17's uniform LR. Distinct from rejected v5's coarser two-rate scheme.

**Controlled change vs v17:** preserve same model architecture (last five feature modules+classifier), v17 warmup across first epoch, WRS, Focal gamma 2, Adam, cosine scheduler, CutMix alpha 1/p 0.5, EMA decay .999, seed 123, 5 lesion-safe folds, batch 8, maximum 15 epochs, patience 4, best EMA checkpoint selection by validation macro F1. Introduce layer-wise decay `LR_DECAY=0.8` with target feature module LRs `[4.096e-5, 5.12e-5, 6.4e-5, 8e-5, 1e-4]` (early→late), classifier `1e-4`; warmup factor applies separately to each optimizer group. Cosine scheduler keeps shared `eta_min=1e-6`, so ratios do not remain exactly constant over epochs.

| Fold | Selected epoch | v18 training best F1 | v18 independently evaluated F1 | v17 independently evaluated F1 |
|---|---:|---:|---:|---:|
| 1 | 3 | 0.7392 | 0.7395 | 0.7481 |
| 2 | 4 | 0.7488 | 0.7488 | 0.7451 |
| 3 | 4 | 0.7394 | 0.7394 | 0.7586 |
| 4 | 7 | 0.7203 | 0.7200 | 0.7195 |
| 5 | 10 | 0.7329 | 0.7329 | 0.7437 |

**Independent five-fold mean ± sample SD** (fold values in chat; computed from reported precision):
- Accuracy **0.8400 ± 0.0066** (v17 **0.8450 ± 0.0091**)
- Macro precision **0.7673 ± 0.0192** (v17 **0.7787 ± 0.0115**)
- Macro recall **0.7180 ± 0.0183** (v17 **0.7217 ± 0.0228**)
- Macro F1 **0.7361 ± 0.0106** (v17 **0.7430 ± 0.0144**, delta **−0.0069**)

Mean per-class independent v18 F1 (computed from rounded fold values): `akiec .6151`, `bcc .7563`, `bkl .7012`, `df .6951`, `mel .5950`, `nv .9202`, `vasc .8700`. Compared with v17 mean per-class F1: improvements concentrated in `df` (+.0241), `bkl` essentially tied (+.0007), melanoma declined (~−.0242), `akiec` declined (~−.0260). Mean validation macro F1 improved in only 2/5 folds versus v17. Training loss continued down while validation loss tended to rise, consistent with overfitting-like behavior; do not assert statistical significance from fold deltas.

**Precision note:** training log vs independent reload differs slightly in folds 1 and 4; use independent results for selected final v18 summary and don't assume a cause without evaluation-precision audit.

**Decision:** reject v18; keep **v17** as the standalone image-only development reference. Do not revisit LR decay without a materially new hypothesis.

---


## D19. Natural sampling + ordinary CrossEntropy control — GITHUB-VERIFIED FIVE-FOLD BEST-CHECKPOINT RESULTS — COMPLETED CONTROL

**Date:** 2026-10-14. **Evidence:** `experiments/convnext/convnext_tiny_cv_train_natural_ce.py`; `reports/convnext_tiny_natural_ce_seed123_fold{fold}_best_summary.json` for folds 1–5, plus 75 tracked reports for this run (epoch reports, training histories and best summaries). The implementation and best summaries were committed by 2026-10-14. **Later 2026-10-14** checkpoint replays (§12c) confirmed all five CE folds; the subsequent paired **2026-10-19** evaluation reproduced the CE checkpoints again alongside Balanced Softmax (D20). The original D19 status sentence was written before those replays. Original run prefix: `convnext_tiny_natural_ce_seed123`.

**Reason for control:** establish a natural-sampling + ordinary-CE baseline for a subsequent **natural-sampling + Balanced Softmax** comparison. This is **not** an isolated loss comparison against v17: vs v17's WRS + Focal Loss, *both sampling and loss* change. The fixed canonical development folds were used for this new run; the older v17 evaluation uses historical folds, and exact ID-level comparability needs to be checked if claiming paired fold deltas.

**Known configuration:** pretrained ConvNeXt-Tiny; last five feature modules and classifier trainable; batch 8; `DataLoader(shuffle=True)` without WRS; `nn.CrossEntropyLoss()`; Adam LR `1e-4`; epoch-one linear warmup (10%→100%); cosine scheduling; CutMix, EMA decay `0.999`, AMP, seed `123`; best checkpoint chosen by validation macro F1 with early stopping. Frozen canonical split: `core.canonical_cv.get_fold_metadata` based on `splits/ham10000_dev_cv_v1.csv`.

| Fold | Best epoch | Best validation macro F1 |
|---|---:|---:|
| 1 | 7 | 0.761806 |
| 2 | 11 | 0.734747 |
| 3 | 12 | 0.746955 |
| 4 | 9 | 0.713179 |
| 5 | 7 | 0.736546 |

**Mean ± sample SD of five selected training/validation best summaries:** macro F1 **0.738647 ± 0.017843** (rounded **0.7386 ± 0.0178**). **Mean validation accuracy = 0.847828** (84.78%), recovered directly from the five selected-epoch JSON reports (`..._fold{fold}_epoch_{best_epoch}_report.json`); individual fold accuracies: `0.859988, 0.851700, 0.849355, 0.829912, 0.848183`. These are selected-validation results, **not independent checkpoint-reload evaluations**. For context, v17 independent mean F1 was **0.7430 ± 0.0144** under its historical folds; the raw mean difference is ~−0.0044 but **do not attribute this difference solely to the loss function** or claim significance.

**Decision:** completed and retained as the **Balanced Softmax control**, not promoted to standalone reference. Next controlled test: keep natural sampling, folds and the remaining training recipe fixed, replace only ordinary CE with Balanced Softmax, then compare on the same validation identities. Model checkpoints and environment are not embedded in Git.

---


## D20. Natural sampling + Balanced Softmax — CHAT-VERIFIED FIVE-FOLD TRAINING AND PAIRED REPLAY — REJECTED AS CE REPLACEMENT

**Date:** 2026-10-19. **Evidence:** user-provided local training logs for Balanced Softmax folds 1–5 and terminal output from `evaluation/convnext_natural_ce_vs_balsoftmax.py`. Local source scripts: `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` and `experiments/convnext/convnext_tiny_cv_train_natural_balsoftmax.py`. **Saved evaluation artifact reported by the user:** `reports/convnext_natural_ce_vs_balsoftmax_summary.json`; per-fold reports: `reports/convnext_tiny_natural_{ce,balsoftmax}_seed123_fold{k}_evaluation.json`. These reports are **reported as saved locally**, not independently read from GitHub or confirmed committed. This entry is **CHAT-VERIFIED** rather than GitHub-verified.

**Hypothesis:** replacing ordinary CE with class-aware Balanced Softmax under **natural sampling** might raise macro F1 and minority-class performance without changing the backbone or sampling method.

**Matched controlled setup:** Same 8,530-image canonical five-fold, lesion-group-safe validation identities (via `core.canonical_cv`); ImageNet-pretrained ConvNeXt-Tiny with last five feature modules + classifier trainable; natural shuffled batches without WeightedRandomSampler; ImageNet transforms; CutMix (alpha 1, probability 0.5); EMA 0.999; Adam LR `1e-4`; first-epoch warmup; cosine scheduler; AMP; batch size 8; seed 123; max 15 epochs, patience 4, best EMA checkpoint selected by validation macro F1. **Major experimental change:** training `CrossEntropyLoss` → `BalancedSoftmaxLoss`, implemented as `cross_entropy(logits.float() + log(train_fold_class_counts), targets)`; training class counts come only from that training fold. Validation loss uses plain CE; evaluation/classification uses **raw logits** without adding counts. Independent run prefixes prevent checkpoint/report overwrites. Separate fold-1 and folds-2–5 invocations were reported; do not infer the final saved `__main__` loop from the locally uploaded copy alone.

**Selected-best validation macro F1 and independently replayed fold results** (printed to four decimals; CE fold-selected original values are more precise in D19):

| Canonical fold | CE best epoch / F1 | Balanced Softmax best epoch / F1 | BS−CE F1 (approx.) |
|---|---:|---:|---:|
| 1 | 7 / 0.7618 | 11 / 0.7549 | −0.0069 |
| 2 | 11 / 0.7347 | 14 / 0.7348 | +0.0001 |
| 3 | 12 / 0.7469 | 15 / 0.7277 | −0.0193 |
| 4 | 9 / 0.7132 | 15 / 0.7072 | −0.0060 |
| 5 | 7 / 0.7366 | 11 / 0.7228 | −0.0137 |

**Paired five-fold aggregate** (mean ± **sample SD**, not pooled OOF; Δ = Balanced Softmax minus CE):

| Metric | CE | Balanced Softmax | Paired Δ |
|---|---:|---:|---:|
| Accuracy | **0.8478 ± 0.0110** | 0.8408 ± 0.0086 | **−0.0070 ± 0.0070** |
| Macro precision | **0.7725 ± 0.0233** | 0.7284 ± 0.0253 | **−0.0441 ± 0.0117** |
| Macro recall | 0.7184 ± 0.0281 | **0.7390 ± 0.0318** | **+0.0206 ± 0.0051** |
| **Macro F1 (selection metric)** | **0.7386 ± 0.0178** | 0.7295 ± 0.0175 | **−0.0092 ± 0.0075** |

**Per-class mean validation F1** (means of five individual-fold class F1 scores; rounded user-reported values):

| Class | CE F1 | Balanced Softmax F1 | BS−CE |
|---|---:|---:|---:|
| akiec | 0.6470 | 0.6317 | −0.0152 |
| bcc | 0.7764 | 0.7816 | +0.0052 |
| bkl | 0.6943 | 0.6925 | −0.0017 |
| df | 0.6622 | 0.6628 | +0.0005 |
| mel | **0.5962** | 0.5808 | **−0.0154** |
| nv | 0.9269 | 0.9243 | −0.0026 |
| vasc | **0.8676** | 0.8328 | **−0.0349** |

**Replay integrity:** All **10 of 10** selected CE/BS checkpoints reloaded on their canonical validation folds and passed best-summary macro-F1 comparison. The printed absolute replay F1 discrepancies range from **0.000004 to 0.000026**, within the predeclared `0.003` replay tolerance. This supports selected-checkpoint evaluation reproducibility, **not** end-to-end retraining determinism or independent external validation.

**Interpretation and decision:** BS increased **macro recall (+0.0206)** at the cost of substantially lower **macro precision (−0.0441)**; mean macro F1 decreased **0.0092**, and `mel` F1 declined **0.0154** (melanoma F1 lower in **4/5** folds). Fold 2 was essentially tied, while the others declined. Fold SD was similar; no defensible stability benefit. **Reject Balanced Softmax as a replacement for natural CE under this configuration**, while preserving the informative precision–recall trade-off as a negative research result. This is a **controlled loss comparison only against natural CE**, not a causal comparison with v17 (which also differs in sampler/loss and historical fold identities).

**Next research action:** evaluate a separately controlled **LDAM** loss against the same natural CE baseline after validating its training-loss interaction with CutMix and class counts. No LDAM training result is yet claimed. The old exposed test cannot be used as a fresh independent test, and the research-only prototype has no clinical validation.

---

## D21. Natural sampling + LDAM — CHAT-VERIFIED FIVE-FOLD TRAINING AND PAIRED REPLAY — REJECTED AS CE REPLACEMENT

**Date:** 2026-10-21. **Evidence:** user-supplied Fold 1 and Folds 2–5 GPU training logs plus paired independent selected-checkpoint evaluation terminal output for natural CE and LDAM. **Reported locally saved** `reports/convnext_natural_ce_vs_ldam_summary.json`; exact evaluated checkpoints identified by run prefixes `convnext_tiny_natural_ce_seed123` and `convnext_tiny_natural_ldam_m05_s30_seed123`. The actual locally edited LDAM source and evaluator commit/paths **must be confirmed before claiming GitHub verification**. This is **CHAT-VERIFIED**, not independently confirmed in GitHub.

**Hypothesis:** under natural sampling, class-frequency-dependent LDAM margins might improve seven-class macro F1 and minority-class classification relative to natural-sampling ordinary Cross-Entropy (D19). **Scientific boundary:** LDAM's classification-margin mechanism and scaling come from Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (NeurIPS 2019), but this experiment tests **unweighted LDAM without Deferred Re-Weighting (DRW)**, not the full LDAM-DRW recipe.

**Controlled configuration:** Same 8,530-image canonical lesion-group-safe development folds and seven-class order `akiec,bcc,bkl,df,mel,nv,vasc`; natural shuffled training batches (no WRS); ImageNet-pretrained ConvNeXt-Tiny, last five feature modules and classifier trainable; same transforms; CutMix (alpha 1, p=0.5, area-adjusted mixing weight), EMA 0.999, Adam lr `1e-4`, first-epoch 0.1→1.0 LR warmup, cosine schedule, AMP, batch 8, seed 123, at most 15 epochs, early stopping patience 4, best EMA checkpoint by validation macro F1. **Deliberate loss-method change:** CE → LDAM with class margin `m_j = 0.5 * n_j^(-1/4) / max_k(n_k^(-1/4))`, logit scale `s=30`, fold-**training-only** class counts. CutMix training loss = area-weighted combination of the LDAM losses for both labels. Validation loss uses plain CE, and classification/replay uses **raw unadjusted logits**. Compared with CE, this loss-method change includes both target-logit margin **and scaling**; it does not isolate the margin's standalone effect. Do **not** compare the numerical training-loss magnitudes of LDAM and CE as though the functions had the same scale.

**Fold-level selected checkpoints and independently replayed macro F1** (printed four decimals):

| Canonical fold | CE best epoch / macro F1 | LDAM best epoch / macro F1 | LDAM − CE F1 |
|---|---:|---:|---:|
| 1 | 7 / 0.7618 | 13 / 0.7611 | −0.0007 |
| 2 | 11 / 0.7347 | 11 / 0.7297 | −0.0050 |
| 3 | 12 / 0.7469 | 9 / 0.7335 | −0.0134 |
| 4 | 9 / 0.7132 | 5 / 0.7048 | −0.0084 |
| 5 | 7 / 0.7366 | 7 / 0.7247 | −0.0119 |

**Paired 5-fold summary** (user-reported means ± **sample SD** across folds, not pooled OOF; Δ = LDAM minus CE; paired Δ SD is the SD of per-fold differences, not a difference of SDs):

| Metric | CE | LDAM | Paired Δ |
|---|---:|---:|---:|
| Accuracy | 0.8478 ± 0.0110 | 0.8423 ± 0.0047 | −0.0055 ± 0.0072 |
| Macro precision | 0.7725 ± 0.0233 | 0.7634 ± 0.0185 | −0.0091 ± 0.0308 |
| Macro recall | 0.7184 ± 0.0281 | 0.7176 ± 0.0307 | −0.0008 ± 0.0150 |
| **Macro F1** | **0.7386 ± 0.0178** | 0.7308 ± 0.0202 | **−0.0079 ± 0.0051** |

**Per-class mean validation F1** (reported means of class-specific scores on the five folds; displayed numbers rounded independently):

| Class | CE F1 | LDAM F1 | LDAM − CE |
|---|---:|---:|---:|
| akiec | 0.6470 | 0.6582 | +0.0112 |
| bcc | 0.7764 | 0.7795 | +0.0031 |
| bkl | 0.6943 | 0.6794 | −0.0148 |
| df | 0.6622 | 0.6365 | −0.0257 |
| mel | 0.5962 | 0.5990 | +0.0027 |
| nv | 0.9269 | 0.9229 | −0.0039 |
| vasc | 0.8676 | 0.8399 | −0.0278 |

**Replay integrity:** All **10/10** CE/LDAM selected checkpoints passed the local paired evaluation check, with reported absolute macro F1 differences ranging approximately `0.000001`–`0.000026`, below the predeclared `0.003` checkpoint replay tolerance. This supports **selected-checkpoint inference replay on development validation folds**, not identical fresh retraining, external testing, or clinical validity.

**Interpretation:** LDAM decreased macro F1 in **all 5 folds**; mean macro precision and accuracy declined, while macro recall was nearly unchanged. `mel` mean F1 improved by only ~0.0027 (positive in 3/5 folds), too small and inconsistent to justify a replacement; `df`, `vasc`, and `bkl` declined more notably. Fold-to-fold F1 SD did not improve. **Decision: reject LDAM (`m_max=0.5`, `s=30`, without DRW) as a replacement for the natural CE control.** Do not generalize this conclusion to all LDAM recipes, DRW variants, or datasets. It is not a paired causal comparison with v17, which differs in loss, sampler, and historical split membership.

**Research priority after D21:** the initial planned long-tail loss series is complete (CE D19, Balanced Softmax D20, LDAM D21). Shift to Stage 3 ROI/segmentation *feasibility* and one controlled lesion-focused input hypothesis; performance improvement remains unproven. Do not exhaustively tune LDAM parameters without new evidence. Preserve the original article's limitations and this negative result for diploma synthesis.

---

## D22. Mask-assisted ROI cropping — CHAT-VERIFIED FIVE-FOLD DEVELOPMENT CV — PROMISING

**Date:** 2026-10-24. **Evidence:** the user's CUDA training and independent replay terminal logs for all five ROI folds; locally generated and staged training/evaluation JSON, `reports/convnext_tiny_natural_ce_roi_margin15_seed123_vs_natural_ce_5fold_summary.json`; the baseline's committed per-fold `reports/convnext_tiny_natural_ce_seed123_fold{1..5}_evaluation.json`. At writing time the ROI code/reports were **staged locally but the commit/push had not been verified**. Evidence label is **CHAT-VERIFIED** until committed files are independently inspected.

**Hypothesis:** mask-assisted lesion-centered cropping may increase the lesion's effective resolution in ConvNeXt's fixed-size input and reduce irrelevant background, improving classification macro F1. Potential downside: lost perilesional context and segmentation-mask errors. This experiment changes **input preprocessing only** against the canonical natural-sampling CE control (D19); it does not compare directly and causally against historical v17, whose training recipe and fold provenance differ.

**Data and input preprocessing:** 8,530 HAM10000 development images from 6,349 lesions. Every development `image_id` matched a `Masks_to_IsicId_mapping.csv` row and a local PNG lesion mask in `Masks.zip`; original image and mask both 600×450 in inspected examples. `preprocessing/roi_crop.py` uses binary mask threshold >127, a lesion bounding box, margin=0.15 per side, `min_fraction=0.35`, `max_area=0.60`, original aspect-ratio preservation, and a whole-image fallback if the mask is empty, mask area >60%, the mask touches a frame boundary, or the expanded crop cannot fit. Outputs are resized to 224×224 and saved separately under `data/processed_roi_margin15_v1/{source_split}/{dx}/{image_id}_224x224.jpg`; original `data/processed` is preserved.

**Dataset-wide preprocessing audit:** 8,530/8,530 exported; no missing image/mask files in the development pool. Crop statuses: `roi` **6,232 (73.1%)**, `fallback_large_crop` **1,749 (20.5%)**, `fallback_large_area` **359 (4.2%)**, `fallback_edge` **190 (2.2%)**, `fallback_empty` **0**. Total fallback: **2,298 (26.9%)**. ROI rates differed by class (akiec 57.3%, bcc 82.7%, bkl 65.4%, df 68.4%, mel 59.4%, nv 76.5%, vasc 86.9%); rules were class-agnostic, but their effects were not class-uniform. The fallback thresholds are **exploratory heuristics**, not validated segmentation-quality rules.

**Controlled training protocol:** ConvNeXt-Tiny with ImageNet pretrained weights; last five feature modules and classifier trainable; seven-class order `akiec,bcc,bkl,df,mel,nv,vasc`; canonical lesion-group-safe frozen folds `splits/ham10000_dev_cv_v1.csv`; natural shuffled training sampling; ordinary CE; CutMix (p=0.5, alpha=1.0); EMA 0.999; Adam lr 1e-4; LR warmup (first epoch 0.1× → 1×) and cosine scheduling; batch size 8; seed 123; maximum 15 epochs and early stopping patience 4; selected EMA checkpoints by held-out macro F1. Both original- and ROI-image variants use the same model transforms and fold assignments. Unique ROI prefix: `convnext_tiny_natural_ce_roi_margin15_seed123`.

**Selected checkpoints and independent replay macro F1** (precision from final 5-fold JSON summary; all scores are *per-fold* validation scores, not pooled OOF):

| Canonical fold | Natural CE macro F1 | ROI macro F1 | ROI − CE | ROI best epoch |
|---|---:|---:|---:|---:|
| 1 | 0.761814 | 0.777217 | +0.015403 | 14 |
| 2 | 0.734743 | 0.734088 | −0.000655 | 5 |
| 3 | 0.746943 | 0.756871 | +0.009928 | 7 |
| 4 | 0.713171 | 0.736785 | +0.023614 | 8 |
| 5 | 0.736571 | 0.751009 | +0.014438 | 12 |

**Matched five-fold summary** (mean ± **sample SD**; Δ is the fold-paired difference, not the difference of standard deviations):

| Metric | Natural CE | ROI + CE | ROI − CE |
|---|---:|---:|---:|
| Accuracy | 0.8478 ± 0.0110 | **0.8548 ± 0.0057** | +0.0070 (approx.) |
| Macro precision | 0.7725 ± 0.0233 | **0.7894 ± 0.0212** | +0.0170 (approx.) |
| Macro recall | 0.7184 ± 0.0281 | **0.7288 ± 0.0348** | +0.0103 (approx.) |
| **Macro F1** | 0.7386 ± 0.0178 | **0.7512 ± 0.0174** | **+0.01255 ± 0.00888** |

**Per-class mean F1** (five-fold means; independently rounded in terminal below; for precise data use the saved JSON summary):

| Class | CE F1 | ROI F1 | ROI − CE |
|---|---:|---:|---:|
| akiec | 0.6470 | 0.6804 | +0.0334 |
| bcc | 0.7764 | 0.7809 | +0.0045 |
| bkl | 0.6943 | 0.7185 | +0.0243 |
| df | 0.6622 | 0.6817 | +0.0195 |
| mel | 0.5962 | 0.6145 | +0.0182 |
| nv | 0.9269 | 0.9285 | +0.0016 |
| vasc | **0.8676** | 0.8539 | **−0.0137** |

**Melanoma:** mean recall increased 0.5427 → 0.5612 and F1 increased 0.5962 → 0.6145; this is a development improvement, but recall remains limited. The `vasc` regression must be preserved in later interpretation. Improvements appeared in **four out of five folds**, and the overall fold SD was similar, so this does not prove improved stability or statistical significance.

**Independent inference replay:** `evaluation/roi_fold_check.py` reloaded all five ROI selected EMA checkpoints on the canonical held-out folds; **5/5 replay F1 values matched the saved best-epoch scores exactly** to the shown precision (absolute difference 0.000000). Each replay saved exact per-fold per-class precision/recall/F1 and the evaluator generated the five-fold paired comparison JSON automatically. Checkpoint replay is **not** reproducible retraining from scratch.

**Artifacts / paths:** `preprocessing/roi_crop.py`, `preprocessing/preview_roi.py`, `experiments/convnext/convnext_tiny_cv_train_roi_margin15.py`, `evaluation/roi_fold_check.py`, `Masks_to_IsicId_mapping.csv`, `roi_mask_quality_check.csv`, five ROI fold evaluations/histories/best summaries, and `reports/convnext_tiny_natural_ce_roi_margin15_seed123_vs_natural_ce_5fold_summary.json`. Raw masks in local `Masks.zip`, generated processed ROI images, and `checkpoints/` must remain outside ordinary Git; reliable reproduction requires obtaining those data artifacts separately.

**Causal and deployment boundaries:** The experiment establishes an **association under a controlled preprocessing comparison** using provided (mask-assisted/oracle) lesion boundaries. It does **not** validate mask generation on unseen inputs, segmentation quality (Dice/IoU), inference cost of an automatic segmenter, error propagation, independent external generalization, clinical performance, or statistical significance. Externally produced masks may have their own training/data provenance; this should be checked before stronger claims.

**Decision:** **PROMISING / RETAIN AS ROI DEVELOPMENT CANDIDATE**, not a new universally accepted image-only standalone reference. Preserve original CE and historical v17 references; proceed with Stage 3 segmentation feasibility rather than treating the 0.7512 result as a deployable model.

**Next research action:** audit mask shapes/components and quality, investigate confidence/uncertainty of **predicted** masks, define improved fallback rules, and design a segmentation evaluation using independent mask predictions (Dice/IoU) before measuring actual end-to-end classification and inference cost. One-major-factor controlled comparisons should be maintained.

---

# 7. Inference / stability experiments

## E1. Horizontal-flip TTA — CLOSED

Tested on ConvNeXt lineage, including v11 stability work.

Across original v11 + seed123 diagnostic:
- without TTA: approximately **0.7186 ± 0.0258**
- with horizontal-flip TTA: approximately **0.7170 ± 0.0202**

**Interpretation:** slight stability improvement, no consistent mean macro-F1 improvement.

## E2. Additional 90° rotation TTA — REJECTED

No reason to retain in preferred inference path.

---

## E3. ConvNeXt v17 + PanDerm partial4 equal-probability soft voting — GITHUB-VERIFIED OOF DEVELOPMENT RESULT — LEADING ENSEMBLE CANDIDATE

**Date:** 2026-10-12. **Evidence:** user terminal exporter/merger output plus attached `oof_convnext_panderm_ensemble_results.json` and `oof_convnext_panderm_ensemble_probabilities.npz` (local report path `reports/oof_convnext_panderm_ensemble_results.json`); original `panderm_cv_train.py`, PanDerm evaluator, and new OOF exporters in this AI + English conversation. As of 2026-10-14, the JSON `reports/oof_convnext_panderm_ensemble_results.json`, prediction archives `reports/oof_convnext_original_folds.npz`, `reports/oof_panderm_original_folds.npz`, `reports/oof_convnext_panderm_ensemble_probabilities.npz`, and exporters under `evaluation/` are committed. This is **pooled out-of-fold development evaluation**, not independent clinical/external test. **Invalid predecessor:** `reports/convnext_v17_panderm_partial4_ensemble_summary.json` contains 0.9323 macro F1 but its own `all_reference_checks_passed` and fold `reference_check_passed` are false; that value must never enter validated comparisons.

**Hypothesis:** ConvNeXt and dermatology-pretrained PanDerm make partially complementary classification mistakes. Equal-weight averaging of their class probabilities may improve pooled macro F1, including minority classes. This is **prediction averaging**, not weight/checkpoint averaging; no new training occurred.

**Critical environment/split discovery:** Initially the ensemble evaluator loaded PanDerm on main `venv` Fold 1 and produced a **spurious** F1 `0.9411`, despite the recorded PanDerm checkpoint's Fold-1 F1 `0.746388`. Checkpoint epoch 8/metrics/head weights matched; three different evaluation routines agreed on the spurious result in `venv`. A diagnostic in original `PanDerm/.venv-panderm` reproduced exactly `0.746388` and accuracy `0.839317`, with **different fold members**:
- main `venv` fold 1: train 6,823, val 1,707; validation class counts `akiec 57, bcc 86, bkl 186, df 20, mel 195, nv 1137, vasc 26`;
- original PanDerm `.venv-panderm` fold 1: train 6,831, val 1,699; validation class counts `akiec 48, bcc 91, bkl 194, df 19, mel 176, nv 1139, vasc 32`.

Two environments produced different folds under the same numeric fold designation. Likely package-version / `StratifiedGroupKFold`-implementation difference is **not yet confirmed as the root cause**. This invalidated earlier naive same-numbered-fold ensembles; **discard 0.9411 and 0.9323** as scores.

**Correction:** Export all 5-fold held-out class-probability vectors separately using each model's **own fold assignments, checkpoint, preprocessing and original Python environment**, then align by `image_id` and verify true labels. Exactly **8,530** image IDs/labels matched; only **1,741/8,530** rows had the same numeric fold ID across models, so numeric-fold matching is not appropriate. Each OOF model prediction is from a checkpoint that excluded its own validation lesion group. Probability average `0.5 × ConvNeXt + 0.5 × PanDerm` chosen without weight-search tuning.

**Pooled 8,530-image OOF results:**

| System | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| ConvNeXt v17 | 0.845135 | 0.775370 | 0.721572 | **0.744418** |
| PanDerm partial4 | 0.835053 | 0.754940 | 0.724938 | **0.737871** |
| 50/50 ensemble | **0.856975** | **0.795758** | **0.736934** | **0.762088** |

Gain vs ConvNeXt pooled: accuracy **+0.011841** (~101 additional correctly classified images), macro F1 **+0.017670**. These are **pooled metrics**, not arithmetic means of five fold scores (v17 original mean **0.7430 ± 0.0144**; comparing directly to pooled `0.7621` would mix metrics).

**Pooled per-class F1:**

| Class | ConvNeXt | PanDerm | Ensemble | Ensemble − ConvNeXt |
|---|---:|---:|---:|---:|
| akiec | 0.6429 | 0.6316 | 0.6531 | +0.0102 |
| bcc | 0.7724 | 0.7570 | 0.7991 | +0.0267 |
| bkl | 0.7022 | 0.7026 | 0.7356 | +0.0334 |
| df | 0.6747 | 0.6628 | 0.6909 | +0.0162 |
| mel | 0.6193 | 0.5984 | 0.6347 | +0.0154 |
| nv | 0.9221 | 0.9159 | 0.9280 | +0.0059 |
| vasc | 0.8775 | 0.8968 | 0.8933 | +0.0158 |

**Error complementarity:** ConvNeXt incorrect/PanDerm correct = **486** examples; ConvNeXt correct/PanDerm incorrect = **572**; both incorrect **835**. The 50/50 ensemble corrected **313** ConvNeXt mistakes and introduced **212** new errors (net +101 correct). These correction numbers were computed from the attached aligned probability arrays during the project analysis, not directly stored in the JSON report. If cited in a formal thesis, rerun/commit the confusion-pattern analysis script.

**Groups by ConvNeXt fold** (NOTE: these are not the original PanDerm folds; PanDerm predictions were matched on IDs from its own held-out folds):

| ConvNeXt grouping | ConvNeXt macro F1 | Ensemble macro F1 | Delta |
|---|---:|---:|---:|
| 1 | 0.7481 | 0.7963 | +0.0481 |
| 2 | 0.7451 | 0.7766 | +0.0315 |
| 3 | 0.7591 | 0.7552 | −0.0039 |
| 4 | 0.7193 | 0.7049 | −0.0144 |
| 5 | 0.7437 | 0.7726 | +0.0290 |

**Decision:** accept **v17 + PanDerm partial4 50/50 ensemble** as the **leading development ensemble candidate** rather than clinically/generalization-validated final model; 3/5 ConvNeXt-fold groups improved, 2 worsened, so group stability, model cost, calibration and independent generalization remain open. Current standalone ConvNeXt reference remains v17. Historical v15 **0.7444 ± 0.0305** is a *fold mean* and is not automatically a matched pooled comparison with this ensemble. Future formal statistical comparisons should account for repeated lesion images and selection on the development data.

---

# 8. PanDerm branch

## F1. PanDerm Base partial4 — GITHUB-VERIFIED REPORTS — REJECTED AS NEW REFERENCE / PROMISING BRANCH

Model:
- PanDerm Base ViT-B/16;
- dermatology-pretrained checkpoint `panderm_bb_data6_checkpoint-499.pth`;
- 12 Transformer blocks;
- new 7-class head;
- `use_rel_pos_bias=False` because checkpoint did not provide those pretrained tables;
- train last 4/12 blocks + `fc_norm` + head;
- trainable parameters: **28,361,479 / 85,813,255**.

For first controlled candidate screening, inherited v13 pipeline:
- same lesion-safe folds;
- same v13 preprocessing;
- WRS;
- Focal γ=2;
- Adam 1e-4;
- cosine scheduler;
- CutMix p=.5;
- EMA .999;
- batch8;
- 15 epochs max, patience4.

Best fold macro F1 / epoch:
- Fold1: **0.7464**, e8
- Fold2: **0.7469**, e11
- Fold3: **0.7638**, e6
- Fold4: **0.7216**, e5
- Fold5: **0.7052**, e10

Aggregate:
- accuracy: **0.8350 ± 0.0087**
- macro F1: **0.7368 ± 0.0232**

Per-class F1:
- akiec **0.6302 ± 0.0486**
- bcc **0.7550 ± 0.0524**
- bkl **0.7026 ± 0.0191**
- df **0.6577 ± 0.0851**
- mel **0.5983 ± 0.0269**
- nv **0.9158 ± 0.0057**
- vasc **0.8980 ± 0.0564**

Comparison with v13:
- overall macro F1: essentially tied (`0.7368` vs `0.7365`);
- better: `bkl`, especially `vasc`;
- slightly weaker: `akiec`, `bcc`, `df`, `mel`, `nv`.

**Decision:** not a new reference, but the branch remains open because only one PanDerm fine-tuning recipe has been tested.

---

## F2. PanDerm Base partial8 — GITHUB-VERIFIED REPORTS — COMPLETE / REJECTED AS NEW REFERENCE

**Date:** 2026-10-09. **Evidence:** five-fold training logs and independently run evaluation terminal output in AI + English lesson; `panderm_cv_train(2).py`, `panderm_partial8_cv_evaluation.py` and local `reports/panderm_base_partial8_ema_seed123_per_class_cv_summary.json`. **CHAT-VERIFIED** only; do not claim reports are GitHub committed.

**Hypothesis:** unfreezing more of the ViT backbone (last 8 instead of last 4 Transformer blocks) may reduce under-adaptation to HAM10000.

**Controlled change:** last 4/12 -> last 8/12 Transformer blocks trainable; `fc_norm` + 7-way head trainable in both cases. Trainable parameters `56,716,039 / 85,813,255` (versus partial4 `28,361,479 / 85,813,255`). Kept the same 5-fold lesion-safe SGKFold, preprocessing, WRS, Focal γ=2, Adam `1e-4`, cosine scheduler, batch 8, CutMix α=1/p=.5, EMA .999, seed 123, 15-epoch maximum, patience 4, and checkpoint selection by validation macro F1.

| Fold | Best epoch | Partial8 best macro F1 | Partial4 best macro F1 | Difference |
|---|---:|---:|---:|---:|
| 1 | 9 | 0.7754 | 0.7464 | +0.0290 |
| 2 | 9 | 0.7221 | 0.7469 | -0.0248 |
| 3 | 14 | 0.7360 | 0.7638 | -0.0278 |
| 4 | 7 | 0.7441 | 0.7216 | +0.0225 |
| 5 | 4 | 0.7108 | 0.7052 | +0.0056 |

**Macro F1 (mean ± sample SD):** partial8 **0.7377 ± 0.0247** vs partial4 **0.7368 ± 0.0232**; delta in mean approximately **+0.0009**. Partial8 improved 3/5 folds but regressed in 2/5. Overfitting-like loss trends appeared in later epochs, but checkpoints were correctly selected by validation macro F1. Fold 1 took 34.2 min; folds 2–5 collectively 116.3 min on RTX 2060 (approx 150.5 min total, from separately reported runs).

**Independent evaluation:** five fold-matched EMA checkpoints gave macro F1 **0.7377 ± 0.0247**. Fold 5 printed `0.7107` in evaluation rather than `0.7108` in training logs; immaterial at the reported precision, but retain both source values. Per-class summary (mean ± sample SD over folds):

| Class | Precision | Recall | F1 | partial4 F1 | F1 delta (partial8 − partial4) |
|---|---|---|---|---|---|
| akiec | 0.6637 ± 0.0597 | 0.6217 ± 0.0364 | 0.6417 ± 0.0461 | 0.6302 | +0.0115 |
| bcc | 0.7672 ± 0.0989 | 0.7562 ± 0.0299 | 0.7595 ± 0.0573 | 0.7550 | +0.0045 |
| bkl | 0.7065 ± 0.0356 | 0.6818 ± 0.0494 | 0.6920 ± 0.0128 | 0.7026 | −0.0106 |
| df | 0.7756 ± 0.2064 | 0.6023 ± 0.1783 | 0.6595 ± 0.1327 | 0.6577 | +0.0018 |
| mel | 0.5884 ± 0.0502 | 0.6152 ± 0.0701 | 0.5970 ± 0.0215 | 0.5983 | −0.0013 |
| nv | 0.9125 ± 0.0179 | 0.9172 ± 0.0272 | 0.9144 ± 0.0048 | 0.9158 | −0.0014 |
| vasc | 0.8969 ± 0.0825 | 0.9072 ± 0.0432 | 0.8995 ± 0.0396 | 0.8980 | +0.0015 |

**Decision:** effectively tied despite nearly doubling trainable parameters; partial8 is **not** a new reference. Four class means increased, three decreased, but improvements to `mel`, `df`, and `vasc` are absent or negligible. `df` F1 variability is higher than partial4 (`0.1327` vs `0.0851`). Relative to ConvNeXt v13, partial8 is stronger in `vasc` (F1 `0.8995` vs `0.8516`) but weaker in the other six class means. No statistical significance claim is made. **No evidence for deeper fine-tuning. The subsequent partial4 + AdamW comparison also failed to improve the PanDerm reference (see F3).**

---


## F3. PanDerm Base partial4 + AdamW — GITHUB-VERIFIED INDEPENDENT REPORTS — REJECTED AS NEW REFERENCE

**Date:** 2026-10-10. **Evidence:** Fold-1 and Folds-2–5 training logs and subsequent independently run five-EMA-checkpoint evaluation output in AI + English conversation; script `panderm_cv_train(3).py`; local report `reports/panderm_base_partial4_adamw_per_class_cv_summary.json`; best EMA checkpoint files `checkpoints/panderm_base_partial4_adamw_fold{fold}_best.pt`. Metrics from the independent evaluation agree with the rounded best-epoch training results. **Original evidence:** conversation logs. **GitHub verified on 2026-10-14:** `reports/panderm_base_partial4_adamw_per_class_cv_summary.json`, five fold `_evaluation.json`, and five fold `_best_summary.json` are present. The exact AdamW-specific source variant and checkpoint hashes still require pinning for complete replay.

**Hypothesis:** AdamW's decoupled weight decay may generalize better than Adam for PanDerm's partially trainable ViT backbone.

**Controlled change vs F1 / partial4:** `Adam(lr=1e-4)` -> `AdamW(lr=1e-4, weight_decay=1e-2)`. Kept last 4/12 blocks plus `fc_norm` and head trainable (`28,361,479 / 85,813,255` parameters), same lesion-safe 5-fold SGKFold and input preprocessing, WRS, Focal Loss γ=2, cosine scheduler (`T_max=15`, `eta_min=1e-6`), batch 8, CutMix α=1 / p=.5, EMA decay .999, seed 123, max 15 epochs, patience 4, and EMA-validation macro F1 checkpoint selection. Fold 1 was run separately before Folds 2–5.

| Fold | AdamW best epoch | AdamW macro F1 | Adam partial4 macro F1 | Difference |
|---|---:|---:|---:|---:|
| 1 | 9 | 0.7494 | 0.7464 | +0.0030 |
| 2 | 11 | 0.7417 | 0.7469 | −0.0052 |
| 3 | 6 | 0.7680 | 0.7638 | +0.0042 |
| 4 | 5 | 0.7216 | 0.7216 | +0.0000 |
| 5 | 5 | 0.7047 | 0.7052 | −0.0005 |

**Mean ± sample SD macro F1:** AdamW **0.7371 ± 0.0246** vs Adam partial4 **0.7368 ± 0.0232**; mean delta **+0.0003** (calculated on displayed 4-decimal best-fold scores). AdamW improved 2 folds, regressed 2, and tied 1. No inference of statistical significance. In several folds training loss declined while validation loss increased after early epochs; best-checkpoint selection and early stopping worked as intended. Fold 1 took 26.0 minutes; Folds 2–5 collectively took 85.9 minutes, about 111.9 minutes total across runs on RTX 2060.

**Independent evaluation of five selected EMA checkpoints:** mean accuracy **0.8372 ± 0.0057** and macro F1 **0.7371 ± 0.0246** (mean ± sample SD from fold-level output rounded to four decimals). Independently evaluated fold macro F1: `[0.7494, 0.7417, 0.7680, 0.7216, 0.7047]`, matching training log checkpoint selection.

| Class | Precision (mean ± SD) | Recall (mean ± SD) | F1 (mean ± SD) | Adam partial4 F1 | F1 delta (AdamW − Adam) |
|---|---|---|---|---:|---:|
| akiec | 0.6244 ± 0.0650 | 0.6402 ± 0.0490 | 0.6315 ± 0.0530 | 0.6302 | +0.0013 |
| bcc | 0.7644 ± 0.0689 | 0.7648 ± 0.0710 | 0.7626 ± 0.0552 | 0.7550 | +0.0076 |
| bkl | 0.6903 ± 0.0298 | 0.7305 ± 0.0543 | 0.7083 ± 0.0212 | 0.7026 | +0.0057 |
| df | 0.7177 ± 0.1685 | 0.6118 ± 0.1661 | 0.6432 ± 0.1071 | 0.6577 | −0.0145 |
| mel | 0.5950 ± 0.0265 | 0.6153 ± 0.0565 | 0.6030 ± 0.0189 | 0.5983 | +0.0047 |
| nv | 0.9228 ± 0.0092 | 0.9124 ± 0.0081 | 0.9175 ± 0.0026 | 0.9158 | +0.0017 |
| vasc | 0.9138 ± 0.0808 | 0.8787 ± 0.0847 | 0.8935 ± 0.0652 | 0.8980 | −0.0045 |

**Interpretation:** five of seven per-class mean F1 values rose slightly; `df` (−0.0145) and `vasc` (−0.0045) worsened. Melanoma improved only +0.0047. Fold-level variability remains substantial for `df` (F1 SD 0.1071). These deltas are descriptive, not evidence of statistical significance.

**Decision:** no meaningful gain; **do not promote AdamW** as the PanDerm reference. Retain partial4 + Adam. Independent model/per-class evaluation is now complete. **Outstanding evidence task, revised 2026-10-14:** JSON evaluations are now on GitHub; checkpoint weights are intentionally excluded from Git. Pin their hashes/backups and the exact AdamW variant; inspect committed fold confusion matrices if the thesis requires deeper interpretation. 

---
# 9. Research roadmap decision — 2026-10-10 (v17 accepted; next candidates conditional)

**Decision/rationale:** Three controlled PanDerm alternatives have been evaluated (partial4 + Adam, partial8 + Adam, partial4 + AdamW). Partial4 + Adam remains the PanDerm reference because neither change brought meaningful macro-F1 gains. The user chose to **finish a limited sequence of promising ConvNeXt optimizations before evaluating a ConvNeXt + PanDerm ensemble**, rather than performing the ensemble first. PanDerm remains a possible complementary ensemble member, not a new standalone reference. ConvNeXt v17 is now accepted as the image-only development reference; v13 is retained as the historical image-only comparator. v15/v16c retain their separate ensemble/multimodal roles.

Current order (v17 run name confirmed locally; v18/v19 provisional):

1. **ConvNeXt warmup (v17):** COMPLETE AND ACCEPTED as the current image-only development reference after independent five-fold checkpoint evaluation (see D17; 0.7430 ± 0.0144 macro F1, mean +0.0065 against v13).
2. **ConvNeXt layer-wise LR decay (candidate v18, conditional):** test a systematic across-depth learning-rate decay only if this supplies a distinct hypothesis from already-rejected ConvNeXt v5 differential LR (`macro F1 0.7018 ± 0.0282`). Preserve folds/evaluation and change one major factor at a time; reconsider its priority after warmup results.
3. **ConvNeXt checkpoint averaging (candidate v19, conditional):** audit availability of multiple compatible training-trajectory checkpoints before experimenting. Best-only saves may require explicit snapshots/new training. Do **not** conflate averaging checkpoints (weights) with averaging independently trained model predictions (soft voting).
4. **ConvNeXt + PanDerm partial4 ensemble:** once these bounded ConvNeXt options have been assessed, analyze fold-matched complementary errors, evaluate equal-weight soft voting if justified, and benchmark inference cost. Compare both v13 and any newly accepted ConvNeXt model.
5. **Final model selection:** compare with v13 standalone, v15 ensemble (`0.7444 ± 0.0305`), and v16c metadata fusion (`0.7425 ± 0.0349`), including per-class trade-offs, fold stability and inference cost. A `+0.005–0.010` gain in mean macro F1 without important regressions is an *informal engineering target*, **not** a statistical-significance criterion. Stop if additional cost/complexity is unjustified.
6. **Evaluation and application:** arrange genuinely independent external data without overlapping HAM10000 images; verify label ordering and per-checkpoint preprocessing; assess ECE and calibrated probabilities with leakage-safe fitting; proceed to research-prototype application with unsupported/OOD input handling.

**Evidence maintenance, independent of method order:** archive PanDerm partial4/partial8/AdamW five-fold evaluation JSON reports to GitHub; review confusion-matrix evidence only when actually retrieved. **v17 training and independent evaluation are complete and accepted; later candidate methods remain planned.**


## Research-roadmap revision — 2026-10-12 (SUPERSEDES pending steps in 2026-10-10 plan)

User explicitly chose to **continue useful advanced research** rather than freeze the application model solely to avoid complexity; roughly 1.5 years remain. Implementation expense alone is not a reason to reject a method. Keep a strict scientific hypothesis and one-major-factor change where practical. Prior `v18` LLRD is complete/rejected; conditional within-trajectory checkpoint averaging requires saved compatible epoch snapshots (v17 saves only best per fold); v17+PanDerm 50/50 OOF ensemble is complete/promising. Do not label the latter externally validated.

Selected order:
1. **Reproducibility gate before more training:** serialize immutable image_id/lesion_id train/val folds, validate no lesion overlap, lock/export both Python environments / `scikit-learn`, `torch`, `timm` versions, preserve checkpoint and transform metadata. The original `same numeric fold` assumption was invalid across local environments.
2. **Class-imbalance/long-tail experiments:** evaluate **Balanced Softmax** and **LDAM** as controlled candidates against v17; define fair comparisons carefully because current WRS + Focal gamma 2 could overcompensate if mixed naively with new class-aware losses. Avoid claiming an isolated loss effect if sampler/loss both change. Compare five-fold macro F1 and minority performance. Naming `v19` is provisional; earlier conditional checkpoint-averaging plan was not run.
3. **ROI / segmentation branch:** study whether lesion-centered inputs remove distracting background; use suitable segmentation annotations/models, train/validation lesion safety, and check cropping errors.
4. **Supervised contrastive learning:** investigate more separable embeddings for `mel/nv/bkl`, accounting for RTX 2060 6 GB, batch size 8, positives per class and memory costs.
5. **Conditional specialized backbone:** PanDerm is already the domain-pretrained branch; only another backbone if there is a distinctive hypothesis and reasonable data/compute evidence.
6. **Selection/evaluation → product:** fair matched development comparison of standalone/ensembles; independent appropriately disjoint external data with no tuning; calibration/OOD and domain-gap checks; then organized local and GitHub files, reproducible README, and app. Begin cleanup/version-control hygiene as soon as reproducibility is locked, even while research continues.

## Confirmed seven-stage research roadmap — 2026-10-15 (current execution order)

**Decision (user-confirmed):** Preserve the 2026-10-12 expanded-research direction but make the **execution order** explicit. This section supersedes older *pending-step ordering*, not completed measurements, rejected experiments or their source provenance. The early-stage reproducibility audit and existing matched OOF ensemble are evidence milestones, not proof of end-to-end training determinism or independent generalization.

1. **Reproducibility audit — operational milestone reached; broader work remains.** Canonical group-safe folds and original-fold exports are frozen; two historical environments are recorded; 15/15 saved checkpoint replays met macro-F1 tolerance 0.003. This verifies checkpoint-level inference replay, **not** fresh-training replication, clean installation, exact linkage of every recovered historical script to its executed run or external validation. Preserve those open limitations while proceeding with controlled research.
2. **Long-tail methods — NEXT.** First test Balanced Softmax against the completed **natural-sampling ordinary CE** five-fold control (D19), keeping folds, sampling and all other relevant training settings fixed. Preflight one fold and check CutMix/class-count handling before all five folds. Balanced Softmax has **not** been trained as of this decision. Evaluate LDAM **after** reviewing Balanced Softmax, with its own stated control and one-major-factor change where feasible.
3. **ROI / segmentation — PLANNED.** Investigate lesion-centered preprocessing, annotation/mask sources, cropping failure modes and lesion-safe validation under a controlled comparison.
4. **Supervised Contrastive Learning — PLANNED.** Test whether better representations help difficult classes; consider positives per class and the RTX 2060 6-GB / batch-8 compute constraints.
5. **Systematic experiment analysis — PLANNED SYNTHESIS.** Collect comparable evidence from the beginning, but conduct the comprehensive, structured assessment **after Stages 2–4 where feasible**: match split/image/lesion IDs, separate fold-mean and pooled OOF macro F1, compare per-class changes, look for patterns, investigate complementary mistakes, and distinguish association from controlled causal findings.
6. **Final ensemble selection — PLANNED.** Based on the synthesis, select a defensible standalone model or ensemble using matched-held-out samples, stable minority-class performance, error diversity, compute/inference cost and selection-bias safeguards. The pre-existing v17 + PanDerm partial4 50/50 pooled OOF macro F1 **0.7620876** is a **development candidate**, not an independently validated final choice.
7. **Independent evaluation and application — PLANNED.** Evaluate on genuinely independent non-overlapping data as available; assess domain shift, calibration, uncertainty and unsupported/OOD inputs before a **research-only** prototype. Do not claim clinical validity or autonomous diagnosis.

**Research boundaries:** Keep the original ensemble hypothesis and all negative results. No model training or new accuracy/F1 measurements are implied by accepting this roadmap. Existing optional directions (such as another specialized backbone or compatible-checkpoint averaging) are **conditional ideas, not additional mandatory stages**. The next executable action is the separate natural-sampling Balanced Softmax trainer plus a safe one-fold preflight, not the application or the final ensemble.


**2026-10-19 execution update (supersedes only the next-action status above):** The natural-sampling CE control (D19) and the matched five-fold Balanced Softmax experiment (D20) are now complete; Balanced Softmax was **rejected as a CE replacement** after paired checkpoint evaluation. Stage 2 continues with **LDAM as the next planned controlled comparison**. All seven strategic stages and the earlier historical decisions remain unchanged.

**2026-10-21 execution update (supersedes only the pending next-action status above):** LDAM (D21) completed matched five-fold training and 10/10 selected CE/LDAM replay checks; it was **rejected as a CE replacement**. Initial Stage 2 long-tail comparisons are complete; **Stage 3 ROI/segmentation feasibility** is the next research step. The Stage 2 methods are not claimed to exhaust all long-tail approaches, and previously recorded results remain unchanged.

**2026-10-24 execution update (supersedes only the pending Stage 3 status above):** Mask-assisted ROI cropping (D22) completed a controlled canonical five-fold comparison against natural CE. Mean macro F1 increased **0.7386 ± 0.0178 → 0.7512 ± 0.0174**; paired mean difference **+0.01255 ± 0.00888**, six of seven classes improved, five checkpoint replays matched training scores. **Stage 3 remains in progress:** mask quality, predicted-mask reliability, Dice/IoU, end-to-end computation and external evaluation are unresolved. Preserve historical Stage 1/2 outcomes and all prior roadmap decisions.

### Additional research objective: cross-experiment patterns and complementary-error ensembles — PLANNED (2026-10-12)

**User hypothesis (not yet systematically tested):** Comparing the accumulated methods, their evaluation metrics, and failure patterns may expose regularities and lead to an ensemble of models with complementary errors that improves macro F1 over individual classifiers. This is a **research synthesis/analysis project**, not a new architecture or a completed experimental result.

**Execution order:** Start assembling the evidence inventory *during* the reproducibility audit; append new methods as they are completed; perform comprehensive analysis and ensemble selection *after* Balanced Softmax/LDAM, ROI/segmentation, and supervised contrastive branches (as feasible). Continue to preserve independent model-development priority and do not stop useful experiments merely to finish the table.

**Planned artifacts / method:**
1. Unified experiment registry: method/version, architectural family, controlled change, trainable data/folds/environment, best-checkpoint policy, validation protocol era, per-fold and aggregate macro F1, per-class precision/recall/F1 (particularly `mel`), accuracy, seed/fold variation, compute/inference cost, evidence/report pointers and accepted/rejected status. Legacy split versus lesion-safe CV, and **mean of fold F1 versus pooled OOF F1**, remain distinct; do not put incomparable values into a single numerical ranking.
2. Descriptive pattern analysis: trace method changes against comparable baselines (e.g. EMA, CutMix, backbone, loss, sampler, resolution), record improvements **and** regressions; label cross-experiment associations as observational when factors/fold assignments differ, not established causal effects.
3. Error-diversity audit: archive class-ordered OOF probabilities and true labels with image/lesion IDs and the checkpoint/fold that excluded each lesion; confirm unique/matching image IDs and consistent labels. Analyze overlapping vs complementary mistakes, per-class confusion and correction/new-error counts, and fold/lesion-group stability.
4. Ensemble candidates: shortlist models by **quality + diversity + inference cost**, start with prespecified equal-probability soft voting, compare to the strongest standalone model *on the same held-out examples and using the same metric definition*. Do not select weights and claim unbiased performance on the very same OOF predictions; use an appropriately separated selection/evaluation procedure, followed by independent external assessment if suitable data exist. Statistical uncertainty should account for repeated images belonging to the same lesion.

**Success criterion:** reproducible, defensible improvement in matched OOF macro F1 without unacceptable minority-class regression or operational cost, followed by corroboration on independent data; negative findings still contribute to the diploma's scientific conclusions.

**Existing preliminary evidence:** E3 ConvNeXt v17 + PanDerm partial4 50/50 ensemble increased **pooled** OOF macro F1 from **0.744418** (ConvNeXt) to **0.762088**, and improved seven of seven pooled class F1 values. Gains were not uniform across ConvNeXt fold groups (3/5 improved). This motivates the planned analysis but does not establish general success or clinical validity. **Status: planned**; no additional model training or new ensemble measurements are claimed by this entry.

**2026-10-14 archival correction:** the v17/v18 independent evaluation JSONs, PanDerm partial4/partial8/AdamW evaluation JSONs, `reports/repro_audit/*` environment exports, immutable split CSV/JSON, and matched OOF JSON/NPZ artifacts are now tracked. v16c `vasc` F1 is recovered as **0.8612**. Remaining tasks: pin missing exact training-script variants (v18 and PanDerm partial8/AdamW), audit historical `aux02`, validate full source/report integrity, verify environment recreatability and hash local checkpoint backups.

---

# 10. Tested-method inventory

## Imbalance / loss
- plain CrossEntropy
- aggressive class-weighted CrossEntropy
- softer sqrt class weighting
- WeightedRandomSampler
- Focal Loss γ=2
- mel-specific Focal alpha=1.25
- confusion-aware Focal λ=.10
- confusion-aware Focal λ=.05
- Balanced Softmax with natural sampling (D20; five-fold controlled test, rejected vs natural CE)
- LDAM with natural sampling, max margin 0.5 / scale 30 / no DRW (D21; five-fold controlled test, rejected vs natural CE)
- auxiliary-loss (`aux02`) experiment
- hard-example sampling

## Augmentation / input
- targeted `df/mel` augmentation
- mel-only augmentation
- mild global augmentation
- MixUp α=.2
- CutMix α=1, p=.5
- CutMix p=.25
- 288×288 higher resolution

## Fine-tuning / optimizers
- frozen transfer learning
- partial ResNet18 unfreezing
- deeper ResNet18 unfreezing
- discriminative LR
- Adam
- AdamW
- CosineAnnealingLR
- ConvNeXt last-3 → last-5 partial fine-tuning
- differential LR
- full ConvNeXt unfreezing at `1e-4`
- full ConvNeXt unfreezing at `1e-5`
- EMA
- SAM
- ConvNeXt one-epoch LR warmup (v17; independently evaluated and accepted as standalone development reference)
- ConvNeXt layer-wise LR decay (v18; independently evaluated and rejected)
- PanDerm partial4
- PanDerm partial8
- PanDerm partial4 + AdamW (rejected as new reference)

## Architectures / systems
- custom CNN
- ResNet18
- EfficientNet-B0
- MobileNetV3-Large
- ConvNeXt-Tiny
- specialist `bkl/mel/nv`
- specialist cascade
- soft-voting ensemble
- ConvNeXt v17 + PanDerm partial4 matched-ID OOF 50/50 ensemble (leading development candidate)
- metadata fusion
- PanDerm Base ViT-B/16

## Inference
- horizontal-flip TTA
- flip + 90° rotation TTA

---

# 11. Chronology anchors from GitHub commits

- **2026-05-09** — project repository / environment foundation.
- **2026-08-24** — `a09a702...` — “Add ResNet18 fine-tuning and imbalance experiments”.
- **2026-09-05** — `d088f951...` — “Add cross-validation and architecture comparison experiments”.
- **2026-09-11** — `f697bef4...` — “feat(model): add focal CV experiments and specialist cascade”.
- **2026-09-25** — `4a32f87...` — “Update ConvNeXt experiments and GPU AMP pipeline”.
- **2026-10-05** — `c596a8e...` — “Update ConvNeXt experiments (v13 + v14)”.
- **2026-10-07** — PanDerm partial4 exists in Project chat/local files; not yet verified as a GitHub commit in this reconstruction.
- **2026-10-09** — PanDerm partial8 complete: training and per-class independent evaluation terminal output recorded in AI + English chat; GitHub archiving pending.
- **2026-10-10** — PanDerm partial4 + AdamW 5-fold training and independent EMA-checkpoint/per-class evaluation complete; GitHub archiving pending.
- **2026-10-10** — ConvNeXt v17 warmup five-fold training and independent per-class EMA-checkpoint evaluation complete; accepted as new standalone development reference.
- **2026-10-12** — v18 LLRD independent five-fold evaluation complete/rejected; corrected cross-environment OOF v17+PanDerm ensemble yields pooled macro F1 `0.7621`, development-only; package/fold reproducibility issue discovered.
- **2026-10-14** — report archival/index (`reports/README.md`, `reports/MANIFEST.csv`) and source reorganizations across `core/`, `experiments/`, `evaluation/`, `preprocessing/`, `archive/` completed. Natural-sampling CrossEntropy five-fold control summaries recovered. Targeted evidence reconciliation in section 12a; full local bulk audit later completed; see section 12a.
- **2026-10-21** — natural-sampling LDAM five-fold experiment and selected CE/LDAM paired checkpoint evaluation completed locally (10/10 replay checks; rejected as CE replacement, D21). GitHub commit status unverified.
- **2026-10-24** — mask-assisted ROI preprocessing applied to 8,530 development images; five-fold canonical CE comparison completed; mean macro F1 0.7512 ± 0.0174, paired gain +0.01255 ± 0.00888; 5/5 ROI checkpoint replays passed; local results staged for GitHub, push unverified (D22).

---

# 12. Items still to verify before diploma finalization

These are explicit TODOs rather than guessed values:

1. **RESOLVED 2026-10-14:** v16c `vasc` mean F1 = **0.861198** from `reports/convnext_tiny_v16c_metadata_residual_evaluation.json` (five fold values);
2. **RECOVERED CANDIDATES 2026-10-15:** auxiliary-head `aux02` training source snapshots found in VS Code Local History; see §12d. Exact execution/checkpoint attribution remains unproven;
3. if the hard-example experiment is discussed in depth, identify its formally selected best epoch rather than quoting epoch 3 by default;
4. decide whether very early class-weighted-CNN debug runs belong in the main diploma experiment table or only in the development-history appendix;
5. **REPORTS ARCHIVED:** committed five-fold PanDerm partial4/partial8/AdamW evaluation JSONs are present; verify exact variant-specific runnable source and checkpoint backups; review confusion matrices as needed;
6. **REPLAY VERIFIED 2026-10-14:** ConvNeXt v17 five fold-specific checkpoints were reloaded in the main environment on saved original folds; all 5 reproduced validation metrics within 0.003 (replay macro F1 **0.743064 ± 0.014585**). Fold 3 differs by one prediction from the separate evaluation report; retain as precision/traceability TODO;
7. **REPORTS ARCHIVED; FOLD IDENTITY + PARTIAL4 REPLAY VERIFIED:** v18 evaluation JSONs, OOF ensemble JSON/NPZ, immutable ConvNeXt canonical split, and both environment exports are committed. PanDerm partial4 checkpoints reproduced original validation metrics across 5 folds using its own saved fold CSV (see §12c); independently pin/replay v18 and additional PanDerm variants if their inclusion in the diploma requires it. The original ConvNeXt and PanDerm numeric fold assignments must stay separate.
8. **COMPLETED 2026-10-14:** run the local 2,113-report integrity/static-source audit; result: no missing/unreadable artifacts, 33 report review notices (30 expected specialist-protocol alerts + 3 v11 evaluation mismatches), two explicitly failed-reference ensemble reports, 92 Python files with no detected syntax/legacy-import errors. **Open:** follow up three v11 mismatches; runtime checkpoint replay subsequently PASSED for 15/15 selected fold runs (see §12c). Full retraining/environment rebuild remains unverified.

---

# 12a. 2026-10-14 GitHub evidence reconciliation and source-audit boundary

**Verified directly from the reorganized GitHub `main` tree and selected committed report contents (not a bulk parsing of all JSON bodies):**

- At GitHub HEAD `4088b9d`, the `reports/` tree contains **2,115 tracked files**: **2,113 evidence artifacts** plus the index (`reports/README.md`) and manifest (`reports/MANIFEST.csv`). **2,111** files are directly inside `reports/` (including the two index files); four audit exports are nested in `reports/repro_audit/`. GitHub's 1,000-item UI listing cap **does not mean the files are absent**.
- ConvNeXt v17 independent `fold1..fold5_evaluation.json` and `_best_summary.json` are tracked, as are corresponding v18 files.
- PanDerm partial4, partial8 and partial4+AdamW fold evaluation JSONs and available aggregate per-class summaries are tracked; this does **not** establish that each exact training source variant can be regenerated from the current project files.
- `reports/convnext_tiny_v16c_metadata_residual_evaluation.json` establishes five `vasc` F1 values and mean **0.861198**.
- `reports/resnet18_focal_aux02_epoch_{1,2,3}_cv_summary.json` exist, with epoch macro F1 means **0.604857**, **0.611989**, **0.593694** respectively. The precise archived `aux02` training implementation is still **not found as a dedicated tracked `.py` file**; record this as a reproducibility gap, not a validated reconstruction.
- `reports/oof_convnext_panderm_ensemble_results.json` stores matched development OOF macro F1 **0.762088**; `reports/convnext_v17_panderm_partial4_ensemble_summary.json` contains a spurious **0.932298** while its own reference checks failed. Mark the latter **INVALID — DO NOT USE**.
- Exact run-specific source and package/weight provenance remain essential: the Git repository excludes raw images, large checkpoints and local environments by design. Pinned PanDerm source commit and environment exports assist reconstruction, but do not by themselves demonstrate reproducibility of all historical runs.

**Scope update — local audit actually completed:** User ran `audit_research_repository.py` on the local repository at Git HEAD `4088b9ded9e422d2285586b0df1429508facf6fd`, producing `local_audit/repository_audit.json` and `.md` at `2026-10-14T16:50:35Z`. This scan parsed/inspected **all 2,113 Git-tracked report artifacts** (2,106 JSON, 4 NPZ containers, 3 CSV), confirmed **zero missing, malformed or untracked report artifacts**, and found zero discrepancies in `reports/MANIFEST.csv`. It statically parsed **92 tracked Python files**, with **zero syntax, detected missing-project-import, or legacy-import errors**. These are integrity/static checks, **not proof of executable pipelines, valid scientific methodology for every run, identical trained model behavior or exact replay**. NPZ/CSV checking was shallow (ZIP integrity/header only), and the scan did not hash/check raw image data, local checkpoints, all third-party runtime imports, or externally evaluate any checkpoint.

**Audit triage:** the 33 reported metric issues comprise (a) **30 expected notices** (15 specialist reports each flagged for its intended 3-class matrix + class order) and (b) **3 unresolved ConvNeXt v11 CutMix independent-evaluation disagreements**. Separately, **2 files** for the earlier v17 + PanDerm naive ensemble explicitly failed reference checks; results around 0.9323/0.9411 are **invalid**. The proper matched-ID 8,530-image OOF evidence remains `reports/oof_convnext_panderm_ensemble_results.json`, macro F1 **0.762088** on development data, not a held-out external test.

**Provenance/operational issues:** both `experiments/convnext/convnext_tiny_cv_train.py` (editable workspace) and `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` have `RUN_NAME="convnext_tiny_natural_ce_seed123"` in the audited working tree. **Do not launch the workspace until the next experiment has a unique output tag**, to avoid overwriting the archived natural-CE reports/checkpoints. The audit also noted two generic `__file__`-relative path warnings, but their observed definitions (`evaluation/build_reports_index.py`: project-root `parents[1]`; `experiments/panderm/panderm_cv_train.py`: `parents[2]`) are consistent with the reorganized source locations. `archive/README.md` remains factually stale about shared utilities residing in the root: update to `core/`.

**Priority after local audit:** (1) resolve v11 evaluation-vs-best checkpoint disparities or mark unreproducible; (2) fix next-run output naming; (3) pin historical v18, PanDerm partial8/AdamW, and ResNet `aux02` exact source where possible; (4) independently hash/check raw images, checkpoint files, pretrained PanDerm weights, both Python environments and fold exports; (5) rerun canonical split validation and selected old-fold evaluation/reload in the correct environments; (6) keep all invalid reports with visible warning instead of deleting them.

# 12b. 2026-10-14 local reproducibility preflight (snapshot before replay)

**Evidence:** User-provided `local_audit/reproducibility_preflight.json` / `.md`; scan at **2026-10-14T17:14:34Z**, Git HEAD `4088b9ded9e422d2285586b0df1429508facf6fd`. This is **local-file evidence** and is not a successful end-to-end checkpoint validation or an independently repeated training run.

- **Canonical CV verification PASSED:** all 8,530 image IDs match live metadata; 5 lesion-group-safe folds with zero train/validation lesion overlap; canonical export SHA256 `a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248`.
- **Historical fold files present:** ConvNeXt original export has 8,530 unique images and validation fold sizes 1,707/1,706/1,706/1,705/1,706; PanDerm original export has 8,530 unique images and sizes 1,699/1,695/1,713/1,702/1,721; PanDerm fold CSV SHA256 `08ffb6dbf84565c41f098b843c299cf00e9a0d80f4a8e7651ec568e1501b8be6`. The two assignments must **not** be conflated.
- **Local checkpoint presence confirmed, 30/30:** five folds each for ConvNeXt v17 warmup, ConvNeXt natural-sampling CrossEntropy, ConvNeXt v18 LLRD, PanDerm Base partial4 EMA, PanDerm Base partial8 EMA, and PanDerm partial4 AdamW. The script checked presence and byte sizes, **not checkpoint SHA256, model reload or model output**. The existence of a file does not prove that it is the intended saved weights.
- **Data present locally:** 10,015 source images and 10,015 processed images by filesystem count. Content identity and preprocessing correspondence have **not** been cryptographically verified.
- **PanDerm prerequisite present:** external checkout HEAD `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` matches expected pinned commit, PanDerm Python executable exists, and pretrained source checkpoint exists. Python package locks, archive backups, replay equivalence and cross-platform behavior are not yet established.
- **Source script fingerprints recorded:** v17 (`435b54ba7a90728ee6dff4518512956124784e12e9565b59e30a06ca92c9fa1c`); natural CE (`63ca67ec4366245493e5527f42763629a3779f7d4bd9ef61fc3c9a55fe768c0a`); local editable workspace (`c3958caccd44d4c01ea43aa26dcde0b3d2777a19ae84071699d50299b979a37d`); PanDerm partial4 source (`77328f71c7789448464dd55eb8a5c019837b48b7d6c580037e1784f503a7ab01`). These are fingerprints of the current inspected files, not guaranteed original source hashes for missing variants.
- **Operational hazard still open:** the editable `experiments/convnext/convnext_tiny_cv_train.py` and the preserved natural-CE script both specify `RUN_NAME = "convnext_tiny_natural_ce_seed123"`. Do **not** execute the editable workspace until its unique future experiment tag is chosen; do not overwrite the natural-CE artifacts.
- **Unresolved source attribution (preflight snapshot; updated in §12d):** at the time of this preflight no dedicated exact version-pinned training source had been identified for ConvNeXt v18, PanDerm partial8, PanDerm partial4 AdamW or ResNet18 aux02. Preserve the reports and checkpoints; label the source gap rather than inventing reconstructed settings.
- **Documentation:** update `archive/README.md` to describe shared utilities in `core/`, not the repository root.

**Preflight action item (completed for these models in §12c):** in the correct original Python environments, independently reload v17 and PanDerm partial4 fold-specific checkpoint weights; use each model's saved original fold CSV, the recorded preprocessing and the exact class order; calculate accuracy and macro F1 without overwriting original reports; compare with the previously saved fold evaluation. For the natural-CE control use the frozen canonical fold assignment. Collect per-checkpoint hashes, package lockfiles and backed-up data/weight hashes separately. A correct inference replay still does **not** demonstrate deterministic retraining reproducibility or independent clinical validity.

---

# 12c. 2026-10-14 complete five-fold checkpoint replays: 15/15 PASS

**Evidence:** User-supplied `replay_results.zip` with 15 individual inference-replay JSON files (five per model), produced 2026-10-14 on local Git HEAD `4088b9ded9e422d2285586b0df1429508facf6fd`. Consolidated per-fold audit and full checkpoint SHA-256 fingerprints: `Checkpoint_Replay_Verification_2026-10-14.md` and `Checkpoint_Replay_Checkpoint_Manifest_2026-10-14.csv` (replacement artifacts, not yet committed to Git). All replays used the saved model-specific image/lesion fold membership, original preprocessing transform, seven-class order, strict state-dict checkpoint reload, and CUDA inference **without retraining** or altering original reports. The configured PASS threshold for macro F1 and accuracy was absolute difference at most **0.003**, together with matching checkpoint-selected epoch.

| Model / model selection | Five-fold macro F1 from replay (sample SD) | All folds | Saved-reference comparison |
|---|---:|---|---|
| ConvNeXt-Tiny v17 warmup, EMA | **0.743064 ± 0.014585** | **PASS 5/5** | Five independent validation evaluation JSON files; maximum abs F1 delta **0.000515962**, maximum abs accuracy delta **0.000586166** |
| PanDerm Base partial4 EMA | **0.736796 ± 0.023202** | **PASS 5/5** | Five independent validation evaluation JSON files; maximum abs F1 delta **0.000011534**, accuracy deltas **0** |
| ConvNeXt natural sampling + Cross-Entropy | **0.738647 ± 0.017843** | **PASS 5/5** | Five saved best-checkpoint validation summaries; F1 and accuracy deltas **0** across all five |

**Checkpoint identities:** 15 unique SHA-256 digests recorded (five per run), with each checkpoint epoch identical to its best-summary epoch. The saved original ConvNeXt fold export SHA256 is `a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248` (also used as the natural-CE canonical export); saved PanDerm fold export SHA256 is `08ffb6dbf84565c41f098b843c299cf00e9a0d80f4a8e7651ec568e1501b8be6`. Historical PanDerm folds are **not** the same numeric membership as ConvNeXt folds.

**Residual discrepancy — v17 Fold 3:** independent evaluation reported macro F1 **0.758614285714**, while the replay obtained **0.759130247614**, identical to the saved training best-summary F1. Replay accuracy was higher by **0.0005861665 = 1/1706 image**; macro F1 differed by **+0.0005159619**. This is within the predetermined tolerance, not an exact prediction match. The root cause has **not** been determined and requires image-ID-level prediction diff and numeric/preprocessing checks if exact equality is required. The remaining v17 folds reproduced independent accuracy exactly, and had smaller F1 differences.

**Environment evidence:** ConvNeXt replays used Python 3.13.7 / PyTorch 2.12.0+cu126 / torchvision 0.27.0+cu126 / sklearn 1.9.0; PanDerm used Python 3.10.11 / PyTorch 2.4.1+cu118 / torchvision 0.19.1+cu118 / sklearn 1.7.2. Fold sizes total 8,530 validation instances per run (every original development image tested exactly once in each run). This is **selected-checkpoint development cross-validation**, not held-out independent external validation.

**Scientific decision:** checkpoint-level **inference reproducibility within 0.003** is established for the 15 specified fold checkpoints. Preserve existing model-acceptance decisions: ConvNeXt v17 remains the image-only development reference; natural CE is a completed comparison/control; PanDerm partial4 is the secondary complementary model. Do **not** reinterpret 0.762088 matched pooled OOF ensemble F1 as a five-fold mean or claim clinical-level generalizability. Invalid ~0.9323 ensemble reports remain flagged as **INVALID**.

**Still open:** (1) full package lock + clean rebuild, independent backups and byte-level hashes for original images and processed data; (2) historical training-source recovery (later addressed with candidate snapshots in §12d, not authenticated to exact checkpoints); (3) three unresolved v11 CutMix best-summary-vs-evaluation discrepancies; (4) updating stale `archive/README.md`; (5) resolving duplicate natural-CE `RUN_NAME` in the editable workspace before new training; (6) training reproducibility and any independent external evaluation. The 30/30 locally *present* checkpoints from preflight include 15 other checkpoints not replayed or hashed by this batch; do not claim all 30 verified by inference.

---


# 12d. 2026-10-15 recovery of historical training source snapshots

**Evidence:** user-supplied `recovered_source_candidates.zip` generated from VS Code
Local History, with SHA256 manifest. Verified 22/22 distinct candidate source-byte hashes,
and all 22 scripts parse as valid Python. These are real historical **editor
snapshots**, not proven exact sources of the saved trained checkpoints.

| Experiment | Selected original Fold 1 / Folds 2–5 training snapshots | Recovered configuration |
|---|---|---|
| ConvNeXt v18 LLRD | `18sU.py` / `87P5.py` | LLRD decay 0.8, 5 late feature modules, per-group Adam, warmup, cosine, EMA, focal loss |
| PanDerm Base partial8 EMA | `RqQr.py` / `uWQC.py` | 8 trainable blocks, Adam, weighted sampling, EMA, focal loss |
| PanDerm Base partial4 AdamW | `H6pT.py` / `ZmWB.py` | 4 trainable blocks, AdamW lr 1e-4 / weight decay 1e-2, weighted sampling, EMA, focal loss |
| ResNet18 aux02 | `vVeV.py` / `Q4NT.py` | 7-class main head and 3-class {bkl,mel,nv} auxiliary head, loss weight 0.2, Adam, 3 epochs |

The original snippets have legacy root-level imports and original path
assumptions. They have been **preserved byte-for-byte**, under
`archive/recovered_sources/snapshots/` with a `.py.txt` suffix (no runnable
training scripts replaced). See `archive/recovered_sources/README.md` and
`provenance.json` for timestamps, SHA256 hashes and rejected transitional
snapshots. Source identity **remains a strong historical attribution, not a
checkpoint-cryptographically-proven training source or a successful retraining
replay**.

**Warning:** PanDerm intermediate `DZZV.py` uses only four trainable blocks while
writing *partial8* tags. `8DSI.py` switches to AdamW with the same stale tags;
`ofRo.py` has eight blocks but still writes *partial4* tags. These revisions
remain in the archive as evidence and must **not** be presented as properly
configured complete experiment implementations.

**Source gap revised:** historical copies exist for all four previously
untracked variants. The remaining blockers concern attribution to the saved
checkpoints, compatibility/runtime replay, data and environment checksums, and
independently reproduced training; this recovery does not resolve them.

---

# 13. Diploma-writing rule

For the final thesis:

- do **not** put every debug run in the main results table;
- preserve them here permanently;
- main thesis tables should emphasize controlled, scientifically comparable experiments under the lesion-safe 5-fold protocol;
- use Era A only to explain how the methodology matured (imbalance, leakage awareness, transfer learning);
- use Era B for formal model comparison and conclusions.

# Research Report Index

Automatically indexed **2,113 Git-tracked report artifacts** across **48 experiment/report groups**.

**The files are still stored at their original paths.** This index does not rename, move, or delete them. The [CSV manifest](MANIFEST.csv) lists every file and can be filtered by experiment, fold, or report type.

Important: reported metrics use different validation protocols. Do not compare fold means with pooled OOF or independent-test results as if they were identical.

**Known invalid historical report:** [`convnext_v17_panderm_partial4_ensemble_summary.json`](convnext_v17_panderm_partial4_ensemble_summary.json) has `all_reference_checks_passed: false` and must **not** be cited as a valid ensemble result. The separately verified pooled OOF comparison is in [`oof_convnext_panderm_ensemble_results.json`](oof_convnext_panderm_ensemble_results.json).

## Browse by model family

- [ConvNeXt](#convnext) — 25 groups
- [PanDerm](#panderm) — 3 groups
- [Ensembles and OOF](#ensembles-and-oof) — 2 groups
- [ResNet](#resnet) — 11 groups
- [EfficientNet](#efficientnet) — 2 groups
- [MobileNet](#mobilenet) — 1 groups
- [Other](#other) — 4 groups

## ConvNeXt

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `convnext_tiny` | 43 | 1, 2, 3, 4, 5 | — | [CV summary](convnext_tiny_epoch_1_cv_summary.json) · [CV summary](convnext_tiny_epoch_1_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v1` | 66 | 1, 2, 3, 4, 5 | 0.6880 ± 0.0273 | [CV summary](convnext_tiny_finetune_v1_cv_summary.json) · [CV summary](convnext_tiny_finetune_v1_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v10` | 72 | 1, 2, 3, 4, 5 | 0.6993 ± 0.0236 | [CV summary](convnext_tiny_finetune_v10_full_unfreeze_cv_summary.json) · [CV summary](convnext_tiny_finetune_v10_full_unfreeze_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v11_cutmix` | 69 | 1, 2, 3, 4, 5 | 0.7176 ± 0.0244 | [CV summary](convnext_tiny_finetune_v11_cutmix_cv_summary.json) · [CV summary](convnext_tiny_finetune_v11_cutmix_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v11_seed123_cutmix` | 77 | 1, 2, 3, 4, 5 | 0.7191 ± 0.0304 | [convnext_tiny_finetune_v11_seed123_cutmix_fold1_best_summary.json](convnext_tiny_finetune_v11_seed123_cutmix_fold1_best_summary.json) · [convnext_tiny_finetune_v11_seed123_cutmix_fold1_training_history.json](convnext_tiny_finetune_v11_seed123_cutmix_fold1_training_history.json) |
| `convnext_tiny_finetune_v11_seed123_cutmix_tta_flip` | 5 | 1, 2, 3, 4, 5 | — | [convnext_tiny_finetune_v11_seed123_cutmix_tta_flip_fold1_evaluation.json](convnext_tiny_finetune_v11_seed123_cutmix_tta_flip_fold1_evaluation.json) · [convnext_tiny_finetune_v11_seed123_cutmix_tta_flip_fold2_evaluation.json](convnext_tiny_finetune_v11_seed123_cutmix_tta_flip_fold2_evaluation.json) |
| `convnext_tiny_finetune_v12_cutmix_p025` | 66 | 1, 2, 3, 4, 5 | 0.7086 ± 0.0320 | [CV summary](convnext_tiny_finetune_v12_cutmix_p025_cv_summary.json) · [CV summary](convnext_tiny_finetune_v12_cutmix_p025_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v13_ema_seed123` | 59 | 1, 2, 3, 4, 5 | 0.7364 ± 0.0343 | [convnext_tiny_finetune_v13_ema_seed123_fold1_best_summary.json](convnext_tiny_finetune_v13_ema_seed123_fold1_best_summary.json) · [convnext_tiny_finetune_v13_ema_seed123_fold1_training_history.json](convnext_tiny_finetune_v13_ema_seed123_fold1_training_history.json) |
| `convnext_tiny_finetune_v14+v13` | 5 | 1, 2, 3, 4, 5 | — | [convnext_tiny_finetune_v14+v13_fold1_evaluation.json](convnext_tiny_finetune_v14%2Bv13_fold1_evaluation.json) · [convnext_tiny_finetune_v14+v13_fold2_evaluation.json](convnext_tiny_finetune_v14%2Bv13_fold2_evaluation.json) |
| `convnext_tiny_finetune_v14_sam_ema_seed123` | 71 | 1, 2, 3, 4, 5 | 0.7370 ± 0.0310 | [convnext_tiny_finetune_v14_sam_ema_seed123_fold1_best_summary.json](convnext_tiny_finetune_v14_sam_ema_seed123_fold1_best_summary.json) · [convnext_tiny_finetune_v14_sam_ema_seed123_fold1_training_history.json](convnext_tiny_finetune_v14_sam_ema_seed123_fold1_training_history.json) |
| `convnext_tiny_finetune_v15` | 1 | 1 | — | [convnext_tiny_finetune_v15_fold1_evaluation.json](convnext_tiny_finetune_v15_fold1_evaluation.json) |
| `convnext_tiny_finetune_v17_warmup_seed123` | 65 | 1, 2, 3, 4, 5 | 0.7431 ± 0.0146 | [convnext_tiny_finetune_v17_warmup_seed123_fold1_best_summary.json](convnext_tiny_finetune_v17_warmup_seed123_fold1_best_summary.json) · [convnext_tiny_finetune_v17_warmup_seed123_fold1_training_history.json](convnext_tiny_finetune_v17_warmup_seed123_fold1_training_history.json) |
| `convnext_tiny_finetune_v18_llrd_seed123` | 63 | 1, 2, 3, 4, 5 | 0.7361 ± 0.0105 | [convnext_tiny_finetune_v18_llrd_seed123_fold1_best_summary.json](convnext_tiny_finetune_v18_llrd_seed123_fold1_best_summary.json) · [convnext_tiny_finetune_v18_llrd_seed123_fold1_training_history.json](convnext_tiny_finetune_v18_llrd_seed123_fold1_training_history.json) |
| `convnext_tiny_finetune_v2` | 64 | 1, 2, 3, 4, 5 | 0.6876 ± 0.0376 | [CV summary](convnext_tiny_finetune_v2_cv_summary.json) · [CV summary](convnext_tiny_finetune_v2_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v3` | 58 | 1, 2, 3, 4, 5 | 0.6998 ± 0.0306 | [CV summary](convnext_tiny_finetune_v3_cv_summary.json) · [CV summary](convnext_tiny_finetune_v3_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v4` | 71 | 1, 2, 3, 4, 5 | 0.7093 ± 0.0099 | [CV summary](convnext_tiny_finetune_v4_cv_summary.json) · [CV summary](convnext_tiny_finetune_v4_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v4_gpu_amp` | 73 | 1, 2, 3, 4, 5 | 0.7115 ± 0.0108 | [CV summary](convnext_tiny_finetune_v4_gpu_amp_cv_summary.json) · [CV summary](convnext_tiny_finetune_v4_gpu_amp_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v5` | 58 | 1, 2, 3, 4, 5 | 0.7018 ± 0.0283 | [CV summary](convnext_tiny_finetune_v5_cv_summary.json) · [CV summary](convnext_tiny_finetune_v5_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v6_global_aug` | 64 | 1, 2, 3, 4, 5 | 0.7088 ± 0.0301 | [CV summary](convnext_tiny_finetune_v6_global_aug_cv_summary.json) · [CV summary](convnext_tiny_finetune_v6_global_aug_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v7_highres_288` | 68 | 1, 2, 3, 4, 5 | 0.6953 ± 0.0225 | [CV summary](convnext_tiny_finetune_v7_highres_288_cv_summary.json) · [CV summary](convnext_tiny_finetune_v7_highres_288_per_class_cv_summary.json) |
| `convnext_tiny_finetune_v8_mixup` | 58 | 1, 2, 3, 4, 5 | 0.7012 ± 0.0269 | [convnext_tiny_finetune_v8_mixup_fold1_best_summary.json](convnext_tiny_finetune_v8_mixup_fold1_best_summary.json) · [convnext_tiny_finetune_v8_mixup_fold1_training_history.json](convnext_tiny_finetune_v8_mixup_fold1_training_history.json) |
| `convnext_tiny_finetune_v9_full_unfreeze` | 70 | 1, 2, 3, 4, 5 | 0.7049 ± 0.0197 | [CV summary](convnext_tiny_finetune_v9_full_unfreeze_cv_summary.json) · [CV summary](convnext_tiny_finetune_v9_full_unfreeze_per_class_cv_summary.json) |
| `convnext_tiny_natural_ce_seed123` | 75 | 1, 2, 3, 4, 5 | 0.7386 ± 0.0178 | [convnext_tiny_natural_ce_seed123_fold1_best_summary.json](convnext_tiny_natural_ce_seed123_fold1_best_summary.json) · [convnext_tiny_natural_ce_seed123_fold1_training_history.json](convnext_tiny_natural_ce_seed123_fold1_training_history.json) |
| `convnext_tiny_v4_tta_flip` | 7 | 1, 2, 3, 4, 5 | — | [CV summary](convnext_tiny_v4_tta_flip_per_class_cv_summary.json) · [convnext_tiny_v4_tta_flip_summary.json](convnext_tiny_v4_tta_flip_summary.json) |
| `convnext_tiny_v4_tta_flip_rot90` | 5 | 1, 2, 3, 4, 5 | — | [convnext_tiny_v4_tta_flip_rot90_fold1_evaluation.json](convnext_tiny_v4_tta_flip_rot90_fold1_evaluation.json) · [convnext_tiny_v4_tta_flip_rot90_fold2_evaluation.json](convnext_tiny_v4_tta_flip_rot90_fold2_evaluation.json) |

## PanDerm

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `panderm_base_partial4_adamw` | 72 | 1, 2, 3, 4, 5 | 0.7371 ± 0.0246 | [CV summary](panderm_base_partial4_adamw_per_class_cv_summary.json) · [panderm_base_partial4_adamw_fold1_best_summary.json](panderm_base_partial4_adamw_fold1_best_summary.json) |
| `panderm_base_partial4_ema_seed123` | 75 | 1, 2, 3, 4, 5 | 0.7369 ± 0.0232 | [panderm_base_partial4_ema_seed123_fold1_best_summary.json](panderm_base_partial4_ema_seed123_fold1_best_summary.json) · [panderm_base_partial4_ema_seed123_fold1_training_history.json](panderm_base_partial4_ema_seed123_fold1_training_history.json) |
| `panderm_base_partial8_ema_seed123` | 76 | 1, 2, 3, 4, 5 | 0.7377 ± 0.0247 | [CV summary](panderm_base_partial8_ema_seed123_per_class_cv_summary.json) · [panderm_base_partial8_ema_seed123_fold1_best_summary.json](panderm_base_partial8_ema_seed123_fold1_best_summary.json) |

## Ensembles and OOF

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `convnext_v17_panderm_partial4_ensemble` | 3 | 1 | — | [convnext_v17_panderm_partial4_ensemble_summary.json](convnext_v17_panderm_partial4_ensemble_summary.json) · [convnext_v17_panderm_partial4_ensemble_fold1.json](convnext_v17_panderm_partial4_ensemble_fold1.json) |
| `oof_convnext_panderm_ensemble` | 1 | — | — | [oof_convnext_panderm_ensemble_results.json](oof_convnext_panderm_ensemble_results.json) |

## ResNet

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `resnet18` | 68 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_epoch_1_cv_summary.json) · [CV summary](resnet18_epoch_2_cv_summary.json) |
| `resnet18_focal` | 41 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_epoch_2_cv_summary.json) |
| `resnet18_focal_aux02` | 41 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_aux02_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_aux02_epoch_1_per_class_cv_summary.json) |
| `resnet18_focal_hard15` | 41 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_hard15_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_hard15_epoch_1_per_class_cv_summary.json) |
| `resnet18_focal_mel_alpha125` | 39 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_mel_alpha125_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_mel_alpha125_epoch_2_cv_summary.json) |
| `resnet18_focal_mel_aug` | 40 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_mel_aug_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_mel_aug_epoch_2_cv_summary.json) |
| `resnet18_focal_mel_confusion` | 39 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_mel_confusion_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_mel_confusion_epoch_2_cv_summary.json) |
| `resnet18_focal_mel_confusion(0.05)` | 39 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_mel_confusion%280.05%29_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_mel_confusion%280.05%29_epoch_2_cv_summary.json) |
| `resnet18_focal_reference_recheck` | 6 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_reference_recheck_per_class_cv_summary.json) · [resnet18_focal_reference_recheck_fold1_epoch_3_evaluation.json](resnet18_focal_reference_recheck_fold1_epoch_3_evaluation.json) |
| `resnet18_focal_specialist` | 39 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_specialist_epoch_1_cv_summary.json) · [CV summary](resnet18_focal_specialist_epoch_2_cv_summary.json) |
| `resnet18_focal_specialist_cascade` | 7 | 1, 2, 3, 4, 5 | — | [CV summary](resnet18_focal_specialist_cascade_cv_summary.json) · [CV summary](resnet18_focal_specialist_cascade_per_class_cv_summary.json) |

## EfficientNet

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `efficientnet_b0` | 26 | 1, 2, 3, 4, 5 | — | [efficientnet_b0_fold1_training_history.json](efficientnet_b0_fold1_training_history.json) · [efficientnet_b0_fold1_epoch_1_report.json](efficientnet_b0_fold1_epoch_1_report.json) |
| `efficientnet_b0_new` | 41 | 1, 2, 3, 4, 5 | — | [CV summary](efficientnet_b0_new_epoch_1_cv_summary.json) · [CV summary](efficientnet_b0_new_epoch_1_per_class_cv_summary.json) |

## MobileNet

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `mobilenet_v3_large` | 41 | 1, 2, 3, 4, 5 | — | [CV summary](mobilenet_v3_large_epoch_1_cv_summary.json) · [CV summary](mobilenet_v3_large_epoch_1_per_class_cv_summary.json) |

## Other

| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |
|---|---:|---:|---:|---|
| `augmented_sampler` | 1 | — | — | [augmented_sampler_training_history.json](augmented_sampler_training_history.json) |
| `dataset` | 1 | — | — | [dataset_summary.json](dataset_summary.json) |
| `Other reports` | 42 | — | — | [oof_convnext_original_folds_report.json](oof_convnext_original_folds_report.json) · [oof_panderm_original_folds_report.json](oof_panderm_original_folds_report.json) |
| `weighted` | 1 | — | — | [weighted_training_history.json](weighted_training_history.json) |

## How to keep this index current

From the project root, run:

```powershell
.\venv\Scripts\python.exe -m evaluation.build_reports_index
```

Commit the regenerated `reports/README.md` and `reports/MANIFEST.csv` alongside newly completed reports. The index intentionally contains **only Git-tracked artifacts**; stage any new report files first, then regenerate.

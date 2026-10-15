# HAM10000 — research evidence audit, second pass

**Date:** 2026-10-14. **Source:** local `repository_audit.json` and `repository_audit.md`, Git head `4088b9ded9e422d2285586b0df1429508facf6fd`, plus targeted review of specialist source and selected reported JSONs. These audits are read-only and do not re-run model training.

## Confirmed integrity findings

- 2,113 Git-tracked research report artifacts: 2,106 valid JSON, 4 well-formed NPZ ZIP containers, 3 readable CSV headers. No missing/untracked report artifacts, no JSON parsing failures, no index-manifest inconsistencies in the checked repository. **NPZ arrays and CSV data rows were not fully semantically validated by the first script.**
- 92 tracked Python files parsed successfully; the static detector found no missing *project* imports, obsolete old-name imports, or syntax errors. This **does not prove that all scripts execute correctly**, that required packages are installed, or that checkpoints load.
- Every reported best-checkpoint five-fold run in the audit inventory has a complete five-fold summary. This statement does not apply to experimental trials saved under different filename schemes or single-fold pilots. A five-fold mean is **not** equivalent to pooled development OOF.

## Classification of the 33 audit issues

| Category | Notices | Action |
|---|---:|---|
| Restricted three-class ResNet18 specialist | 30 (15 reports × two notices each) | Expected: each has 3×3 confusion matrix, class order `bkl, mel, nv`. These reports describe a three-class model and must not be compared as a seven-class classifier. |
| ConvNeXt v11 CutMix best checkpoint versus evaluator | 3 | Real unresolved discrepancies. Re-evaluate checkpoint/source/transform/IDs before claiming exact v11 reproduction. |

ConvNeXt v11 fold discrepancies (`best_macro_f1` in best-summary versus `macro_f1` in separate evaluation):

| Fold | Best summary | Evaluation | Evaluation − best |
|---|---:|---:|---:|
| 1 | 0.751912 | 0.736414 | −0.015498 |
| 2 | 0.720694 | 0.704486 | −0.016209 |
| 5 | 0.692059 | 0.711129 | +0.019070 |

The saved fold 1 epoch 6 JSON agrees with the `best_summary.json`; this is a genuine **across-artifact disagreement**, not a simple wrong epoch in the summary. No root cause is established from the audit; avoid speculative fixes or deleting either source. Folds 3–4 were not flagged at the script's >0.015 difference threshold, not proof of exact agreement.

## Invalid predecessor results

Two files explicitly report reference-check failure:

- `reports/convnext_v17_panderm_partial4_ensemble_fold1.json`
- `reports/convnext_v17_panderm_partial4_ensemble_summary.json`

The earlier ensemble F1 ≈ 0.9323 and PanDerm F1 ≈ 0.9411 were obtained from a mismatched evaluation environment/fold assignment. Do **not** include these values in model rankings or diploma tables. Preserve as a documented failure. The valid aligned 8,530-image **development OOF** result is `reports/oof_convnext_panderm_ensemble_results.json`, ensemble pooled macro F1 **0.762088**, compared with ConvNeXt pooled **0.744418**. It is neither clinical nor disjoint external validation.

## Script/provenance risks

1. The editable workspace `experiments/convnext/convnext_tiny_cv_train.py` and completed natural-CE source `experiments/convnext/convnext_tiny_cv_train_natural_ce.py` share `RUN_NAME="convnext_tiny_natural_ce_seed123"`. Rename the active workspace tag before *any* new training, or it can overwrite reports/checkpoints of the completed control.
2. Dedicated, frozen training scripts for v18 LLRD, PanDerm partial8/AdamW and ResNet `aux02` are not identifiable as versioned tracked files. Reports alone cannot reconstruct their exact training logic. Seek original snapshots or label historical methods partially reproducible.
3. `evaluation/build_reports_index.py` derives repo root with `Path(__file__).resolve().parents[1]`; `experiments/panderm/panderm_cv_train.py` uses `.parents[2]`. These **appear appropriate** for the reorganized tree. The prior path-risk notices were heuristic; full PanDerm runtime/source import still needs checking in its own environment.
4. `archive/README.md` refers to shared utilities in repository root, which is stale (they are now in `core/`). Update the description and caution that archived runs require old checkpoints and fold IDs.
5. Large checkpoints, image data and local Python environments are intentionally ignored by Git; the audit did not inventory/checksum these external artifacts. Historical ConvNeXt and PanDerm numeric folds differ across environments; original image/lesion held-out IDs must be matched to the respective checkpoints.
6. The project does not yet have a demonstrated **full deterministic reproduction** of the accepted ConvNeXt v17 and PanDerm ensemble results, nor independent disjoint clinical/generalization testing. Static validation is useful evidence integrity, not that stronger claim.

## Recommended closeout sequence

1. Prevent the workspace `RUN_NAME` collision (before any new experiment).
2. Run the read-only `evaluation/reproducibility_preflight.py` with the user's main local environment. Inspect discovered checkpoint and source weights availability, PanDerm source commit, both historical fold exports and canonical CV verification.
3. Record checkpoint SHA256, external pretrained-weight SHA256, immutable splits, dependency lockfiles (`pip freeze` from both environments), exact PyTorch/CUDA and hardware, source Git SHA, and deterministic RNG seeds.
4. In each original environment, reload one representative accepted checkpoint and verify its held-out image/lesion set, class mapping, transforms and F1 against the saved fold evaluation. Do not rerun or overwrite training report filenames to do so.
5. For unexplained v11 discrepancies, either recover the original exact evaluator, checkpoint and fold membership and resolve differences or explicitly label independent v11 reproducibility as unresolved.
6. Preserve all historical evidence. Only after provenance checks decide whether to resume controlled Balanced Softmax training; a new run must have a unique name and use canonical folds.

**Conclusion:** The report archive passes basic completeness and static integrity checks. The project is **not yet established as fully reproducible end to end**; the gaps are specific and actionable rather than evidence that all 2,113 reports are unreliable.

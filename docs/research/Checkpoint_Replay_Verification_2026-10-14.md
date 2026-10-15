# Checkpoint replay verification — 2026-10-14

**Result: PASS 15/15** for inference-only replay using the original saved lesion-safe fold assignments. All 15 checkpoints loaded, passed strict state-dict loading, matched their saved best epoch, and met the script tolerance of **0.003** for both macro F1 and accuracy. All 15 checkpoints have recorded **distinct SHA-256 fingerprints**. This does not establish exact retraining reproducibility or independent clinical validity.

- Source evidence: user-supplied `replay_results.zip` (15 JSON files), ZIP SHA-256: `2c7f47e448fbde0c084f4af1c629ebf5524ce71d3c8123f3a5973a2509405489`.
- Git commit reported in all replays: `4088b9ded9e422d2285586b0df1429508facf6fd`.
- Procedure: CUDA inference, `ConvNeXt_Tiny_Weights.DEFAULT.transforms()`, seven-class order `akiec, bcc, bkl, df, mel, nv, vasc`; `torch.load(..., weights_only=True)` and `load_state_dict(..., strict=True)`; original image-to-fold exports checked for lesion leakage, evaluated **without retraining or modifying original reports**.
- Accuracy and macro F1 were compared with the saved **independent evaluation** for v17 and PanDerm, and the saved **best-checkpoint validation summary** for natural CE. **Both kinds of reference are development validation, not an external test.**

## Five-fold results

| Run | Fold CV macro F1 from replay, mean ± sample SD | Max absolute F1 difference from saved reference | Max absolute accuracy difference | Verdict |
|---|---:|---:|---:|---|
| ConvNeXt-Tiny v17 (warmup) | **0.743064 ± 0.014585** | 0.000515962 | 0.000586166 | PASS 5/5 |
| PanDerm Base partial4 EMA | **0.736796 ± 0.023202** | 0.000011534 | 0.000000000 | PASS 5/5 |
| ConvNeXt-Tiny natural sampling + Cross-Entropy | **0.738647 ± 0.017843** | 0.000000000 | 0.000000000 | PASS 5/5 |

**Interpretation:** v17's replay aggregate `0.743064 ± 0.014585` is calculated from current replay predictions. The stored independent-evaluation aggregate is approximately `0.742997 ± 0.014374`. For PanDerm, the replay aggregate is `0.736796 ± 0.023202` and the stored evaluation aggregate is approximately `0.736797 ± 0.023206`. Natural CE reproduces its saved fold-best scores exactly: `0.738647 ± 0.017843`. These are **five-fold averages**, not pooled OOF metrics; historical ConvNeXt and PanDerm used different fold assignments and should not be treated as a fold-paired comparison.

## Per-fold replay evidence

| Run | Fold | Val images | Replayed macro F1 | Reference macro F1 | Δ F1 | Δ accuracy | Epoch | SHA-256 prefix |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| convnext_v17 | 1 | 1,707 | 0.748126681 | 0.748128571 | -0.000001890 | +0.000000000 | 3 | `953944ac07ba2980` |
| convnext_v17 | 2 | 1,706 | 0.745062258 | 0.745071429 | -0.000009171 | +0.000000000 | 4 | `c425de7578503f56` |
| convnext_v17 | 3 | 1,706 | 0.759130248 | 0.758614286 | +0.000515962 | +0.000586166 | 8 | `d5cfbb8a8988284a` |
| convnext_v17 | 4 | 1,705 | 0.719340189 | 0.719514286 | -0.000174096 | +0.000000000 | 5 | `a59d4081b1c32def` |
| convnext_v17 | 5 | 1,706 | 0.743661098 | 0.743657143 | +0.000003955 | +0.000000000 | 10 | `99dc121fabac5771` |
| panderm_partial4 | 1 | 1,699 | 0.746388466 | 0.746400000 | -0.000011534 | +0.000000000 | 8 | `873fda0670825a92` |
| panderm_partial4 | 2 | 1,695 | 0.746936040 | 0.746942857 | -0.000006817 | +0.000000000 | 11 | `9e150c88e8f7f9b2` |
| panderm_partial4 | 3 | 1,713 | 0.763817567 | 0.763814286 | +0.000003282 | +0.000000000 | 6 | `1ff874aa10f71e1a` |
| panderm_partial4 | 4 | 1,702 | 0.721646969 | 0.721642857 | +0.000004112 | +0.000000000 | 5 | `58d512f9de09ca25` |
| panderm_partial4 | 5 | 1,721 | 0.705193067 | 0.705185714 | +0.000007353 | +0.000000000 | 10 | `dfec53e0d40f1376` |
| convnext_natural_ce | 1 | 1,707 | 0.761805560 | 0.761805560 | +0.000000000 | +0.000000000 | 7 | `c68aaa48ea204428` |
| convnext_natural_ce | 2 | 1,706 | 0.734747427 | 0.734747427 | +0.000000000 | +0.000000000 | 11 | `cbb0d2dfc5179c77` |
| convnext_natural_ce | 3 | 1,706 | 0.746954636 | 0.746954636 | +0.000000000 | +0.000000000 | 12 | `0c050061f04314cb` |
| convnext_natural_ce | 4 | 1,705 | 0.713179473 | 0.713179473 | +0.000000000 | +0.000000000 | 9 | `9a7911b74f690f89` |
| convnext_natural_ce | 5 | 1,706 | 0.736545607 | 0.736545607 | +0.000000000 | +0.000000000 | 7 | `eeb2f63605476a28` |

### Residual difference to keep visible

**ConvNeXt v17, Fold 3:** replay accuracy is greater than the independent evaluation by `0.0005861665` (= **1 / 1,706 validation images**); replay macro F1 is greater by `0.0005159619`. Both remain within the predeclared `0.003` tolerance. The replay's F1 `0.759130247614` is the same value recorded in Fold 3's training `best_summary` but differs from the separate independent evaluation `0.758614285714`. Without image-ID-level prediction comparison, the reason (e.g., precision, model/data differences, or numerical instability) is **not established**. Do not describe all replays as *bit-for-bit identical*.

All other v17 folds have smaller than 0.000175 F1 differences and zero accuracy difference. PanDerm partial4 has at most 0.000012 F1 discrepancy and zero accuracy differences. Natural CE has zero differences in F1 and accuracy for all five folds.

## Frozen identities and runtime

| Run | Original saved-fold CSV SHA-256 | Python | PyTorch / CUDA | scikit-learn |
|---|---|---|---|---|
| convnext_v17 | `a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248` | 3.13.7 | `2.12.0+cu126` | 1.9.0 |
| panderm_partial4 | `08ffb6dbf84565c41f098b843c299cf00e9a0d80f4a8e7651ec568e1501b8be6` | 3.10.11 | `2.4.1+cu118` | 1.7.2 |
| convnext_natural_ce | `a55ff247594caa882a0cd3c45daabaa056fa864bdba850bfb172b6a3b0a67248` | 3.13.7 | `2.12.0+cu126` | 1.9.0 |

- PanDerm original folds: **1,699 / 1,695 / 1,713 / 1,702 / 1,721** images, **8,530 total**.
- ConvNeXt historical/canonical folds: **1,707 / 1,706 / 1,706 / 1,705 / 1,706** images, **8,530 total**.
- The replay tool validated that held-out image/lesion assignments match current metadata; this does not independently verify every historical training run.
- PanDerm upstream source checkout was independently observed by preflight at commit `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`; that is not a hash of local trained PanDerm checkpoints.

## Remaining scope / action boundaries

**Confirmed now:** all 15 particular checkpoint files can be reloaded in the selected local environment to reproduce the stored **development validation** macro F1 and accuracy within 0.003. Their hashes provide an identity manifest for backups. No source experiment report was modified.

**Still not confirmed:** independent rebuild/retraining; reproducibility from a fresh environment; image/preprocessing byte-level hashes and independent checkpoint backups; version-pinned source for ConvNeXt v18, PanDerm partial8/AdamW and ResNet18 aux02; resolution of ConvNeXt v11 CutMix mismatches; separate external/clinical validation. The stale archive README and duplicated editable-workspace `RUN_NAME` also remain operational tasks until the user updates them.

For any new training run, **do not use** `RUN_NAME=convnext_tiny_natural_ce_seed123` in the editable ConvNeXt workspace, because it collides with saved natural-CE results.

## Checkpoint identity manifest

The companion `Checkpoint_Replay_Checkpoint_Manifest_2026-10-14.csv` contains the full 64-character SHA-256 of each locally replayed checkpoint, run, original fold, byte size, saved epoch, reference metrics and replay status. Hashes are not backups; weights must be retained securely outside Git.

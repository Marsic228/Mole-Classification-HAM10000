# Phase 2 verification retention

Phase 3A preserved the complete historical verification bundle on 5 November 2026. This is a compact retention record, not a new experiment history.

## Scope and confirmed results

Phase 2 consolidated `evaluation/compare_predicted_roi.py`, `evaluation/export_predicted_roi.py` and `experiments/segmentation/train.py` for five canonical folds. Verification covered classification replay, predicted ROI export replay and segmentation training equivalence checks.

- All five retained classification CSVs reproduce 8,530 historical classification rows byte-for-byte. Phase 3A independently compared the existing files; no inference was rerun.
- All 8,530 retained ROI JPEGs match the historical output file sets, sizes and SHA-256 values. Phase 3A independently checked existing bytes; no export was rerun.
- The retained `final_tests.txt` records 20/20 automated tests passed. Historical initialization and loader-order checks, bounded synthetic optimizer-step checks, epoch/selection and checkpoint-schema checks passed. These are retained historical results, not fresh Phase 3A execution.

## Limits

Full historical retraining reproducibility was not established. Some mask/pixel/geometry comparisons used representative subsets, including 57 real-image samples plus fixtures; full historical masks were not retained. Installed libraries, JPEG behavior, devices and environment-specific numerical behavior remain reproducibility considerations. Historical test logs establish past execution, not a fresh rerun. Current source/checkpoint fingerprint equality does not authenticate exact historical training-source attribution. Phase 3A ran no model inference, training, optimizer updates or automated test suite.

## Evidence locations

- Complete permanent local bundle (local-only, excluded from Git): `data/verification/phase2-2026-11-05/`, relative to repository root and excluded by the existing `data/` ignore rule.
- [Retention manifest](retention_manifest.json): complete bundle-relative SHA-256 inventory, directory structure, file count, sizes, pre/post preservation checks, Git state and the distinction between historical results and Phase 3A checks.
- Original external Temp location, recorded only as historical provenance: `<LOCAL_TEMP>/ham10000-phase2-bab3dcea-483f-4382-b247-d07b0ac76b60/`.
- [Phase 2 implementation and verification record](../../../docs/research/Phase2_Script_Consolidation.md).
- [Phase 1 research history](../../../docs/research/Diploma_Experiment_History.md), [source provenance](../../../docs/research/SOURCE_PROVENANCE.md) and [evidence guide](../README.md).
- [Phase 1 source/checkpoint fingerprints](../phase1_artifact_manifest.json). The permanent bundle also retains `preservation_baseline.json`, `final_new_file_fingerprints.json`, `final_preservation_verification.json` and `accepted_external_relocations.json` unchanged.

## Retention and Phase 3A checks

The complete bundle contains **8,583 files, 59,359,920 bytes**. Source and destination relative file sets, directory sets, every file size and every SHA-256 digest matched. Embedded absolute paths and original filenames were preserved. Phase 3A also rechecked preservation baselines, original/consolidated source and tests, all existing checkpoint/data/report bytes, and the three repaired report links, allowing only the three authorized documentation edits.

The original Phase 1 and Phase 2 fingerprint records remain unchanged. The two already-relocated geometry artifacts are verified at their accepted paths. No original experimental evidence or Temp content was deleted. The initial source count/size and retained outputs agree with the preceding audit; a previously authenticated bundle-wide hash inventory of every scratch log was not available, so this manifest captures the source bytes observed during Phase 3A.

The full verification bundle stays local and ignored. Only this compact summary and manifest were created in the Git-visible reports directory; the 48 detailed logs and harnesses were not copied here. The summary and manifest can later be considered for Git after user review. No Git inclusion policy, attributes, report-index generator, environment or scientific result was changed.

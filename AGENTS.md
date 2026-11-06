# HAM10000 — working instructions for Codex

Goal: reliable, understandable diploma research with minimal maintenance overhead. Follow the user's **current approved task**; do not start unrelated work.

## Reuse first; do not duplicate fold scripts

- Inspect the current working tree, canonical modules, existing tests and documented run arguments before editing. Never overwrite uncommitted work.
- Prefer **one parameterized source per experiment family**, capable of processing folds 1–5 through a `--fold` argument and a simple sequential runner if needed. Produce separate **checkpoints, logs and reports** for each fold, not five copies of the source. New per-fold implementations require a specific justified difference and the user's consent.
- Existing consolidated segmentation entry points are `experiments/segmentation/train.py`, `evaluation/export_predicted_roi.py`, and `evaluation/compare_predicted_roi.py`. **Each currently requires `--fold 1..5`; none currently accepts `--all-folds`.** Verify the actual CLI before running; never invent flags. The 15 historical fold-specific scripts are preserved evidence, not starting templates.
- Keep shared model/data/loss logic in `core/`, experiments in `experiments/`, preprocessing in `preprocessing/`, evaluation/export in `evaluation/`, and numerical evidence under `reports/`. Avoid root-level scratch scripts and duplicate documentation.
- Verify frozen split identity (`splits/ham10000_dev_cv_v1.csv`), unique `image_id`, zero train/validation `lesion_id` overlap, class ordering and per-fold output naming. Never reuse an occupied `RUN_NAME` or overwrite prior checkpoints/reports. Match baselines on image IDs, not merely numeric fold IDs. Separate fold-mean F1 from pooled OOF F1 and development evidence from independent validation.

## Privacy by construction

- Do not put credentials, tokens, API keys, private keys, emails, usernames, account IDs, machine-specific home/Temp paths or sensitive environment values into published code, Git artifacts, reports, manifests, terminal excerpts or final summaries. Never print a detected secret in a finding.
- Prefer repository-relative paths, stable artifact IDs and explicit allowlisted report fields. If exact private provenance must be retained, keep its original in an **existing ignored local verification area** and publish only a sanitized copy. Preserve numerical evidence and explain any intentionally changed hashes.
- Before publishing, inspect generated text/JSON/CSV, including untracked files to be selected. **Scan the exact staged Git blobs**, not just the working tree. Do not stage datasets, masks, raw images, checkpoints, image archives, environments, private verification bundles or credentials.
- Use the repository's lightweight gate after exact-path staging:

  `python -B scripts/prepublish_check.py`

  The gate checks staged content only, reports locations/types of findings without revealing values, and is heuristic: a PASS is **not proof of zero secrets**. Block on any finding and investigate safely.

## Fast default workflow

1. Identify one concrete task and minimum affected files; use the current inspected source.
2. Make the approved smallest change; run targeted syntax/tests and one-fold preflight if relevant. Full historical audits, dataset hashes and five-fold reruns are for material scientific/provenance changes, **not every publish**.
3. Summarize changed paths, verification and open limits briefly. Obtain approval before Git staging/commits/pushes as applicable. Stage only explicitly approved exact paths, never `git add .` / `git add -A`.
4. Run `python -B scripts/prepublish_check.py` **after staging and before a commit**. Review `git diff --cached --stat` and the actual staged content; a failed gate must not be bypassed. Confirm committed content and remote fast-forward separately, only if explicitly authorized.

Do not silently change `.gitignore`, `.gitattributes`, existing experimental scripts, frozen splits, report indexes, historical evidence or Obsidian. No force-push, cleanup, archive, retraining or other phase without specific approval.

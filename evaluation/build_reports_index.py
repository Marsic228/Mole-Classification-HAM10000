r"""Generate a browsable index of Git-tracked research reports.

Install as evaluation/build_reports_index.py and run from the repository root:
    .\venv\Scripts\python.exe -m evaluation.build_reports_index

Does NOT move, delete, or modify existing report files.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUTPUT_README = REPORTS / "README.md"
OUTPUT_MANIFEST = REPORTS / "MANIFEST.csv"
FOLD_RE = re.compile(r"_fold(\d+)(?=_|$)", re.IGNORECASE)


def tracked_reports() -> list[Path]:
    """Use Git rather than globbing so links don't point to local-only files."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--", "reports/"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    paths = []
    for line in result.stdout.splitlines():
        p = ROOT / line
        if p.is_file() and p.resolve() not in {
            OUTPUT_README.resolve(), OUTPUT_MANIFEST.resolve()
        }:
            paths.append(p)
    return sorted(paths, key=lambda p: p.as_posix().casefold())


def segmentation_report_for(path: Path) -> tuple[str, str, str] | None:
    """Classify the reviewed segmentation evidence using its directory context."""
    try:
        relative = path.relative_to(REPORTS / "segmentation")
    except ValueError:
        return None

    root_reports = {
        "README.md": ("segmentation_documentation", "", "evidence guide"),
        "automatic_roi_5fold_summary.json": (
            "automatic_roi_reference_vs_predicted", "", "five-fold research summary"),
        "largest_component_roi_5fold_summary.json": (
            "largest_component_roi", "", "five-fold comparison summary"),
        "phase1_artifact_manifest.json": (
            "segmentation_provenance", "", "artifact fingerprint manifest"),
        "fold5_regressions_ranked_by_geometry.csv": (
            "roi_geometry_diagnostic", "5", "ranked regressions"),
        "roi_fold5_geometry_analysis_report.md": (
            "roi_geometry_diagnostic", "5", "historical geometry narrative"),
        "phase2_verification/README.md": (
            "phase2_verification", "", "verification documentation"),
        "phase2_verification/retention_manifest.json": (
            "phase2_verification", "", "retention inventory"),
    }
    if relative.as_posix() in root_reports:
        return root_reports[relative.as_posix()]

    parent = re.fullmatch(r"small_unet_fold([1-5])_seed123", relative.parent.as_posix())
    if not parent:
        raise ValueError(f"Unrecognized segmentation report: {relative}")
    fold = parent.group(1)
    named_fold = re.search(r"(?:^|_)fold(\d+)(?=_|$)", path.stem)
    if named_fold and named_fold.group(1) != fold:
        raise ValueError(f"Conflicting directory/filename folds: {relative}")

    fold_reports = {
        "history.json": ("small_unet_seed123", "segmentation training history"),
        "best_validation_per_image.csv": (
            "small_unet_seed123", "selected-validation Dice/IoU"),
        f"classifier_predicted_vs_reference_fold{fold}.csv": (
            "automatic_roi_reference_vs_predicted", "paired classification predictions"),
        f"largest_component_roi_comparison_fold{fold}.csv": (
            "largest_component_roi", "paired LCC classification predictions"),
        f"largest_component_roi_comparison_fold{fold}_summary.json": (
            "largest_component_roi", "fold LCC comparison summary"),
        "roi_retention_audit.csv": (
            "roi_retention_audit", "retention audit (256x192)"),
        "roi_retention_audit_original_resolution.csv": (
            "roi_retention_audit", "retention audit (original-resolution crop geometry)"),
        f"roi_geometry_diagnostic_fold{fold}.csv": (
            "roi_geometry_diagnostic", "per-image original-resolution geometry"),
        f"roi_geometry_diagnostic_fold{fold}_summary.json": (
            "roi_geometry_diagnostic", "fold geometry summary"),
        f"mask_component_audit_fold{fold}.csv": (
            "mask_component_audit", "per-image connected-component audit"),
        f"mask_component_audit_fold{fold}_summary.json": (
            "mask_component_audit", "fold connected-component summary"),
    }
    if path.name not in fold_reports:
        raise ValueError(f"Unrecognized segmentation report: {relative}")
    run, kind = fold_reports[path.name]
    return run, fold, kind


def run_for(path: Path, known_runs: list[str]) -> tuple[str, str, str]:
    """Return experiment/run, fold number, and report kind."""
    segmentation = segmentation_report_for(path)
    if segmentation is not None:
        return segmentation
    stem = path.stem
    match = FOLD_RE.search(stem)
    if match:
        run = stem[: match.start()]
        fold = match.group(1)
    else:
        # Associate aggregate reports with the longest matching run prefix.
        run = next((r for r in known_runs if stem.startswith(r + "_")), "")
        fold = ""
        if not run:
            for suffix in (
                "_per_class_cv_summary", "_cv_summary", "_per_class_summary",
                "_training_history", "_best_summary", "_evaluation_summary",
                "_ensemble_summary", "_summary", "_results",
            ):
                if stem.endswith(suffix):
                    run = stem[: -len(suffix)]
                    break
        if not run:
            run = "Other reports"

    if re.search(r"_epoch_\d+_report$", stem):
        kind = "epoch"
    elif stem.endswith("_best_summary"):
        kind = "best checkpoint summary"
    elif stem.endswith("_training_history"):
        kind = "training history"
    elif "cv_summary" in stem:
        kind = "cross-validation summary"
    elif stem.endswith("_evaluation") or re.search(r"_fold\d+_evaluation$", stem):
        kind = "evaluation"
    elif "oof" in stem:
        kind = "OOF"
    elif "summary" in stem:
        kind = "summary"
    else:
        kind = "artifact"
    return run, fold, kind


def category(run: str) -> str:
    if run in {"segmentation_provenance", "phase2_verification"}:
        return "Provenance and verification"
    if run in {
        "small_unet_seed123", "automatic_roi_reference_vs_predicted",
        "largest_component_roi", "roi_retention_audit",
        "roi_geometry_diagnostic", "mask_component_audit",
        "segmentation_documentation",
    }:
        return "Segmentation and automatic ROI"
    low = run.lower()
    if "convnext" in low and "panderm" in low or "ensemble" in low or low.startswith("oof_"):
        return "Ensembles and OOF"
    if "convnext" in low:
        return "ConvNeXt"
    if "panderm" in low:
        return "PanDerm"
    if "resnet" in low:
        return "ResNet"
    if "efficientnet" in low:
        return "EfficientNet"
    if "mobilenet" in low:
        return "MobileNet"
    if "metadata" in low:
        return "Metadata"
    return "Other"


def github_link(path: Path) -> str:
    relative = path.relative_to(REPORTS).as_posix()
    return quote(relative, safe="/")


def best_fold_f1(paths: list[Path]) -> str:
    """Read only explicit best-checkpoint summary values, never infer metrics."""
    by_fold = {}
    for path in paths:
        if not path.name.endswith("_best_summary.json"):
            continue
        match = FOLD_RE.search(path.stem)
        if not match:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        value = obj.get("best_macro_f1") if isinstance(obj, dict) else None
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            by_fold[int(match.group(1))] = float(value)
    # Only claim aggregate if a complete five-fold set exists.
    if set(by_fold) != {1, 2, 3, 4, 5}:
        return "—"
    vals = [by_fold[i] for i in range(1, 6)]
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def choose_links(paths: list[Path]) -> str:
    def prioritize(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        if "cv_summary" in name:
            priority = 0
        elif name.endswith("_best_summary.json") and "_fold1_" in name:
            priority = 1
        elif "oof" in name and path.suffix.lower() == ".json":
            priority = 2
        elif "_training_history" in name and "_fold1_" in name:
            priority = 3
        elif name.endswith("summary.json"):
            priority = 4
        else:
            priority = 9
        return (priority, name)

    picks = sorted(paths, key=prioritize)[:2]
    return " · ".join(f"[{'CV summary' if 'cv_summary' in p.name else p.name}]({github_link(p)})" for p in picks)


def main() -> None:
    paths = tracked_reports()
    if not paths:
        raise SystemExit("No Git-tracked report files found under reports/.")

    known_runs = sorted({
        p.stem[:m.start()]
        for p in paths
        if not p.is_relative_to(REPORTS / "segmentation")
        and (m := FOLD_RE.search(p.stem))
    }, key=len, reverse=True)

    groups: dict[str, list[Path]] = defaultdict(list)
    folds_by_run: dict[str, set[int]] = defaultdict(set)
    manifest = []
    for path in paths:
        run, fold, kind = run_for(path, known_runs)
        groups[run].append(path)
        if fold:
            folds_by_run[run].add(int(fold))
        manifest.append({
            "file": path.relative_to(ROOT).as_posix(),
            "run": run,
            "category": category(run),
            "fold": fold,
            "type": kind,
            "bytes": path.stat().st_size,
        })

    REPORTS.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=[
            "file", "run", "category", "fold", "type", "bytes"
        ])
        writer.writeheader()
        writer.writerows(manifest)

    cats = defaultdict(list)
    for run, grouped_paths in groups.items():
        cats[category(run)].append((run, grouped_paths))

    lines = [
        "# Research Report Index",
        "",
        f"Automatically indexed **{len(paths):,} Git-tracked report artifacts** "
        f"across **{len(groups)} experiment/report groups**.",
        "",
        "**The files are still stored at their original paths.** This index does not "
        "rename, move, or delete them. The [CSV manifest](MANIFEST.csv) lists every file "
        "and can be filtered by experiment, fold, or report type.",
        "",
        "Important: reported metrics use different validation protocols. Do not compare "
        "fold means with pooled OOF or independent-test results as if they were identical.",
        "",
        "**Known invalid historical report:** "
        "[`convnext_v17_panderm_partial4_ensemble_summary.json`]"
        "(convnext_v17_panderm_partial4_ensemble_summary.json) has "
        "`all_reference_checks_passed: false` and must **not** be cited as a valid "
        "ensemble result. The separately verified pooled OOF comparison is in "
        "[`oof_convnext_panderm_ensemble_results.json`]"
        "(oof_convnext_panderm_ensemble_results.json).",
        "",
        "**Segmentation evidence:** [guide](segmentation/README.md), "
        "[Phase 2 verification](segmentation/phase2_verification/README.md), and "
        "[diagnostic source and numerical evidence](../roi_diagnostics/). "
        "Private verification bundles stay local and are not indexed.",
        "",
        "## Browse by model family",
        "",
    ]
    cat_order = ["ConvNeXt", "PanDerm", "Ensembles and OOF", "ResNet", "EfficientNet", "MobileNet", "Metadata", "Segmentation and automatic ROI", "Provenance and verification", "Other"]
    for cat in cat_order:
        if cat in cats:
            anchor = cat.lower().replace(" ", "-")
            lines.append(f"- [{cat}](#{anchor}) — {len(cats[cat])} groups")
    lines.append("")
    for cat in cat_order:
        if cat not in cats:
            continue
        lines += [f"## {cat}", "", "| Run / report group | Files | Folds | Best-checkpoint macro F1 (5 folds) | Starting points |", "|---|---:|---:|---:|---|"]
        for run, gpaths in sorted(cats[cat], key=lambda x: x[0].casefold()):
            fold_ids = folds_by_run[run]
            fold_str = ", ".join(str(i) for i in sorted(fold_ids)) if fold_ids else "—"
            run_md = run.replace("|", "\\|")
            lines.append(f"| `{run_md}` | {len(gpaths)} | {fold_str} | {best_fold_f1(gpaths)} | {choose_links(gpaths)} |")
        lines.append("")
    lines += [
        "## How to keep this index current",
        "",
        "From the project root, run:",
        "",
        "```powershell",
        r".\venv\Scripts\python.exe -m evaluation.build_reports_index",
        "```",
        "",
        "Commit the regenerated `reports/README.md` and `reports/MANIFEST.csv` "
        "alongside newly completed reports. The index intentionally contains **only "
        "Git-tracked artifacts**; stage any new report files first, then regenerate.",
        "",
    ]
    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Indexed {len(paths)} tracked report files in {len(groups)} groups.")
    print("Created reports/README.md and reports/MANIFEST.csv.")
    print("Only reports/README.md and reports/MANIFEST.csv were regenerated; evidence files were unchanged.")


if __name__ == "__main__":
    main()

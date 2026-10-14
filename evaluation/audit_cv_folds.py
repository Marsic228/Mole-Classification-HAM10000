"""Export and compare the REAL fold assignments from cross_validation_split.py.

Run separately in each Python environment, from the project root. This script
DOES NOT change model code or train a model. It never rewrites an existing export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from collections import Counter

FIELDS = ("image_id", "lesion_id", "dx", "source_split", "fold")
PACKAGES = ("scikit-learn", "torch", "torchvision", "timm", "pandas", "numpy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_export(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not set(FIELDS).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing columns in {path}: {FIELDS}")
        by_id = {}
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in by_id:
                raise ValueError(f"Blank or duplicate image_id in {path}: {image_id}")
            if row["fold"] not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"Invalid fold for {image_id}")
            by_id[image_id] = row
    return by_id


def export(args: argparse.Namespace) -> None:
    # Import exactly the splitting implementation used by existing ConvNeXt
    # code. Check its identity in the saved manifest for other projects.
    project = Path(args.project).resolve()
    if not (project / "cross_validation_split.py").is_file():
        raise FileNotFoundError(f"cross_validation_split.py not found in {project}")
    os.chdir(project)
    sys.path.insert(0, str(project))
    module = importlib.import_module("core.cross_validation_split")

    metadata = module.metadata.reset_index(drop=True)
    required = {"image_id", "lesion_id", "dx", "source_split"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {required - set(metadata.columns)}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Missing image/lesion/class/source values")
    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in the development pool")
    if set(metadata["source_split"]) != {"train", "val"}:
        raise ValueError("Unexpected source split; test-set contamination is possible")
    if len(metadata) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} images, found {len(metadata)}")

    all_rows = []
    seen = set()
    fold_stats = []
    for fold, (train_idx, val_idx) in enumerate(
        module.splitter.split(
            metadata[["image_id"]], metadata["dx"], metadata["lesion_id"]
        ),
        start=1,
    ):
        train = metadata.iloc[train_idx]
        val = metadata.iloc[val_idx]
        overlap = set(train["lesion_id"]) & set(val["lesion_id"])
        if overlap:
            raise ValueError(f"Lesion leakage in fold {fold}: {len(overlap)}")
        if fold == 1 and args.expected_fold1_val is not None:
            if len(val) != args.expected_fold1_val:
                raise ValueError(
                    f"Fold 1 has {len(val)} validation images, expected "
                    f"{args.expected_fold1_val}. STOP: this may be the wrong "
                    "environment or splitting implementation."
                )
        fold_stats.append({
            "fold": fold, "train": len(train), "validation": len(val),
            "lesion_overlap": len(overlap),
            "validation_class_counts": dict(sorted(Counter(val["dx"]).items())),
        })
        for image_id, lesion_id, dx, source_split in val[
            ["image_id", "lesion_id", "dx", "source_split"]
        ].itertuples(index=False, name=None):
            if image_id in seen:
                raise ValueError(f"Image assigned to multiple validation folds: {image_id}")
            seen.add(image_id)
            all_rows.append({
                "image_id": str(image_id), "lesion_id": str(lesion_id),
                "dx": str(dx), "source_split": str(source_split),
                "fold": str(fold),
            })

    if len(fold_stats) != 5 or len(seen) != len(metadata):
        raise ValueError("Expected 5 folds and exactly one validation fold per image")
    lesion_folds = {}
    for row in all_rows:
        lesion = row["lesion_id"]
        if lesion in lesion_folds and lesion_folds[lesion] != row["fold"]:
            raise ValueError(f"Lesion is split across folds: {lesion}")
        lesion_folds[lesion] = row["fold"]

    outdir = (project / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    csv_file = outdir / f"{args.tag}_folds.csv"
    manifest_file = outdir / f"{args.tag}_environment.json"
    if csv_file.exists() or manifest_file.exists():
        raise FileExistsError("Export already exists; use a new tag to preserve evidence")

    with csv_file.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: row["image_id"]))

    relevant_files = (
        "cross_validation_split.py", "split_dataset.py", "split_config.py",
        "validate_image_paths.py", "HAM10000_metadata.csv",
    )
    manifest = {
        "tag": args.tag,
        "script_purpose": "original runtime fold assignment audit; not new canonical folds",
        "project_root": str(project),
        "split_module": str(Path(module.__file__).resolve()),
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {package: version(package) for package in PACKAGES},
        "full_environment_packages": dict(sorted(
            (dist.metadata.get("Name", "unknown"), dist.version)
            for dist in importlib.metadata.distributions()
        )),
        "class_order_alphabetical": sorted(set(metadata["dx"])),
        "source_file_sha256": {
            name: sha256_file(project / name)
            for name in relevant_files if (project / name).is_file()
        },
        "images": len(all_rows),
        "lesions": len(lesion_folds),
        "csv_file": str(csv_file),
        "csv_sha256": sha256_file(csv_file),
        "fold_stats": fold_stats,
    }
    with manifest_file.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Exported {len(all_rows)} images, {len(lesion_folds)} lesions")
    for s in fold_stats:
        print(f"  Fold {s['fold']}: train={s['train']}, val={s['validation']}, overlap=0")
    print(f"CSV: {csv_file}\nManifest: {manifest_file}")


def compare(args: argparse.Namespace) -> None:
    left = read_export(Path(args.left))
    right = read_export(Path(args.right))
    if left.keys() != right.keys():
        raise ValueError(
            f"Different image sets: left-only={len(left.keys() - right.keys())}, "
            f"right-only={len(right.keys() - left.keys())}; cannot compare fold IDs"
        )
    for image_id in left:
        for field in ("lesion_id", "dx", "source_split"):
            if left[image_id][field] != right[image_id][field]:
                raise ValueError(f"{field} mismatch for {image_id}")
    matches = sum(left[k]["fold"] == right[k]["fold"] for k in left)
    n = len(left)
    print(f"Matched image IDs and true labels: {n}/{n}")
    print(f"Same numeric fold: {matches}/{n} ({matches/n:.1%})")
    print(f"Different numeric fold: {n-matches}/{n} ({1-matches/n:.1%})")
    print("Outcome: " + (
        "same fold membership by ID" if matches == n
        else "DIFFERENT FOLDS — never match checkpoints by fold number alone"
    ))
    print("Next: preserve BOTH exports; decide one canonical split for NEW training only.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="Run separately in each model's environment")
    exp.add_argument("--tag", required=True, help="e.g. convnext or panderm")
    exp.add_argument("--project", default=".", help="repository root")
    exp.add_argument("--output-dir", default="reports/repro_audit")
    exp.add_argument("--expected-images", type=int, default=8530)
    exp.add_argument("--expected-fold1-val", type=int)
    exp.set_defaults(func=export)
    cmp = sub.add_parser("compare", help="Compare two completed CSV exports")
    cmp.add_argument("--left", required=True)
    cmp.add_argument("--right", required=True)
    cmp.set_defaults(func=compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

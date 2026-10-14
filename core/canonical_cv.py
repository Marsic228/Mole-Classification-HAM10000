"""Freeze and use one lesion-safe HAM10000 development CV assignment.

Run from the Mole Classification repository root.
This does not overwrite the original ConvNeXt/PanDerm exports or retrain models.
For NEW experiments only: `from core.canonical_cv import get_fold_metadata`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

DEFAULT_CSV = Path("splits/ham10000_dev_cv_v1.csv")
DEFAULT_MANIFEST = Path("splits/ham10000_dev_cv_v1.json")
REQUIRED_COLUMNS = ("image_id", "lesion_id", "dx", "source_split", "fold")
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
N_IMAGES = 8530
N_LESIONS = 6349


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(piece)
    return sha.hexdigest()


def load_assignments(path: Path) -> tuple[dict[str, dict], dict[int, dict]]:
    assignments: dict[str, dict] = {}
    lesion_to_fold: dict[str, int] = {}
    fold_counts: dict[int, Counter] = {f: Counter() for f in range(1, 6)}
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if not set(REQUIRED_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Bad columns in {path}: need {REQUIRED_COLUMNS}")
        for row in reader:
            image = row["image_id"].strip()
            lesion = row["lesion_id"].strip()
            dx = row["dx"].strip()
            source = row["source_split"].strip()
            fold_s = row["fold"].strip()
            if not image or not lesion or image in assignments:
                raise ValueError(f"Blank/duplicate image_id or empty lesion_id: {image!r}")
            if dx not in CLASS_ORDER or source not in {"train", "val"}:
                raise ValueError(f"Invalid class/source for {image}: {dx}, {source}")
            if fold_s not in {"1", "2", "3", "4", "5"}:
                raise ValueError(f"Invalid fold {fold_s!r} for {image}")
            fold = int(fold_s)
            if lesion in lesion_to_fold and lesion_to_fold[lesion] != fold:
                raise ValueError(f"Lesion leakage: {lesion} is assigned to multiple folds")
            lesion_to_fold[lesion] = fold
            assignments[image] = dict(image_id=image, lesion_id=lesion,
                                      dx=dx, source_split=source, fold=fold)
            fold_counts[fold][dx] += 1

    if len(assignments) != N_IMAGES or len(lesion_to_fold) != N_LESIONS:
        raise ValueError(f"Expected {N_IMAGES} images/{N_LESIONS} lesions; got "
                         f"{len(assignments)}/{len(lesion_to_fold)}")
    if any(sum(c.values()) == 0 for c in fold_counts.values()):
        raise ValueError("Some folds have no validation images")
    return assignments, {f: {"val": sum(c.values()), "by_class": dict(sorted(c.items()))}
                         for f, c in fold_counts.items()}


def check_against_metadata(assignments: dict[str, dict]):
    # Existing project metadata construction; do NOT regenerate splits here.
    from core.cross_validation_split import metadata
    meta = metadata.copy().reset_index(drop=True)
    for col in ("image_id", "lesion_id", "dx", "source_split"):
        if col not in meta.columns or meta[col].isna().any():
            raise ValueError(f"Metadata missing/NULL field: {col}")
        meta[col] = meta[col].astype(str)
    if meta["image_id"].duplicated().any():
        raise ValueError("Metadata has duplicate image IDs")
    if set(meta["image_id"]) != set(assignments):
        raise ValueError("Canonical IDs do not match current development pool")
    for image, lesion, dx, source in meta[
        ["image_id", "lesion_id", "dx", "source_split"]
    ].itertuples(index=False, name=None):
        old = assignments[image]
        if (old["lesion_id"], old["dx"], old["source_split"]) != (lesion, dx, source):
            raise ValueError(f"Metadata differs from frozen export: {image}")
    return meta


def verified_assignments(csv_path=DEFAULT_CSV, manifest_path=DEFAULT_MANIFEST):
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    actual = checksum(csv_path)
    if actual != manifest.get("frozen_csv_sha256"):
        raise ValueError("Frozen CSV SHA256 does not match its manifest; STOP")
    rows, counts = load_assignments(csv_path)
    return rows, counts


def get_fold_metadata(fold_number: int, csv_path=DEFAULT_CSV,
                      manifest_path=DEFAULT_MANIFEST):
    """Return train, validation DataFrames using permanent image IDs, not row indices.

    Use ONLY for new experiments. Existing saved checkpoints retain their own folds.
    """
    if fold_number not in range(1, 6):
        raise ValueError("fold_number must be 1..5")
    assignments, _ = verified_assignments(csv_path, manifest_path)
    meta = check_against_metadata(assignments)
    val_image_ids = {k for k, row in assignments.items() if row["fold"] == fold_number}
    is_val = meta["image_id"].isin(val_image_ids)
    train, validation = meta.loc[~is_val].copy(), meta.loc[is_val].copy()
    if set(train["lesion_id"]) & set(validation["lesion_id"]):
        raise ValueError("Lesion leakage between train and validation")
    return train, validation


def freeze(args):
    source = Path(args.source)
    origin_manifest = Path(args.origin_manifest)
    dest, dest_manifest = Path(args.dest), Path(args.dest_manifest)
    if dest.exists() or dest_manifest.exists():
        raise FileExistsError("Frozen destination already exists. Never overwrite it.")
    with origin_manifest.open("r", encoding="utf-8") as fp:
        original = json.load(fp)
    if original.get("tag") != "convnext":
        raise ValueError("Expected original ConvNeXt export manifest")
    if checksum(source) != original.get("csv_sha256"):
        raise ValueError("Original source hash mismatch; STOP")
    assignments, counts = load_assignments(source)
    check_against_metadata(assignments)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact bytes from the audited ConvNeXt export.
    with source.open("rb") as src, dest.open("xb") as target:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            target.write(chunk)
    data = {
        "name": "ham10000_dev_cv_v1",
        "use": "Future experiments ONLY; historical checkpoints retain their original folds",
        "origin": "convnext (previously audited venv, NOT PanDerm original environment)",
        "origin_export": str(source),
        "origin_export_sha256": original["csv_sha256"],
        "origin_environment_manifest": str(origin_manifest),
        "origin_package_versions": original.get("package_versions", {}),
        "origin_source_sha256": original.get("source_file_sha256", {}),
        "frozen_csv_sha256": checksum(dest),
        "n_images": len(assignments),
        "n_lesions": len({r["lesion_id"] for r in assignments.values()}),
        "n_folds": 5,
        "fold_counts": counts,
        "class_order": list(CLASS_ORDER),
    }
    try:
        with dest_manifest.open("x", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    print(f"LOCKED: {dest} ({len(assignments)} images, 5 folds)")
    print(f"SHA256: {data['frozen_csv_sha256']}")
    print(f"Manifest: {dest_manifest}")
    for f, count in counts.items():
        print(f"  Fold {f}: train={N_IMAGES-count['val']}, val={count['val']}")
    print("Existing ConvNeXt/PanDerm checkpoints and exports are NOT changed.")


def verify(args):
    assignments, counts = verified_assignments(args.csv, args.manifest)
    meta = check_against_metadata(assignments)
    for fold in range(1, 6):
        ids = {k for k, row in assignments.items() if row["fold"] == fold}
        mask = meta["image_id"].isin(ids)
        train = meta.loc[~mask]
        val = meta.loc[mask]
        if set(train["lesion_id"]) & set(val["lesion_id"]):
            raise ValueError(f"Lesion leakage in fold {fold}")
        if len(val) != counts[fold]["val"]:
            raise ValueError("Fold count disagreement")
        print(f"Fold {fold}: train={len(train)}, val={len(val)}, overlap=0")
    print(f"PASS: {len(assignments)} image IDs match the live metadata; checksum OK")


def envdiff(args):
    with Path(args.left).open("r", encoding="utf-8") as fp:
        left = json.load(fp)
    with Path(args.right).open("r", encoding="utf-8") as fp:
        right = json.load(fp)
    print("Environment comparison (differences only):")
    for name in sorted(set(left.get("package_versions", {})) |
                       set(right.get("package_versions", {}))):
        a = left.get("package_versions", {}).get(name)
        b = right.get("package_versions", {}).get(name)
        if a != b:
            print(f"  {name}: ConvNeXt={a!r}; PanDerm={b!r}")
    for name in sorted(set(left.get("source_file_sha256", {})) |
                       set(right.get("source_file_sha256", {}))):
        a = left.get("source_file_sha256", {}).get(name)
        b = right.get("source_file_sha256", {}).get(name)
        if a != b:
            print(f"  FILE CONTENT DIFFERS: {name} (ConvNeXt={str(a)[:12]} / PanDerm={str(b)[:12]})")
    print("Different package versions suggest possible causes; this does NOT prove the root cause.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="task", required=True)
    f = sub.add_parser("freeze", help="Lock ConvNeXt export for new experiments")
    f.add_argument("--source", default="reports/repro_audit/convnext_folds.csv")
    f.add_argument("--origin-manifest", default="reports/repro_audit/convnext_environment.json")
    f.add_argument("--dest", default=str(DEFAULT_CSV))
    f.add_argument("--dest-manifest", default=str(DEFAULT_MANIFEST))
    f.set_defaults(func=freeze)
    v = sub.add_parser("verify", help="Validate frozen assignment with live project metadata")
    v.add_argument("--csv", default=str(DEFAULT_CSV))
    v.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    v.set_defaults(func=verify)
    e = sub.add_parser("envdiff", help="Show environmental differences")
    e.add_argument("--left", default="reports/repro_audit/convnext_environment.json")
    e.add_argument("--right", default="reports/repro_audit/panderm_environment.json")
    e.set_defaults(func=envdiff)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

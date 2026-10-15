"""Read-only, single-fold replay of saved HAM10000 research checkpoints.

Run from the Mole Classification repository root. Uses the ORIGINAL saved
image-to-fold export for each model, not a re-generated sklearn split.

Examples:
    .\\venv\\Scripts\\python.exe -m evaluation.replay_checkpoint_fold --model convnext_v17 --fold 1
    .\\PanDerm\\.venv-panderm\\Scripts\\python.exe -m evaluation.replay_checkpoint_fold --model panderm_partial4 --fold 1 --batch-size 4

Never writes to reports/ or checkpoints/. Produces a new JSON in local_audit/.
This is a development-validation replay, NOT an independent test or a
proof of exact retraining reproducibility.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "convnext_v17": {
        "tag": "convnext_tiny_finetune_v17_warmup_seed123",
        "fold_csv": "reports/repro_audit/convnext_folds.csv",
        "reference": "evaluation",
    },
    "convnext_natural_ce": {
        "tag": "convnext_tiny_natural_ce_seed123",
        "fold_csv": "splits/ham10000_dev_cv_v1.csv",
        "reference": "best_summary",
    },
    "panderm_partial4": {
        "tag": "panderm_base_partial4_ema_seed123",
        "fold_csv": "reports/repro_audit/panderm_folds.csv",
        "reference": "evaluation",
    },
}
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_validation_metadata(path: Path, fold: int):
    """Use image IDs exported with this model; reject mismatched metadata."""
    from core.cross_validation_split import metadata

    expected = {"image_id", "lesion_id", "dx", "source_split", "fold"}
    rows = {}
    lesion_to_fold = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Fold CSV missing columns: {path}")
        for row in reader:
            image_id, lesion_id = row["image_id"], row["lesion_id"]
            fold_id = int(row["fold"])
            if fold_id not in range(1, 6) or not image_id or not lesion_id or image_id in rows:
                raise ValueError("Bad/duplicate image or fold in saved export")
            if lesion_id in lesion_to_fold and lesion_to_fold[lesion_id] != fold_id:
                raise ValueError(f"Lesion leakage within saved export: {lesion_id}")
            lesion_to_fold[lesion_id] = fold_id
            rows[image_id] = row
    if len(rows) != 8530:
        raise ValueError(f"Saved fold export has {len(rows)} rows, expected 8530")

    frame = metadata.copy().reset_index(drop=True)
    if len(frame) != 8530 or frame["image_id"].duplicated().any():
        raise ValueError("Live development metadata count/IDs differ")
    if set(frame["image_id"].astype(str)) != set(rows):
        raise ValueError("Saved fold export image IDs differ from live metadata")

    for image_id, lesion_id, dx, source_split in frame[
        ["image_id", "lesion_id", "dx", "source_split"]
    ].astype(str).itertuples(index=False, name=None):
        stored = rows[image_id]
        if (stored["lesion_id"], stored["dx"], stored["source_split"]) != (
            lesion_id, dx, source_split
        ):
            raise ValueError(f"Saved metadata mismatch for {image_id}")

    val_ids = {image_id for image_id, row in rows.items() if int(row["fold"]) == fold}
    val = frame.loc[frame["image_id"].isin(val_ids)].copy()
    train = frame.loc[~frame["image_id"].isin(val_ids)]
    if len(val) != len(val_ids):
        raise ValueError("Validation size mismatch")
    if set(train["lesion_id"]) & set(val["lesion_id"]):
        raise ValueError("Train/validation lesions overlap")
    return val, len(train), len(rows), sha256(path)


def load_model(kind: str, checkpoint: Path, device):
    import torch
    from torchvision.models import convnext_tiny
    import torch.nn as nn

    if kind.startswith("convnext_"):
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 7)
    else:
        from experiments.panderm.panderm_cv_train import build_panderm_model
        model = build_panderm_model()

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError("Checkpoint missing model_state_dict")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).eval(), saved.get("epoch"), len(saved["model_state_dict"])


def replay(kind: str, fold: int, batch_size: int, tolerance: float, device_name: str, hash_checkpoint: bool):
    import numpy as np
    import torch
    import torchvision
    import sklearn
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from torchvision.models import ConvNeXt_Tiny_Weights
    from core.cross_validation_dataset import CrossValidationDataset

    config = CONFIGS[kind]
    tag = config["tag"]
    fold_csv = ROOT / config["fold_csv"]
    ckpt = ROOT / "checkpoints" / f"{tag}_fold{fold}_best.pt"
    reference_path = ROOT / "reports" / (
        f"{tag}_fold{fold}_{config['reference']}.json"
    )
    for path in (fold_csv, ckpt, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    val, train_count, total, fold_hash = frozen_validation_metadata(fold_csv, fold)
    dataset = CrossValidationDataset(
        val, ROOT / "data" / "processed", transform=ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    )
    if dataset.classes != CLASSES:
        raise ValueError(f"Unexpected dataset class order: {dataset.classes}")
    if len(dataset) != len(val):
        raise ValueError("Dataset and exported fold have different sizes")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model, epoch, state_dict_entries = load_model(kind, ckpt, device)
    predictions = []
    labels = []
    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(batch_images)
            if logits.ndim != 2 or logits.shape[1] != len(CLASSES):
                raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits")
            predictions.extend(logits.float().argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    del model
    if len(labels) != len(dataset) or labels != dataset.targets:
        raise ValueError("Validation targets do not align")

    f1 = float(f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0))
    accuracy = float(accuracy_score(labels, predictions))
    reference = json.loads(reference_path.read_text(encoding="utf-8-sig"))
    expected_f1 = float(reference["macro_f1"] if config["reference"] == "evaluation" else reference["best_macro_f1"])
    summary_path = ROOT / "reports" / f"{tag}_fold{fold}_best_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
    expected_epoch = summary.get("best_epoch")
    if config["reference"] == "evaluation":
        expected_accuracy = float(reference["overall_accuracy"])
        if reference.get("class_names") != CLASSES:
            raise ValueError("Saved independent evaluation uses a different class order")
        if reference.get("total_checked") != len(dataset):
            raise ValueError("Saved independent evaluation count differs from export")
    else:
        best_epoch_report = ROOT / "reports" / f"{tag}_fold{fold}_epoch_{expected_epoch}_report.json"
        epoch_metrics = json.loads(best_epoch_report.read_text(encoding="utf-8-sig"))
        expected_accuracy = float(epoch_metrics["val_accuracy"])
    delta = f1 - expected_f1
    accuracy_delta = accuracy - expected_accuracy
    epoch_match = (int(epoch) == int(expected_epoch)) if epoch is not None and expected_epoch is not None else None
    status = "PASS" if abs(delta) <= tolerance and abs(accuracy_delta) <= tolerance and epoch_match is True else "MISMATCH"
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)

    output = {
        "kind": kind,
        "fold": fold,
        "status": status,
        "tolerance": tolerance,
        "reference_kind": config["reference"],
        "reported_reference_is_selected_validation": True,
        "method": "replay from original saved fold CSV; inference only, no retraining",
        "git_head": git.stdout.strip() if git.returncode == 0 else None,
        "saved_fold_export": str(fold_csv.relative_to(ROOT)),
        "saved_fold_export_sha256": fold_hash,
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "checkpoint_bytes": ckpt.stat().st_size,
        "checkpoint_sha256": sha256(ckpt) if hash_checkpoint else None,
        "checkpoint_epoch": epoch,
        "expected_best_epoch": expected_epoch,
        "checkpoint_epoch_matches_summary": epoch_match,
        "checkpoint_state_dict_entries": state_dict_entries,
        "reference_file": str(reference_path.relative_to(ROOT)),
        "train_count": train_count,
        "validation_count": len(dataset),
        "saved_export_total": total,
        "macro_f1_replayed": f1,
        "macro_f1_reference": expected_f1,
        "macro_f1_difference": delta,
        "accuracy_replayed": accuracy,
        "accuracy_reference": expected_accuracy,
        "accuracy_difference": accuracy_delta,
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = ROOT / "local_audit" / "replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{kind}_fold{fold}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with outfile.open("x", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)
        fp.write("\n")
    print(f"{kind}, fold {fold}: {status}; macro F1 replayed={f1:.6f} / expected={expected_f1:.6f} (delta {delta:+.6f}); accuracy={accuracy:.6f} / expected={expected_accuracy:.6f}; epoch match={epoch_match}")
    print(f"Saved: {outfile.relative_to(ROOT)}")
    return status == "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--tolerance", type=float, default=0.003)
    parser.add_argument("--hash-checkpoint", action="store_true", help="Also SHA256-hash the checkpoint file (slower)")
    args = parser.parse_args()
    if args.batch_size < 1 or args.tolerance < 0:
        parser.error("batch-size must be >= 1; tolerance must be nonnegative")
    ok = replay(args.model, args.fold, args.batch_size, args.tolerance, args.device, args.hash_checkpoint)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

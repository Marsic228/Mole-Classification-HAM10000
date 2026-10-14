"""Export out-of-fold predictions using EACH MODEL'S ORIGINAL ENVIRONMENT.

Run twice, from the Mole Classification project root:

    python -m evaluation.export_model_oof_predictions --model convnext
    .\\PanDerm\\.venv-panderm\\Scripts\\python.exe -m evaluation.export_model_oof_predictions --model panderm

Never evaluate one model on the other environment's fold list.
The two NPZ files can then be combined by image ID with merge_model_oof_ensemble.py.
This is DEVELOPMENT OOF evaluation, not an external clinical test.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
import torch.nn as nn

from cross_validation_dataset import CrossValidationDataset
from cross_validation_split import get_fold_metadata

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CHECKPOINTS = {
    "convnext": "checkpoints/convnext_tiny_finetune_v17_warmup_seed123_fold{fold}_best.pt",
    "panderm": "checkpoints/panderm_base_partial4_ema_seed123_fold{fold}_best.pt",
}
EXPECTED_F1 = {
    "convnext": [0.7481, 0.7451, 0.7586, 0.7195, 0.7437],
    "panderm": [0.7464, 0.7469, 0.7638, 0.7216, 0.7052],
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = ConvNeXt_Tiny_Weights.DEFAULT.transforms()


def load_model(model_name: str, checkpoint_path: Path) -> nn.Module:
    if model_name == "convnext":
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 7)
    else:
        from experiments.panderm.panderm_cv_train import build_panderm_model
        model = build_panderm_model()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(DEVICE).eval()


@torch.inference_mode()
def predict(model, data_loader):
    chunks = []
    for images, _ in data_loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            logits = model(images)
        if logits.shape[1] != len(CLASSES):
            raise ValueError(f"Incorrect number of classes: {tuple(logits.shape)}")
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CHECKPOINTS), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    print(f"Model: {args.model} | device: {DEVICE} | scikit-learn: {SKLEARN_VERSION}", flush=True)
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_all, lesions_all, labels_all, fold_all, probs_all = [], [], [], [], []
    fold_reports = []
    for fold in range(1, 6):
        checkpoint_path = Path(CHECKPOINTS[args.model].format(fold=fold))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        train_metadata, val_metadata = get_fold_metadata(fold)
        # This explicitly checks the lesion-safe split in THIS interpreter.
        train_lesions = set(train_metadata["lesion_id"].astype(str))
        val_lesions = set(val_metadata["lesion_id"].astype(str))
        overlap = train_lesions.intersection(val_lesions)
        if overlap:
            raise RuntimeError(f"Fold {fold}: {len(overlap)} lesions leak into training")

        dataset = CrossValidationDataset(val_metadata, "data/processed", transform=TRANSFORM)
        if list(dataset.classes) != CLASSES:
            raise RuntimeError(f"Class order mismatch: {dataset.classes}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"Fold {fold}: {len(train_metadata)} train / {len(dataset)} validation; loading {checkpoint_path.name}", flush=True)
        model = load_model(args.model, checkpoint_path)
        probabilities = predict(model, loader)
        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        image_ids = val_metadata["image_id"].astype(str).tolist()
        lesion_ids = val_metadata["lesion_id"].astype(str).tolist()
        labels = np.asarray(dataset.targets, dtype=np.int64)
        if not (len(image_ids) == len(lesion_ids) == len(labels) == probabilities.shape[0]):
            raise RuntimeError(f"Fold {fold}: metadata and probabilities do not align")
        correct_class_ids = [CLASSES.index(name) for name in val_metadata["dx"]]
        if not np.array_equal(labels, np.asarray(correct_class_ids, dtype=np.int64)):
            raise RuntimeError(f"Fold {fold}: label order mismatch")
        if len(image_ids) != len(set(image_ids)):
            raise RuntimeError(f"Fold {fold}: duplicate image IDs")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise RuntimeError(f"Fold {fold}: invalid softmax probabilities")
        f1 = f1_score(labels, probabilities.argmax(axis=1), labels=list(range(7)), average="macro", zero_division=0)
        accuracy = accuracy_score(labels, probabilities.argmax(axis=1))
        expected = EXPECTED_F1[args.model][fold - 1]
        delta = f1 - expected
        print(f"  accuracy={accuracy:.4f}, macro F1={f1:.4f}, reference={expected:.4f}, delta={delta:+.4f}", flush=True)
        fold_reports.append({"fold": fold, "train_count": len(train_metadata), "val_count": len(dataset),
                             "macro_f1": float(f1), "accuracy": float(accuracy), "expected_f1": expected,
                             "reference_difference": float(delta)})

        ids_all.extend(image_ids)
        lesions_all.extend(lesion_ids)
        labels_all.extend(labels.tolist())
        fold_all.extend([fold] * len(image_ids))
        probs_all.append(probabilities)

    if len(ids_all) != 8530 or len(ids_all) != len(set(ids_all)):
        raise RuntimeError(f"Expected 8530 unique OOF image IDs; got {len(ids_all)} rows, {len(set(ids_all))} unique")
    lesion_to_fold = {}
    for lesion_id, fold in zip(lesions_all, fold_all):
        if lesion_id in lesion_to_fold and lesion_to_fold[lesion_id] != fold:
            raise RuntimeError(f"Lesion {lesion_id} appears in multiple validation folds")
        lesion_to_fold[lesion_id] = fold
    if any(abs(r["reference_difference"]) > 0.003 for r in fold_reports):
        raise RuntimeError("One or more model reference scores did not reproduce. No NPZ was written; investigate environment/preprocessing.")

    npz_path = output_dir / f"oof_{args.model}_original_folds.npz"
    np.savez_compressed(npz_path,
                        image_ids=np.asarray(ids_all, dtype=str),
                        lesion_ids=np.asarray(lesions_all, dtype=str),
                        labels=np.asarray(labels_all, dtype=np.int64),
                        folds=np.asarray(fold_all, dtype=np.int64),
                        probabilities=np.concatenate(probs_all, axis=0),
                        class_names=np.asarray(CLASSES, dtype=str))
    with (output_dir / f"oof_{args.model}_original_folds_report.json").open("w", encoding="utf-8") as f:
        json.dump({"model": args.model, "sklearn_version": SKLEARN_VERSION,
                   "torch_version": torch.__version__, "fold_reports": fold_reports,
                   "total_images": len(ids_all), "npz": str(npz_path)}, f, indent=2)
    print(f"Saved {npz_path} ({len(ids_all)} OOF predictions)")


if __name__ == "__main__":
    main()



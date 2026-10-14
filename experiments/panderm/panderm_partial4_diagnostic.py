"""Audit PanDerm partial4 Fold 1 evaluation against the saved EMA checkpoint.

Run from the Mole Classification project root with the existing environment:
    python panderm_partial4_diagnostic.py

Does not retrain, edit checkpoints or modify existing reports.
Compares three evaluation pathways on the same fold and the same images.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset
from experiments.panderm.panderm_cv_train import (
    DEVICE, FocalLoss, build_panderm_model,
    run_validation_epoch_with_macro_f1,
)
from trained_model_evaluation import build_evaluation_report, calculate_overall_accuracy

FOLD = 1
CHECKPOINT = Path(f"checkpoints/panderm_base_partial4_ema_seed123_fold{FOLD}_best.pt")
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def predict_as_in_ensemble(model, val_loader):
    all_labels, all_predictions = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            all_predictions.extend(probs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.asarray(all_labels), np.asarray(all_predictions)


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    train_meta, val_meta = get_fold_metadata(FOLD)
    train_lesions = set(train_meta["lesion_id"])
    val_lesions = set(val_meta["lesion_id"])
    print("Fold:", FOLD)
    print("Train samples:", len(train_meta), "Validation samples:", len(val_meta))
    print("Lesion ID overlap:", len(train_lesions & val_lesions))

    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    val_dataset = CrossValidationDataset(val_meta, "data/processed", transform=transform)
    if list(val_dataset.classes) != CLASSES:
        raise ValueError(f"Class-order mismatch: {val_dataset.classes} != {CLASSES}")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    print("Classes:", val_dataset.classes)
    print("Validation label counts:", {name: val_dataset.targets.count(i) for i, name in enumerate(CLASSES)})
    print("Transform:", transform)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    print("Checkpoint:", CHECKPOINT)
    print("Saved epoch:", checkpoint.get("epoch"))
    print("Saved validation metrics:", checkpoint.get("metrics"))

    model = build_panderm_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE).eval()
    print("EMA checkpoint loaded with strict=True")

    # Path A: exactly the validation function used inside panderm_cv_train.py.
    val_loss, acc_a, f1_a = run_validation_epoch_with_macro_f1(
        model, val_loader, FocalLoss(gamma=2.0), DEVICE
    )
    print(f"A / training validation: val_loss={val_loss:.6f}, accuracy={acc_a:.6f}, macro_f1={f1_a:.6f}")

    # Path B: original independent evaluation used in panderm_cv_evaluation.py.
    report = build_evaluation_report(model, val_loader, val_dataset.classes, len(CLASSES))
    acc_b = calculate_overall_accuracy(report["confusion_matrix"])
    f1_b = sum(c["f1"] for c in report["class_metrics"].values()) / len(CLASSES)
    print(f"B / existing evaluation: accuracy={acc_b:.6f}, macro_f1={f1_b:.6f}")

    # Path C: exactly the autocast + softmax + argmax logic in the ensemble.
    labels, predictions = predict_as_in_ensemble(model, val_loader)
    if not np.array_equal(labels, np.asarray(val_dataset.targets)):
        raise AssertionError("DataLoader labels differ from dataset.targets")
    acc_c = accuracy_score(labels, predictions)
    f1_c = f1_score(labels, predictions, labels=list(range(7)), average="macro", zero_division=0)
    print(f"C / ensemble prediction: accuracy={acc_c:.6f}, macro_f1={f1_c:.6f}")

    print("A vs B F1 difference:", round(f1_a - f1_b, 6))
    print("B vs C F1 difference:", round(f1_b - f1_c, 6))
    print("Recorded vs A F1 difference:", round(f1_a - checkpoint["metrics"]["val_macro_f1"], 6))
    print("Recorded vs A accuracy difference:", round(acc_a - checkpoint["metrics"]["val_accuracy"], 6))
    print("No files were changed.")


if __name__ == "__main__":
    main()

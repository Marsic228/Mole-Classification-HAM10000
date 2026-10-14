import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from core.cross_validation_split import get_fold_metadata
from experiments.metadata.metadata_features import MetadataFeatureEncoder
from experiments.metadata.convnext_tiny_metadata_fusion import FrozenImageMetadataResidual


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

BATCH_SIZE = 256


def align_metadata(metadata, image_ids):
    return (
        metadata
        .set_index("image_id")
        .loc[image_ids]
        .reset_index()
    )


def build_fold_validation_data(fold_number):
    cache = torch.load(
        f"cache/convnext_v13_features_fold{fold_number}.pt",
        map_location="cpu",
    )

    train_metadata, val_metadata = get_fold_metadata(
        fold_number
    )

    train_metadata = align_metadata(
        train_metadata,
        cache["train_image_ids"],
    )

    val_metadata = align_metadata(
        val_metadata,
        cache["val_image_ids"],
    )

    encoder = MetadataFeatureEncoder()
    encoder.fit(train_metadata)

    val_metadata_features = encoder.transform(
        val_metadata
    )

    dataset = TensorDataset(
        cache["val_image_features"].float(),
        val_metadata_features,
        cache["val_labels"],
    )

    return dataset, encoder


def load_v16c_model(fold_number, encoder):
    checkpoint_path = (
        "checkpoints/"
        f"convnext_tiny_v16c_metadata_residual_fold{fold_number}_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Verify that evaluation preprocessing matches training.
    assert np.isclose(
        encoder.age_median,
        checkpoint["age_median"],
    )
    assert np.isclose(
        encoder.age_mean,
        checkpoint["age_mean"],
    )
    assert np.isclose(
        encoder.age_std,
        checkpoint["age_std"],
    )

    model = FrozenImageMetadataResidual(
        image_dim=768,
        metadata_dim=20,
        num_classes=7,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


def collect_predictions(model, loader):
    all_labels = []
    all_v13_predictions = []
    all_v16c_predictions = []

    with torch.inference_mode():
        for (
            image_features,
            metadata_features,
            labels,
        ) in loader:
            image_features = image_features.to(DEVICE)
            metadata_features = metadata_features.to(DEVICE)

            # Baseline: frozen v13 image classifier only.
            v13_logits = model.image_classifier(
                image_features
            )

            # v16c: same v13 logits + metadata residual.
            v16c_logits = model(
                image_features,
                metadata_features,
            )

            all_labels.extend(
                labels.tolist()
            )

            all_v13_predictions.extend(
                v13_logits.argmax(dim=1).cpu().tolist()
            )

            all_v16c_predictions.extend(
                v16c_logits.argmax(dim=1).cpu().tolist()
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_v13_predictions),
        np.asarray(all_v16c_predictions),
    )


def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(7)),
            average="macro",
            zero_division=0,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=list(range(7)),
            average=None,
            zero_division=0,
        )
    )

    per_class = {}

    for index, class_name in enumerate(CLASS_NAMES):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(7)),
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_fold(fold_number):
    print(f"\n########## FOLD {fold_number} ##########")

    dataset, encoder = build_fold_validation_data(
        fold_number
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model, checkpoint = load_v16c_model(
        fold_number,
        encoder,
    )

    labels, v13_predictions, v16c_predictions = (
        collect_predictions(
            model,
            loader,
        )
    )

    v13_metrics = calculate_metrics(
        labels,
        v13_predictions,
    )

    v16c_metrics = calculate_metrics(
        labels,
        v16c_predictions,
    )

    print(
        "v13  | "
        f"acc={v13_metrics['accuracy']:.4f} | "
        f"macro_P={v13_metrics['macro_precision']:.4f} | "
        f"macro_R={v13_metrics['macro_recall']:.4f} | "
        f"macro_F1={v13_metrics['macro_f1']:.4f}"
    )

    print(
        "v16c | "
        f"acc={v16c_metrics['accuracy']:.4f} | "
        f"macro_P={v16c_metrics['macro_precision']:.4f} | "
        f"macro_R={v16c_metrics['macro_recall']:.4f} | "
        f"macro_F1={v16c_metrics['macro_f1']:.4f}"
    )

    print("Per-class F1 delta (v16c - v13):")

    per_class_delta = {}

    for class_name in CLASS_NAMES:
        baseline_f1 = v13_metrics["per_class"][class_name]["f1"]
        fusion_f1 = v16c_metrics["per_class"][class_name]["f1"]
        delta = fusion_f1 - baseline_f1

        per_class_delta[class_name] = float(delta)

        print(
            f"  {class_name:5s}: "
            f"{baseline_f1:.4f} -> {fusion_f1:.4f} "
            f"({delta:+.4f})"
        )

    # The evaluation should reproduce the checkpoint-selected score.
    assert np.isclose(
        v16c_metrics["macro_f1"],
        checkpoint["best_macro_f1"],
        atol=1e-8,
    ), (
        f"Fold {fold_number}: evaluation macro F1 "
        f"{v16c_metrics['macro_f1']:.10f} does not match "
        f"checkpoint {checkpoint['best_macro_f1']:.10f}"
    )

    return {
        "fold": fold_number,
        "best_epoch": int(checkpoint["best_epoch"]),
        "v13": v13_metrics,
        "v16c": v16c_metrics,
        "per_class_f1_delta": per_class_delta,
    }


def mean_std(values):
    return mean(values), stdev(values)


def summarize_results(fold_results):
    summary = {
        "v13": {},
        "v16c": {},
        "delta": {},
        "per_class": {},
    }

    metric_names = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]

    for metric_name in metric_names:
        v13_values = [
            result["v13"][metric_name]
            for result in fold_results
        ]

        v16c_values = [
            result["v16c"][metric_name]
            for result in fold_results
        ]

        delta_values = [
            v16c - v13
            for v13, v16c in zip(
                v13_values,
                v16c_values,
            )
        ]

        v13_mean, v13_std = mean_std(v13_values)
        v16c_mean, v16c_std = mean_std(v16c_values)
        delta_mean, delta_std = mean_std(delta_values)

        summary["v13"][metric_name] = {
            "mean": v13_mean,
            "std": v13_std,
        }

        summary["v16c"][metric_name] = {
            "mean": v16c_mean,
            "std": v16c_std,
        }

        summary["delta"][metric_name] = {
            "mean": delta_mean,
            "std": delta_std,
        }

    for class_name in CLASS_NAMES:
        v13_f1_values = [
            result["v13"]["per_class"][class_name]["f1"]
            for result in fold_results
        ]

        v16c_f1_values = [
            result["v16c"]["per_class"][class_name]["f1"]
            for result in fold_results
        ]

        delta_values = [
            v16c - v13
            for v13, v16c in zip(
                v13_f1_values,
                v16c_f1_values,
            )
        ]

        v13_mean, v13_std = mean_std(v13_f1_values)
        v16c_mean, v16c_std = mean_std(v16c_f1_values)
        delta_mean, delta_std = mean_std(delta_values)

        summary["per_class"][class_name] = {
            "v13_f1_mean": v13_mean,
            "v13_f1_std": v13_std,
            "v16c_f1_mean": v16c_mean,
            "v16c_f1_std": v16c_std,
            "delta_f1_mean": delta_mean,
            "delta_f1_std": delta_std,
        }

    return summary


def print_summary(summary):
    print("\n===== v16c EVALUATION SUMMARY =====")

    labels = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1",
    }

    for metric_name, label in labels.items():
        v13 = summary["v13"][metric_name]
        v16c = summary["v16c"][metric_name]
        delta = summary["delta"][metric_name]

        print(
            f"{label:16s} | "
            f"v13 {v13['mean']:.4f} ± {v13['std']:.4f} | "
            f"v16c {v16c['mean']:.4f} ± {v16c['std']:.4f} | "
            f"delta {delta['mean']:+.4f} ± {delta['std']:.4f}"
        )

    print("\n===== PER-CLASS F1 =====")

    for class_name in CLASS_NAMES:
        metrics = summary["per_class"][class_name]

        print(
            f"{class_name:5s} | "
            f"v13 {metrics['v13_f1_mean']:.4f} ± "
            f"{metrics['v13_f1_std']:.4f} | "
            f"v16c {metrics['v16c_f1_mean']:.4f} ± "
            f"{metrics['v16c_f1_std']:.4f} | "
            f"delta {metrics['delta_f1_mean']:+.4f} ± "
            f"{metrics['delta_f1_std']:.4f}"
        )


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    fold_results = []

    for fold_number in range(1, 6):
        result = evaluate_fold(
            fold_number
        )
        fold_results.append(result)

    summary = summarize_results(
        fold_results
    )

    print_summary(summary)

    report = {
        "experiment": "v16c_frozen_image_metadata_residual_evaluation",
        "class_names": CLASS_NAMES,
        "folds": fold_results,
        "summary": summary,
    }

    report_dir = Path("reports")
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir
        / "convnext_tiny_v16c_metadata_residual_evaluation.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nSaved report:", report_path)

"""Paired 5-fold development evaluation: natural CE vs Balanced Softmax.

Run from the project/repository root:
    python -m evaluation.convnext_natural_ce_vs_balsoftmax

This uses the same canonical fold metadata, input transform, class order and
RAW model logits as the two training experiments. It evaluates each fold's
selected-best EMA checkpoint on its development validation fold; it is NOT
an independent or external test.

Creates:
  reports/<run_name>_fold<k>_evaluation.json       (10 per-fold reports)
  reports/convnext_natural_ce_vs_balsoftmax_summary.json
"""

import json
import statistics
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import CrossValidationDataset
from evaluation.trained_model_evaluation import (
    build_evaluation_report,
    calculate_overall_accuracy,
    save_evaluation_report,
)


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
RUNS = {
    "ce": "convnext_tiny_natural_ce_seed123",
    "balanced_softmax": "convnext_tiny_natural_balsoftmax_seed123",
}
FOLDS = range(1, 6)
BATCH_SIZE = 8
CHECKPOINT_DIR = Path("checkpoints")
REPORT_DIR = Path("reports")
SUMMARY_PATH = REPORT_DIR / "convnext_natural_ce_vs_balsoftmax_summary.json"
REPLAY_TOLERANCE = 0.003  # Same tolerance as the project's checkpoint replay audit.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(run_name, fold):
    return CHECKPOINT_DIR / f"{run_name}_fold{fold}_best.pt"


def best_summary_path(run_name, fold):
    return REPORT_DIR / f"{run_name}_fold{fold}_best_summary.json"


def load_model(path):
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, len(CLASS_NAMES))

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    return model


def macro_metrics(class_metrics):
    if set(class_metrics) != set(CLASS_NAMES):
        raise ValueError(
            f"Class metrics do not match expected order/classes: {list(class_metrics)}"
        )
    return {
        f"macro_{metric}": sum(float(class_metrics[name][metric]) for name in CLASS_NAMES)
        / len(CLASS_NAMES)
        for metric in ("precision", "recall", "f1")
    }


def describe(values):
    """Mean and sample SD of five folds, not pooled OOF metrics."""
    vals = [float(x) for x in values]
    return {
        "mean": statistics.mean(vals),
        "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def preflight():
    missing = [
        str(checkpoint_path(run_name, fold))
        for run_name in RUNS.values()
        for fold in FOLDS
        if not checkpoint_path(run_name, fold).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required checkpoints are missing; no evaluation was started:\n"
            + "\n".join(missing)
        )


def check_replay(run_name, fold, observed_macro_f1):
    """Compare against saved best-epoch score, when the summary is available."""
    path = best_summary_path(run_name, fold)
    if not path.is_file():
        print(f"  WARNING: no training summary found at {path}; replay unverified")
        return None

    with path.open("r", encoding="utf-8") as handle:
        training_summary = json.load(handle)
    expected = float(training_summary["best_macro_f1"])
    difference = abs(observed_macro_f1 - expected)
    if difference > REPLAY_TOLERANCE:
        raise RuntimeError(
            f"Fold {fold}, {run_name}: replay macro F1={observed_macro_f1:.6f} "
            f"differs from training best={expected:.6f} by {difference:.6f}; "
            "check fold identity, preprocessing, class order and checkpoint."
        )
    print(f"  Replay check PASS: |macro F1 difference|={difference:.6f}")
    return {"training_best_macro_f1": expected, "absolute_difference": difference}


def evaluate_fold(fold, transform):
    _, val_metadata = get_fold_metadata(fold)
    val_dataset = CrossValidationDataset(
        val_metadata, "data/processed", transform=transform
    )
    actual_names = tuple(val_dataset.classes)
    if actual_names != CLASS_NAMES:
        raise ValueError(
            f"Fold {fold}: dataset class order {actual_names} != {CLASS_NAMES}"
        )

    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    results = {}

    for label, run_name in RUNS.items():
        print(f"\nFold {fold} | {label} | {run_name}")
        model = load_model(checkpoint_path(run_name, fold))
        try:
            # The saved Balanced Softmax model is used with unadjusted/raw logits.
            # No class-frequency adjustment is applied during evaluation.
            with torch.inference_mode():
                report = build_evaluation_report(
                    model, loader, list(CLASS_NAMES), len(CLASS_NAMES)
                )
        finally:
            del model

        report["overall_accuracy"] = float(
            calculate_overall_accuracy(report["confusion_matrix"])
        )
        report.update(macro_metrics(report["class_metrics"]))

        replay = check_replay(run_name, fold, float(report["macro_f1"]))
        if replay is not None:
            report["training_replay_check"] = replay

        path = REPORT_DIR / f"{run_name}_fold{fold}_evaluation.json"
        save_evaluation_report(report, str(path))
        results[label] = report

        print(
            f"  Accuracy {report['overall_accuracy']:.4f} | "
            f"macro F1 {report['macro_f1']:.4f}"
        )
        for name in CLASS_NAMES:
            metrics = report["class_metrics"][name]
            print(
                f"  {name:5} P {metrics['precision']:.4f} "
                f"R {metrics['recall']:.4f} F1 {metrics['f1']:.4f}"
            )

    return results


def summarize(all_folds):
    output = {
        "evaluation_type": "selected-best checkpoint / canonical 5-fold development validation",
        "warning": "Not external test. Fold means and SDs are not pooled OOF scores.",
        "class_names": list(CLASS_NAMES),
        "folds": {},
        "aggregate": {},
        "per_class": {},
    }
    for fold, reports in all_folds.items():
        ce, bs = reports["ce"], reports["balanced_softmax"]
        output["folds"][str(fold)] = {
            "ce_macro_f1": float(ce["macro_f1"]),
            "balanced_softmax_macro_f1": float(bs["macro_f1"]),
            "delta_balanced_softmax_minus_ce": float(bs["macro_f1"] - ce["macro_f1"]),
        }

    metrics = ("overall_accuracy", "macro_precision", "macro_recall", "macro_f1")
    for metric in metrics:
        ce = [reports["ce"][metric] for reports in all_folds.values()]
        bs = [reports["balanced_softmax"][metric] for reports in all_folds.values()]
        output["aggregate"][metric] = {
            "ce": describe(ce),
            "balanced_softmax": describe(bs),
            "paired_delta_balanced_softmax_minus_ce": describe(
                [b - c for c, b in zip(ce, bs)]
            ),
        }

    for name in CLASS_NAMES:
        output["per_class"][name] = {}
        for metric in ("precision", "recall", "f1"):
            ce = [r["ce"]["class_metrics"][name][metric] for r in all_folds.values()]
            bs = [r["balanced_softmax"]["class_metrics"][name][metric] for r in all_folds.values()]
            output["per_class"][name][metric] = {
                "ce": describe(ce),
                "balanced_softmax": describe(bs),
                "paired_delta_balanced_softmax_minus_ce": describe(
                    [float(b) - float(c) for c, b in zip(ce, bs)]
                ),
            }
    return output


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(torch.cuda.get_device_name(0))

    preflight()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()

    all_folds = {}
    for fold in FOLDS:
        print(f"\n===== CANONICAL FOLD {fold} =====")
        all_folds[fold] = evaluate_fold(fold, transform)

    summary = summarize(all_folds)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n===== PAIRED 5-FOLD SUMMARY (mean ± sample SD) =====")
    for metric, methods in summary["aggregate"].items():
        ce = methods["ce"]
        bs = methods["balanced_softmax"]
        delta = methods["paired_delta_balanced_softmax_minus_ce"]
        print(
            f"{metric:18} CE {ce['mean']:.4f} ± {ce['sample_sd']:.4f} | "
            f"BS {bs['mean']:.4f} ± {bs['sample_sd']:.4f} | "
            f"Δ {delta['mean']:+.4f} ± {delta['sample_sd']:.4f}"
        )
    print("\nPer-class mean F1 (BS minus CE):")
    for name in CLASS_NAMES:
        f1 = summary["per_class"][name]["f1"]
        print(
            f"{name:5} CE {f1['ce']['mean']:.4f} | "
            f"BS {f1['balanced_softmax']['mean']:.4f} | "
            f"Δ {f1['paired_delta_balanced_softmax_minus_ce']['mean']:+.4f}"
        )
    print(f"\nSaved paired summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

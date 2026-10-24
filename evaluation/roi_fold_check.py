
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import classification_report, f1_score

from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import CrossValidationDataset
import sys

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
if FOLD not in range(1, 6):
    raise ValueError("Fold must be between 1 and 5")

RUN = "convnext_tiny_natural_ce_roi_margin15_seed123"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

checkpoint_path = Path(f"checkpoints/{RUN}_fold{FOLD}_best.pt")
summary_path = Path(f"reports/{RUN}_fold{FOLD}_best_summary.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_metadata = get_fold_metadata(FOLD)

dataset = CrossValidationDataset(
    val_metadata,
    "data/processed_roi_margin15_v1",
    transform=ConvNeXt_Tiny_Weights.DEFAULT.transforms()
)

assert dataset.classes == CLASSES

loader = DataLoader(dataset, batch_size=8, shuffle=False)

model = convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(
    model.classifier[2].in_features, 7
)

checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.to(device).eval()

predictions, labels = [], []

with torch.inference_mode():
    for images, targets in loader:
        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda")
        ):
            logits = model(images.to(device))

        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(targets.tolist())

macro_f1 = f1_score(
    labels, predictions,
    labels=list(range(7)),
    average="macro",
    zero_division=0
)

summary = json.loads(summary_path.read_text())
expected = summary["best_macro_f1"]

print("Checkpoint epoch:", checkpoint["epoch"])
print(f"Replay macro F1: {macro_f1:.6f}")
print(f"Training macro F1: {expected:.6f}")
print(f"Difference: {macro_f1 - expected:+.6f}")

print(classification_report(
    labels, predictions,
    labels=list(range(7)),
    target_names=CLASSES,
    zero_division=0,
    digits=4
))

assert abs(macro_f1 - expected) <= 0.003, \
    "Checkpoint replay mismatch"

details = classification_report(
    labels,
    predictions,
    labels=list(range(7)),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

roi_report = {
    "fold": FOLD,
    "run_name": RUN,
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "total_checked": len(labels),
    "class_names": CLASSES,
    "overall_accuracy": details["accuracy"],
    "macro_precision": details["macro avg"]["precision"],
    "macro_recall": details["macro avg"]["recall"],
    "macro_f1": details["macro avg"]["f1-score"],
    "class_metrics": {
        name: {
            "precision": details[name]["precision"],
            "recall": details[name]["recall"],
            "f1": details[name]["f1-score"],
            "support": int(details[name]["support"]),
        }
        for name in CLASSES
    },
}

report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

report_path = report_dir / f"{RUN}_fold{FOLD}_evaluation.json"

if report_path.exists():
    existing = json.loads(report_path.read_text(encoding="utf-8"))
    if existing != roi_report:
        raise RuntimeError(f"Existing report differs: {report_path}")
else:
    report_path.write_text(
        json.dumps(roi_report, indent=2),
        encoding="utf-8",
    )

print("Fold report verified:", report_path)

from statistics import mean, stdev

roi_paths = [
    report_dir / f"{RUN}_fold{fold}_evaluation.json"
    for fold in range(1, 6)
]

if all(path.is_file() for path in roi_paths):
    baseline_run = "convnext_tiny_natural_ce_seed123"

    ce_paths = [
        report_dir / f"{baseline_run}_fold{fold}_evaluation.json"
        for fold in range(1, 6)
    ]

    missing = [str(p) for p in ce_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline evaluations: " + ", ".join(missing)
        )

    roi = [json.loads(p.read_text()) for p in roi_paths]
    ce = [json.loads(p.read_text()) for p in ce_paths]

    def describe(values):
        return {
            "mean": mean(values),
            "sample_sd": stdev(values),
        }

    def compare(roi_values, ce_values):
        return {
            "roi": describe(roi_values),
            "natural_ce": describe(ce_values),
            "paired_delta": describe([
                a - b for a, b in zip(roi_values, ce_values)
            ]),
        }

    summary = {
        "evaluation_type": "canonical five-fold development CV",
        "roi_run": RUN,
        "baseline_run": baseline_run,
        "warning": "Not an independent external test",
        "folds": [],
        "aggregate": {},
        "per_class": {},
    }

    for fold, (r, b) in enumerate(zip(roi, ce), start=1):
        if r["fold"] != fold:
            raise ValueError(f"ROI fold mismatch: {fold}")
        if r["total_checked"] != b["total_checked"]:
            raise ValueError(f"Validation size mismatch: {fold}")
        if r["class_names"] != b["class_names"]:
            raise ValueError(f"Class mismatch: {fold}")

        summary["folds"].append({
            "fold": fold,
            "roi_macro_f1": r["macro_f1"],
            "ce_macro_f1": b["macro_f1"],
            "delta": r["macro_f1"] - b["macro_f1"],
        })

    for metric in (
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        summary["aggregate"][metric] = compare(
            [r[metric] for r in roi],
            [b[metric] for b in ce],
        )

    for name in CLASSES:
        summary["per_class"][name] = {
            metric: compare(
                [r["class_metrics"][name][metric] for r in roi],
                [b["class_metrics"][name][metric] for b in ce],
            )
            for metric in ("precision", "recall", "f1")
        }

    summary_path = report_dir / (
        f"{RUN}_vs_natural_ce_5fold_summary.json"
    )

    if summary_path.exists():
        existing = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        if existing != summary:
            raise RuntimeError(
                f"Existing summary differs: {summary_path}"
            )
    else:
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    result = summary["aggregate"]["macro_f1"]

    print("\nFive-fold ROI macro F1:", result["roi"])
    print("Paired improvement:", result["paired_delta"])
    print("Summary verified:", summary_path)

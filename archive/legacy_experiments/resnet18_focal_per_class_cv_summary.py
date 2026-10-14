import json
import statistics
from pathlib import Path

epoch = 2

report_paths = [
    Path(f"reports/resnet18_focal_reference_recheck_fold1_epoch_3_evaluation.json"),
    Path(f"reports/resnet18_focal_reference_recheck_fold2_epoch_3_evaluation.json"),
    Path(f"reports/resnet18_focal_reference_recheck_fold3_epoch_3_evaluation.json"),
    Path(f"reports/resnet18_focal_reference_recheck_fold4_epoch_3_evaluation.json"),
    Path(f"reports/resnet18_focal_reference_recheck_fold5_epoch_3_evaluation.json"),
]

fold_metrics = []

for path in report_paths:
    with open(path , "r", encoding="utf-8") as file:
        data = json.load(file)

    fold_metrics.append(data["class_metrics"])


class_names = fold_metrics[0].keys()

per_class_values = {}

for class_name in class_names:
    precision_values = []
    recall_values = []
    f1_values = []

    for fold in fold_metrics:
        class_metrics = fold[class_name]

        precision_values.append(class_metrics["precision"])
        recall_values.append(class_metrics["recall"])
        f1_values.append(class_metrics["f1"])

    per_class_values[class_name] = {
        "precision": precision_values,
        "recall": recall_values,
        "f1": f1_values
    }


per_class_summary = {}

for class_name, metrics in per_class_values.items():
    per_class_summary[class_name] = {
        "precision_mean": statistics.mean(metrics["precision"]),
        "precision_sd": statistics.stdev(metrics["precision"]),

        "recall_mean": statistics.mean(metrics["recall"]),
        "recall_sd": statistics.stdev(metrics["recall"]),

        "f1_mean": statistics.mean(metrics["f1"]),
        "f1_sd": statistics.stdev(metrics["f1"])
    }


output_path = Path(
    f"reports/resnet18_focal_reference_recheck_per_class_cv_summary.json"
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(per_class_summary, file, indent=2)

print(f"Saved to: {output_path}")


for class_name, metrics in per_class_summary.items():
    print(
        f"{class_name:6} | "
        f"P {metrics['precision_mean']:.4f} ± {metrics['precision_sd']:.4f} | "
        f"R {metrics['recall_mean']:.4f} ± {metrics['recall_sd']:.4f} | "
        f"F1 {metrics['f1_mean']:.4f} ± {metrics['f1_sd']:.4f}"
    )
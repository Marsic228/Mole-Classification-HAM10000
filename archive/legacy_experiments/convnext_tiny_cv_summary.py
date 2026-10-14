import json
from statistics import mean, stdev

from trained_model_evaluation import save_evaluation_report


fold_reports = []

for fold_number in range(1, 6):
    path = (
        f"reports/"
        f"convnext_tiny_finetune_v12_cutmix_p025_"
        f"fold{fold_number}_evaluation.json"
    )

    with open(path, "r", encoding="utf-8") as file:
        fold_reports.append(json.load(file))


# -----------------------------
# Overall CV metrics
# -----------------------------

metric_names = [
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]

cv_summary = {}

for metric_name in metric_names:
    values = [
        report[metric_name]
        for report in fold_reports
    ]

    cv_summary[metric_name] = {
        "mean": mean(values),
        "std": stdev(values),
    }

save_evaluation_report(
    cv_summary,
    "reports/convnext_tiny_finetune_v12_cutmix_p025_cv_summary.json"
)

print("\n===== CV SUMMARY =====")

for metric_name, result in cv_summary.items():
    print(
        f"{metric_name}: "
        f"{result['mean']:.4f} ± "
        f"{result['std']:.4f}"
    )


# -----------------------------
# Per-class CV metrics
# -----------------------------

class_names = fold_reports[0]["class_metrics"].keys()

per_class_summary = {}

for class_name in class_names:
    per_class_summary[class_name] = {}

    for metric_name in [
        "precision",
        "recall",
        "f1",
    ]:
        values = [
            report["class_metrics"][class_name][metric_name]
            for report in fold_reports
        ]

        per_class_summary[class_name][metric_name] = {
            "mean": mean(values),
            "std": stdev(values),
        }

save_evaluation_report(
    per_class_summary,
    "reports/convnext_tiny_finetune_v12_cutmix_p025_per_class_cv_summary.json"
)

print("\nPer-class:")

for class_name, metrics in per_class_summary.items():
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = metrics["f1"]

    print(
        f"{class_name:6s} | "
        f"P {p['mean']:.4f} ± {p['std']:.4f} | "
        f"R {r['mean']:.4f} ± {r['std']:.4f} | "
        f"F1 {f1['mean']:.4f} ± {f1['std']:.4f}"
    )
from pathlib import Path
import json

def load_json(path):
    path = Path(path)
    with open(path, "r") as file:
        loaded = json.load(file)
    return loaded

def build_evaluation_summary(report):
    max_batches = report["max_batches"]
    total_checked = report["total_checked"]
    class_metrics = report["class_metrics"]

    lines = []

    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append(f"Max batches: {max_batches}")
    lines.append(f"Total checked validation images: {total_checked}")
    lines.append("")

    lines.append("## Class-by-class metrics")
    lines.append("")

    for class_name, metrics in class_metrics.items():
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        lines.append(f"- {class_name}: precision={precision}, recall={recall}, f1={f1}")

    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("This is a limited metric pipeline check.")
    lines.append("The baseline is not fully trained.")
    lines.append("Current class metrics are not final model quality.")

    return "\n".join(lines)

def save_text(text, path):
    path = Path(path)
    with open(path, "w") as file:
        file.write(text)

def main():
    report = load_json("class_metrics_report.json")
    text = build_evaluation_summary(report)
    save_text(text, "evaluation_summary.md")
    print(text)
if __name__ == "__main__":
    main()
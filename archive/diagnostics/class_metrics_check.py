from pathlib import Path
import json

from metrics import calculate_class_metrics


def load_confusion_matrix_report(path):
    path = Path(path)
    with open(path, "r") as file:
        loaded = json.load(file)
    return loaded


def save_class_metrics_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)



def main():
    confusion_matrix = load_confusion_matrix_report("confusion_matrix_report.json")

    matrix = confusion_matrix["matrix"]
    class_names = confusion_matrix["class_names"]

    class_metrics = calculate_class_metrics(matrix, class_names)

    class_metrics_report = {
        "max_batches": confusion_matrix["max_batches"],
        "total_checked": confusion_matrix["total_checked"],
        "class_names": class_names,
        "class_metrics": class_metrics,
    }

    save_class_metrics_report(class_metrics_report, "class_metrics_report.json")

    print("Saved class_metrics_report.json")
    print(class_metrics)


if __name__ == "__main__":
    main()
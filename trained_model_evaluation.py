import json
from pathlib import Path

import torch
from torchvision import transforms

from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders
from archive.diagnostics.baseline_cnn_forward_check import build_baseline_cnn
from evaluation.confusion_matrix_check import collect_validation_predictions, build_confusion_matrix
from evaluation.metrics import calculate_class_metrics

def load_trained_model(checkpoint_path, num_classes):
   checkpoint_path = Path(checkpoint_path)
   model = build_baseline_cnn(num_classes)
   checkpoint = torch.load(checkpoint_path)
   model.load_state_dict(checkpoint["model_state_dict"])
   model.eval()
   return model

def build_evaluation_report(model, val_loader, class_names, num_classes):
    all_predictions, all_labels = collect_validation_predictions(
        model,
        val_loader,
        max_batches=None,
    )

    matrix = build_confusion_matrix(all_predictions, all_labels, num_classes)

    class_metrics = calculate_class_metrics(matrix, class_names)

    report = {
        "class_names": class_names,
        "total_checked": len(all_predictions),
        "confusion_matrix": matrix,
        "class_metrics": class_metrics,
    }

    return report

def save_evaluation_report(data, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def calculate_overall_accuracy(matrix):
    correct = 0

    for class_index in range(len(matrix)):
        correct += matrix[class_index][class_index]

    total_samples = 0
    for row in matrix:
        total_samples += sum(row)
    if total_samples == 0:
        return 0.0

    return correct / total_samples

def calculate_majority_class_baseline(matrix):
    total_samples = 0
    largest_row_sum = 0

    for row in matrix:
        row_sum = sum(row)

        total_samples += row_sum

        if row_sum > largest_row_sum:
            largest_row_sum = row_sum

    if total_samples == 0:
        return 0.0

    return largest_row_sum / total_samples

def main():
    tensor_transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    class_names = datasets["val"].classes
    num_classes = len(class_names)

    trained_model = load_trained_model("checkpoints/epoch_3.pt", num_classes)

    evaluation_report = build_evaluation_report(
        trained_model,
        loaders["val"],
        class_names,
        num_classes,
    )

    overall_accuracy = calculate_overall_accuracy(
        evaluation_report["confusion_matrix"]
    )

    majority_baseline_accuracy = calculate_majority_class_baseline(
        evaluation_report["confusion_matrix"]
    )

    evaluation_report["overall_accuracy"] = overall_accuracy
    evaluation_report["majority_class_baseline_accuracy"] = majority_baseline_accuracy
    evaluation_report["accuracy_above_majority_baseline"] = (
        overall_accuracy - majority_baseline_accuracy
    )

    save_evaluation_report(
        evaluation_report,
        "reports/trained_model_evaluation_report.json",
    )

    print("Saved trained model evaluation report.")
    print("Total checked:", evaluation_report["total_checked"])
    print("Overall accuracy:", evaluation_report["overall_accuracy"])
    print(
        "Majority-class baseline accuracy:",
        evaluation_report["majority_class_baseline_accuracy"],
    )
    print(
        "Accuracy above majority-class baseline:",
        evaluation_report["accuracy_above_majority_baseline"],
    )
if __name__ == "__main__":
    main()

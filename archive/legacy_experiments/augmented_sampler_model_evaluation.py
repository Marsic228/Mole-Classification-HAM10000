import json
from pathlib import Path

import torch
from dataset_loader_check import (
    build_transforms,
    build_split_datasets,
    build_split_loaders,
)

from baseline_cnn_forward_check import build_baseline_cnn
from confusion_matrix_check import collect_validation_predictions, build_confusion_matrix
from metrics import calculate_class_metrics
from baseline_experiment import calculate_majority_class_baseline

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_trained_model(checkpoint_path, num_classes):
    checkpoint_path = Path(checkpoint_path)
    model = build_baseline_cnn(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def calculate_overall_accuracy(matrix):
    correct_predictions = 0
    for class_index in range(len(matrix)):
        correct_predictions += matrix[class_index][class_index]

    total_samples = 0
    for row in matrix:
        row_sum = sum(row)
        total_samples += row_sum

    if total_samples == 0:
        return 0.0

    return correct_predictions / total_samples



def build_evaluation_report(model, val_loader, class_names, num_classes):
    all_predictions, all_labels = collect_validation_predictions(
        model,
        val_loader,
        max_batches=None,
    )
    matrix = build_confusion_matrix(all_predictions, all_labels, num_classes)
    class_metrics = calculate_class_metrics(matrix, class_names)
    overall_accuracy = calculate_overall_accuracy(matrix)
    majority_class_baseline_accuracy = calculate_majority_class_baseline(matrix)
    report = {
        "total_checked": len(all_labels),
        "class_names": class_names,
        "confusion_matrix": matrix,
        "class_metrics": class_metrics,
        "overall_accuracy": overall_accuracy,
        "majority_class_baseline_accuracy": majority_class_baseline_accuracy,
        "accuracy_above_majority_baseline": overall_accuracy - majority_class_baseline_accuracy,
    }
    return report


def main():
    transforms_dict = build_transforms()

    datasets = build_split_datasets("data/processed", transforms_dict)
    loaders = build_split_loaders(datasets, batch_size=8)

    class_names = datasets["val"].classes
    num_classes = len(class_names)

    model = load_trained_model(
        "checkpoints/augmented_sampler_epoch_2.pt",
        num_classes=num_classes,
    )

    report = build_evaluation_report(
        model,
        loaders["val"],
        class_names,
        num_classes,
    )

    save_json(report, "reports/augmented_sampler_epoch_2_evaluation_report.json")

    print("Augmented sampler model evaluation:")
    print("Total checked:", report["total_checked"])
    print("Overall accuracy:", report["overall_accuracy"])
    print("Majority baseline:", report["majority_class_baseline_accuracy"])
    print("Accuracy above majority baseline:", report["accuracy_above_majority_baseline"])
    print("Class metrics:")
    print(report["class_metrics"])


if __name__ == "__main__":
    main()
import json
from pathlib import Path

from torchvision import models
from torch import nn

import torch
from dataset_loader_check import build_split_datasets, build_split_loaders
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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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



def build_evaluation_report(model, data_loader, class_names, num_classes):
    all_predictions, all_labels = collect_validation_predictions(
        model,
        data_loader,
        max_batches=None,
    )
    matrix = build_confusion_matrix(all_predictions, all_labels, num_classes)
    class_metrics = calculate_class_metrics(matrix, class_names)
    macro_precision = sum(metrics["precision"] for metrics in class_metrics.values()) / len(class_metrics)
    macro_recall = sum(metrics["recall"] for metrics in class_metrics.values()) / len(class_metrics)
    macro_f1 = sum(metrics["f1"] for metrics in class_metrics.values()) / len(class_metrics)
    overall_accuracy = calculate_overall_accuracy(matrix)
    majority_class_baseline_accuracy = calculate_majority_class_baseline(matrix)
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            loss = loss_function(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)


    average_loss = total_loss / total_samples
    report = {
        "total_checked": len(all_labels),
        "class_names": class_names,
        "confusion_matrix": matrix,
        "class_metrics": class_metrics,
        "overall_accuracy": overall_accuracy,
        "test_loss": average_loss,
        "majority_class_baseline_accuracy": majority_class_baseline_accuracy,
        "accuracy_above_majority_baseline": overall_accuracy - majority_class_baseline_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    return report


def main():
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()
    resnet_transforms = {
        "train": transform,
        "special_train": None,
        "val": transform,
        "test": transform,
    }

    datasets = build_split_datasets("data/processed", resnet_transforms)
    loaders = build_split_loaders(datasets, batch_size=8)

    class_names = datasets["test"].classes
    num_classes = len(class_names)

    model = load_trained_model(
        "checkpoints/resnet18_layer3_layer4_epoch_3.pt",
        num_classes=num_classes,
    )

    print("Test samples:", len(datasets["test"]))
    print("Classes:", class_names)
    
    report = build_evaluation_report(
        model,
        loaders["test"],
        class_names,
        num_classes,
    )

    save_json(report, "reports/resnet18_layer3_layer4_epoch_3_test_report.json")

    print("ResNet18 model evaluation:")
    print("Total checked:", report["total_checked"])
    print("Overall accuracy:", report["overall_accuracy"])
    print("Majority baseline:", report["majority_class_baseline_accuracy"])
    print("Accuracy above majority baseline:", report["accuracy_above_majority_baseline"])
    print("Class metrics:")
    print(report["class_metrics"])
    print("Macro precision:", report["macro_precision"])
    print("Macro recall:", report["macro_recall"])
    print("Macro F1:", report["macro_f1"])
    print("Test loss:", report["test_loss"])

if __name__ == "__main__":
    main()
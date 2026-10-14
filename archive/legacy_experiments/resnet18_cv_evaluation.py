import torch
import torch.nn as nn
from statistics import mean, stdev

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from trained_model_evaluation import (
    build_evaluation_report,
    save_evaluation_report,
    calculate_overall_accuracy,
    calculate_majority_class_baseline,
)

def load_resnet_model(checkpoint_path, num_classes):
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

def calculate_macro_metrics(class_metrics):
    num_classes = len(class_metrics)

    macro_precision = sum(
        metrics["precision"]
        for metrics in class_metrics.values()
    ) / num_classes

    macro_recall = sum(
        metrics["recall"]
        for metrics in class_metrics.values()
    ) / num_classes

    macro_f1 = sum(
        metrics["f1"]
        for metrics in class_metrics.values()
    ) / num_classes

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

for epoch in range(1, 4):
    epoch_metrics = {
        "accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }

    print(f"\n========== EPOCH {epoch} ==========")

    for fold_number in range(1, 6):
        _, val_metadata = get_fold_metadata(fold_number)

        val_dataset = CrossValidationDataset(
            val_metadata,
            "data/processed",
            transform=transform
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False
        )

        class_names = val_dataset.classes
        num_classes = len(class_names)

        model = load_resnet_model(
            f"checkpoints/resnet18_fold{fold_number}_epoch_{epoch}.pt",
            num_classes
        )

        report = build_evaluation_report(
            model,
            val_loader,
            class_names,
            num_classes
        )

        matrix = report["confusion_matrix"]

        report["overall_accuracy"] = calculate_overall_accuracy(matrix)

        macro_metrics = calculate_macro_metrics(
            report["class_metrics"]
        )

        report.update(macro_metrics)

        save_evaluation_report(
            report,
            f"reports/resnet18_fold{fold_number}_epoch_{epoch}_evaluation.json"
        )

        epoch_metrics["accuracy"].append(
            report["overall_accuracy"]
        )

        epoch_metrics["macro_precision"].append(
            report["macro_precision"]
        )

        epoch_metrics["macro_recall"].append(
            report["macro_recall"]
        )

        epoch_metrics["macro_f1"].append(
            report["macro_f1"]
        )

        print(
            f"Fold {fold_number}: "
            f"accuracy={report['overall_accuracy']:.4f}, "
            f"macro_f1={report['macro_f1']:.4f}"
        )

    summary = {}

    for metric_name, values in epoch_metrics.items():
        summary[metric_name] = {
            "mean": mean(values),
            "std": stdev(values),
        }

    print(f"\nEpoch {epoch} summary:")

    for metric_name, values in summary.items():
        print(
            f"{metric_name}: "
            f"{values['mean']:.4f} ± {values['std']:.4f}"
        )

    save_evaluation_report(
        summary,
        f"reports/resnet18_epoch_{epoch}_cv_summary.json"
    )
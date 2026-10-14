import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from trained_model_evaluation import (
    build_evaluation_report,
    save_evaluation_report,
    calculate_overall_accuracy,
)

def load_efficientnet_model(checkpoint_path, num_classes):
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model

weights = EfficientNet_B0_Weights.DEFAULT
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
    print(f"\n===== EPOCH {epoch} =====")

    for fold_number in range(1, 6):
        _, val_metadata = get_fold_metadata(
            fold_number
        )

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

        model = load_efficientnet_model(
            f"checkpoints/efficientnet_b0_new_fold{fold_number}_epoch_{epoch}.pt",
            num_classes
        )

        report = build_evaluation_report(
            model,
            val_loader,
            class_names,
            num_classes
        )

        matrix = report["confusion_matrix"]

        report["overall_accuracy"] = (
            calculate_overall_accuracy(matrix)
        )

        macro_metrics = calculate_macro_metrics(
            report["class_metrics"]
        )

        report.update(macro_metrics)

        save_evaluation_report(
            report,
            f"reports/efficientnet_b0_new_fold{fold_number}_epoch_{epoch}_evaluation.json"
        )

        print(f"\nFold {fold_number}")
        print("Accuracy:", report["overall_accuracy"])
        print("Macro precision:", report["macro_precision"])
        print("Macro recall:", report["macro_recall"])
        print("Macro F1:", report["macro_f1"])

        print("Class metrics:")

        for class_name, metrics in report["class_metrics"].items():
            print(class_name, metrics)
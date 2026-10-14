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
)

class SpecialistCascade(nn.Module):
    def __init__(self, reference_model, specialist_model):
        super().__init__()
        self.reference_model = reference_model
        self.specialist_model = specialist_model

    def forward(self, images):
        reference_logits = self.reference_model(images)
        reference_predictions = reference_logits.argmax(dim=1)

        route_mask = (
            (reference_predictions == 2) |
            (reference_predictions == 4) |
            (reference_predictions == 5)
        )

        final_logits = reference_logits.clone()

        if route_mask.any():
            specialist_logits = self.specialist_model(
                images[route_mask]
            )

            # For routed samples, only bkl/mel/nv
            # are allowed as final predictions.
            final_logits[route_mask] = torch.finfo(
                final_logits.dtype
            ).min

            specialist_to_full = {
                0: 2,  # bkl
                1: 4,  # mel
                2: 5,  # nv
            }

            for specialist_idx, full_idx in specialist_to_full.items():
                final_logits[
                    route_mask,
                    full_idx
                ] = specialist_logits[:, specialist_idx]

        return final_logits


def load_resnet_model(checkpoint_path, num_classes):
    
    model = resnet18(weights=None)

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

REFERENCE_EPOCH = 3
SPECIALIST_EPOCH = 2

FULL_MAPPING = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

epoch_metrics = {
    "accuracy": [],
    "macro_precision": [],
    "macro_recall": [],
    "macro_f1": [],
}

for fold_number in range(1, 6):
    _, val_metadata = get_fold_metadata(fold_number)

    # IMPORTANT: no filtering here
    val_dataset = CrossValidationDataset(
        val_metadata,
        "data/processed",
        transform=transform
    )

    assert val_dataset.class_to_idx == FULL_MAPPING

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    class_names = val_dataset.classes

    reference_model = load_resnet_model(
        f"checkpoints/resnet18_focal_fold{fold_number}_epoch_{REFERENCE_EPOCH}.pt",
        num_classes=7
    )

    specialist_model = load_resnet_model(
        f"checkpoints/resnet18_focal_specialist_fold{fold_number}_epoch_{SPECIALIST_EPOCH}.pt",
        num_classes=3
    )

    model = SpecialistCascade(
        reference_model,
        specialist_model
    )

    model.eval()

    report = build_evaluation_report(
        model,
        val_loader,
        class_names,
        num_classes=7
    )

    matrix = report["confusion_matrix"]

    report["overall_accuracy"] = calculate_overall_accuracy(matrix)

    macro_metrics = calculate_macro_metrics(
        report["class_metrics"]
    )

    report.update(macro_metrics)

    save_evaluation_report(
        report,
        f"reports/resnet18_focal_specialist_cascade_fold{fold_number}_evaluation.json"
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

print("\nCascade summary:")

for metric_name, values in summary.items():
    print(
        f"{metric_name}: "
        f"{values['mean']:.4f} ± {values['std']:.4f}"
    )

save_evaluation_report(
    summary,
    "reports/resnet18_focal_specialist_cascade_cv_summary.json"
)
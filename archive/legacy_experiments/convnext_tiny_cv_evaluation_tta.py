import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from trained_model_evaluation import (
    build_evaluation_report,
    save_evaluation_report,
    calculate_overall_accuracy,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))

def load_convnext_model(checkpoint_path, num_classes):
    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
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

    model = model.to(DEVICE)
    model.eval()

    return model

class HorizontalFlipTTA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        logits_original = self.model(images)

        images_flipped = torch.flip(images, dims=[3])
        logits_flipped = self.model(images_flipped)

        return (logits_original + logits_flipped) / 2

weights = ConvNeXt_Tiny_Weights.DEFAULT
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

all_reports = []

for fold_number in range(1, 6):
    print(f"\n===== FOLD {fold_number} =====")

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
        batch_size=32,
        shuffle=False
    )

    class_names = val_dataset.classes
    num_classes = len(class_names)

    base_model = load_convnext_model(
        f"checkpoints/convnext_tiny_finetune_v11_cutmix_"
        f"fold{fold_number}_best.pt",
        num_classes
    )

    model = HorizontalFlipTTA(base_model)
    model.eval()

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
        f"reports/convnext_tiny_finetune_v11_cutmix_"
        f"fold{fold_number}_evaluation.json"
    )

    all_reports.append(report)

    print("Accuracy:", report["overall_accuracy"])
    print("Macro precision:", report["macro_precision"])
    print("Macro recall:", report["macro_recall"])
    print("Macro F1:", report["macro_f1"])

    print("Class metrics:")

    for class_name, metrics in report["class_metrics"].items():
        print(class_name, metrics)
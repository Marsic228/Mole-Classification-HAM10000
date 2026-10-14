import torch
from torchvision import transforms
from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
import json
from pathlib import Path

def collect_validation_predictions(model, val_loader, max_batches=None):
    model.eval()

    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)

            batch_predictions = logits.argmax(dim=1)

            all_predictions.extend(
                batch_predictions.cpu().tolist()
            )
            all_labels.extend(labels.tolist())

            if (
                max_batches is not None
                and len(all_predictions)
                >= max_batches * labels.size(0)
            ):
                break

    return all_predictions, all_labels

def build_confusion_matrix(all_predictions, all_labels, num_classes):
    matrix = []

    for _ in range(num_classes):
        row = [0] * num_classes
        matrix.append(row)

    for true_label, predicted_label in zip(all_labels, all_predictions):
        matrix[true_label][predicted_label] += 1

    return matrix

def print_confusion_matrix(matrix, class_names):
    for class_name, row in zip(class_names, matrix):
        print(class_name, row)

def save_confusion_matrix_report(report, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)

    all_predictions, all_labels = collect_validation_predictions(
        model, 
        loaders["val"], 
        max_batches=20
    )
    print("Total checked:", len(all_labels))
    matrix = build_confusion_matrix(all_predictions, all_labels, num_classes=7)
    class_names = datasets["val"].classes
    print_confusion_matrix(matrix, class_names)
    report = {
    "max_batches": 20,
    "total_checked": len(all_labels),
    "class_names": class_names,
    "matrix": matrix,
    }
    save_confusion_matrix_report(report, "confusion_matrix_report.json")



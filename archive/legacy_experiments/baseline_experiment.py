from datetime import datetime
from pathlib import Path
import json

from torchvision import transforms

from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
from training_step_check import build_loss_function, build_optimizer
from one_epoch_training_check import run_one_train_epoch, run_validation_epoch
from confusion_matrix_check import collect_validation_predictions, build_confusion_matrix
from metrics import calculate_class_metrics

def save_json(data, path):
    path = Path(path)
    with open(path, "w") as file:
        json.dump(data, file, indent=2)

def build_experiment_report(train_max_batches, val_max_batches, train_loss, val_loss, val_accuracy):
    created_at = datetime.now().isoformat()
    report = {
        "created_at": created_at,
        "train_max_batches": train_max_batches,
        "val_max_batches": val_max_batches,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "interpretation": "This is a limited baseline training experiment, not final model quality.",
    }

    return report

def run_baseline_experiment(train_max_batches, val_max_batches):
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, 0.001)
    
    train_loss = run_one_train_epoch(model, loaders["train"], loss_function, optimizer, max_batches=train_max_batches)
    val_loss, val_accuracy = run_validation_epoch(model, loaders["val"], loss_function, max_batches=val_max_batches)
    report = build_experiment_report(train_max_batches, val_max_batches, train_loss, val_loss, val_accuracy)
    return report, model, loaders

def calculate_majority_class_baseline(matrix):
    class_counts = []
    for matrix_row in matrix:
        class_count = sum(matrix_row)
        class_counts.append(class_count)
        
    largest_class_count = max(class_counts)
    total_samples = sum(class_counts)

    if total_samples == 0:
        return 0.0

    return largest_class_count / total_samples

def main():
    train_max_batches = 100
    val_max_batches = None
    report, model, loaders = run_baseline_experiment(train_max_batches, val_max_batches)
    all_predictions, all_labels = collect_validation_predictions(model, loaders["val"], max_batches=val_max_batches)
    matrix = build_confusion_matrix(all_predictions, all_labels, num_classes=7)
    majority_class_baseline_accuracy = calculate_majority_class_baseline(matrix)
    class_names = loaders["val"].dataset.classes
    class_metrics = calculate_class_metrics(matrix, class_names)
    report["class_names"] = class_names
    report["confusion_matrix"] = matrix
    report["majority_class_baseline_accuracy"] = majority_class_baseline_accuracy
    report["class_metrics"] = class_metrics
    report["total_validation_checked"] = len(all_labels)
    save_json(report, "baseline_experiment_report.json")
    print(report)
if __name__ == "__main__":
    main()



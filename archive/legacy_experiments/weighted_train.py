import json
from pathlib import Path

import torch
from torchvision import transforms

from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders
from archive.diagnostics.baseline_cnn_forward_check import build_baseline_cnn
from archive.diagnostics.training_step_check import build_optimizer
from archive.diagnostics.one_epoch_training_check import run_one_train_epoch, run_validation_epoch
from archive.legacy_experiments.weighted_training_check import (
    count_images_per_class,
    build_class_weights,
    build_weighted_loss_function,
    build_loss_function,
)

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, path)


def build_epoch_report(epoch, train_loss, val_loss, val_accuracy):
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    return metrics


def run_sampler_training_experiment(num_epochs, train_max_batches=None, val_max_batches=None):
    transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform)
    loaders = build_split_loaders(datasets, batch_size=8)
    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, learning_rate=0.001)

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_one_train_epoch(
            model,
            loaders["train"],
            loss_function,
            optimizer,
            max_batches=train_max_batches,
        )

        val_loss, val_accuracy = run_validation_epoch(
            model,
            loaders["val"],
            loss_function,
            max_batches=val_max_batches,
        )

        epoch_report = build_epoch_report(epoch, train_loss, val_loss, val_accuracy)
        history.append(epoch_report)

        save_json(epoch_report, f"reports/weighted_epoch_{epoch}_report.json")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/weighted_epoch_{epoch}.pt",
        )
    
    save_json(history, "reports/weighted_training_history.json")
    
    return history


def main():
    history = run_sampler_training_experiment(
        num_epochs=3,
        train_max_batches=None,
        val_max_batches=None,
    )

    print("Weighted training history:")
    print(history)


if __name__ == "__main__":
    main()
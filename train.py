from pathlib import Path
import json
import torch
from torchvision import transforms

from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders
from archive.diagnostics.baseline_cnn_forward_check import build_baseline_cnn
from archive.diagnostics.training_step_check import build_loss_function, build_optimizer, run_training_step
from archive.diagnostics.one_epoch_training_check import run_validation_epoch

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
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

def run_one_full_train_epoch(model, train_loader, loss_function, optimizer):
    model.train()
    losses = []

    for images, labels in train_loader:
        loss = run_training_step(model, images, labels, loss_function, optimizer)
        losses.append(loss)

    average_loss = sum(losses) / len(losses)
    return average_loss

def build_epoch_report(epoch, train_loss, val_loss, val_accuracy):
    result = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }

    return result

def run_training_experiment(num_epochs):
    transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, learning_rate=0.001)

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_one_full_train_epoch(
            model,
            loaders["train"],
            loss_function,
            optimizer,
        )

        val_loss, val_accuracy = run_validation_epoch(
            model,
            loaders["val"],
            loss_function,
            max_batches=None,
        )

        epoch_report = build_epoch_report(epoch, train_loss, val_loss, val_accuracy)
        history.append(epoch_report)

        save_json(epoch_report, f"reports/epoch_{epoch}_report.json")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/epoch_{epoch}.pt",
        )
    
    return history

def main():
    history = run_training_experiment(num_epochs=3)
    save_json(history, "reports/training_history.json")
    print("Training complete")
    print(history)
if __name__ == "__main__":
    main()
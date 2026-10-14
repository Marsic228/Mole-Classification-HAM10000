from pathlib import Path
import json
import torch
from torchvision import models
from torch import nn

from dataset_loader_check import build_split_datasets, build_split_loaders
from training_step_check import build_loss_function, run_training_step
from one_epoch_training_check import run_validation_epoch

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

def build_optimizer(model):
    optimizer = torch.optim.Adam([
        {
            "params": model.layer3.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.layer4.parameters(),
            "lr": 3e-5
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        }
    ])
    return optimizer

def get_optimizer_learning_rates(optimizer):
    learning_rates = []

    for group in optimizer.param_groups:
        learning_rates.append(group["lr"])

    return learning_rates

def run_one_full_train_epoch(model, train_loader, loss_function, optimizer, max_batches=None):
    model.eval()
    model.fc.train()
    model.layer4.train()
    model.layer3.train()

    losses = []

    for images, labels in train_loader:
        loss = run_training_step(model, images, labels, loss_function, optimizer)
        losses.append(loss)
        if max_batches is not None and len(losses) >= max_batches:
            break

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

def run_training_experiment(num_epochs, train_max_batches=None, val_max_batches=None):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 7)

    transform = weights.transforms()
    resnet_transforms = {
        "train": transform,
        "special_train": None,
        "val": transform,
        "test": transform,
    }

    datasets = build_split_datasets("data/processed", resnet_transforms)
    loaders = build_split_loaders(datasets, batch_size=8)
    loss_function = build_loss_function()

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)

    optimizer = build_optimizer(model)

    learning_rates = get_optimizer_learning_rates(optimizer)
    print("Optimizer learning rates:", learning_rates)

    history = []

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss = run_one_full_train_epoch(
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

        epoch_report = build_epoch_report(
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                epoch_report,
                "checkpoints/resnet18_discriminative_lr_best.pt",
            )

        history.append(epoch_report)

        save_json(epoch_report, f"reports/resnet18_discriminative_lr_epoch_{epoch}_report.json")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/resnet18_discriminative_lr_epoch_{epoch}.pt",
        )
    
    return history

def main():
    history = run_training_experiment(
        num_epochs=3,
        train_max_batches=None,
        val_max_batches=None,
    )
    
    save_json(history, "reports/resnet18_discriminative_lr_training_history.json")
    print("Training complete")
    print(history)
if __name__ == "__main__":
    main()
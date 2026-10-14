import torch
import torch.nn as nn
from training_step_check import run_training_step

import torch.nn.functional as F

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from utils import count_images_per_class, build_class_weights
from train import save_json, save_checkpoint, build_epoch_report
from one_epoch_training_check import run_validation_epoch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()

SPECIALIST_CLASSES = {"mel", "nv", "bkl"}


def filter_specialist_metadata(metadata):
    return (
        metadata[
            metadata["dx"].isin(SPECIALIST_CLASSES)
        ]
        .reset_index(drop=True)
    )

def run_resnet_train_epoch(
    model,
    train_loader,
    loss_function,
    optimizer
):
    model.eval()

    model.layer3.train()
    model.layer4.train()
    model.fc.train()

    losses = []

    for images, labels in train_loader:
        loss = run_training_step(
            model,
            images,
            labels,
            loss_function,
            optimizer
        )
        losses.append(loss)

    return sum(losses) / len(losses)

def run_fold(fold_number, num_epochs=3):
    weights = ResNet18_Weights.DEFAULT

    transform = weights.transforms()

    train_metadata, val_metadata = get_fold_metadata(fold_number)

    train_metadata = filter_specialist_metadata(train_metadata)
    val_metadata = filter_specialist_metadata(val_metadata)

    train_dataset = CrossValidationDataset(
        train_metadata,
        "data/processed",
        transform=transform
    )

    val_dataset = CrossValidationDataset(
        val_metadata,
        "data/processed",
        transform=transform
    )

    print("Train classes:", train_dataset.class_to_idx)
    print("Val classes:", val_dataset.class_to_idx)

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    print("Train class counts:", count_images_per_class(train_dataset))
    print("Val class counts:", count_images_per_class(val_dataset))

    expected_mapping = {
        "bkl": 0,
        "mel": 1,
        "nv": 2,
    }

    assert train_dataset.class_to_idx == expected_mapping
    assert val_dataset.class_to_idx == expected_mapping

    class_counts = count_images_per_class(train_dataset)
    class_weights = build_class_weights(class_counts)
    sample_weights = class_weights[train_dataset.targets]

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # Fresh model for this fold
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 3)

    loss_function = FocalLoss(gamma=2.0)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_resnet_train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer
        )

        val_loss, val_accuracy = run_validation_epoch(
            model,
            val_loader,
            loss_function,
            max_batches=None
        )

        epoch_report = build_epoch_report(
            epoch,
            train_loss,
            val_loss,
            val_accuracy
        )

        history.append(epoch_report)

        print(f"Fold {fold_number}:", epoch_report)

        save_json(
            epoch_report,
            f"reports/resnet18_focal_specialist_fold{fold_number}_epoch_{epoch}_report.json"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/resnet18_focal_specialist_fold{fold_number}_epoch_{epoch}.pt"
        )

        
    save_json(
        history,
        f"reports/resnet18_focal_specialist_fold{fold_number}_training_history.json"
    )

    return history

if __name__ == "__main__":
    for fold_number in range(1, 6):
        print(f"\n===== FOLD {fold_number} =====")
        run_fold(fold_number, num_epochs=3)

    
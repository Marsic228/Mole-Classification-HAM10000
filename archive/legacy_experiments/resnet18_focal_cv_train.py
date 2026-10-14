import torch
import torch.nn as nn
import math

import torch.nn.functional as F

from training_step_check import run_training_step
from one_epoch_training_check import run_validation_epoch

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset

from utils import count_images_per_class, build_class_weights
from train import save_json, save_checkpoint, build_epoch_report


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

HARD_CLASSES = {2, 4, 5}


def calculate_per_sample_focal_loss(
    logits,
    targets,
    gamma=2.0
):
    ce_loss = F.cross_entropy(
        logits,
        targets,
        reduction="none"
    )

    pt = torch.exp(-ce_loss)

    return ((1 - pt) ** gamma) * ce_loss


def build_online_hard_mask(
    model,
    dataset,
    hard_fraction=0.30,
    batch_size=8
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_losses = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)

            losses = calculate_per_sample_focal_loss(
                logits,
                labels,
                gamma=2.0
            )

            all_losses.extend(losses.tolist())
            all_targets.extend(labels.tolist())

    losses = torch.tensor(all_losses)
    targets = torch.tensor(all_targets)

    hard_mask = torch.zeros(
        len(dataset),
        dtype=torch.bool
    )

    for class_index in HARD_CLASSES:
        class_indices = torch.where(
            targets == class_index
        )[0]

        number_hard = max(
            1,
            math.ceil(
                len(class_indices) * hard_fraction
            )
        )

        class_losses = losses[class_indices]

        hardest_local_indices = torch.topk(
            class_losses,
            k=number_hard
        ).indices

        hardest_dataset_indices = (
            class_indices[hardest_local_indices]
        )

        hard_mask[hardest_dataset_indices] = True

    return hard_mask

def build_hard_sample_weights(
    dataset,
    hard_mask,
    hard_multiplier=1.5
):
    class_counts = count_images_per_class(dataset)
    class_weights = build_class_weights(class_counts)

    targets = torch.tensor(dataset.targets)

    base_weights = class_weights[dataset.targets]
    sample_weights = base_weights.clone()

    sample_weights[hard_mask] *= hard_multiplier

    # Preserve total sampling mass for every class.
    for class_index in range(7):
        class_mask = targets == class_index

        original_total = base_weights[class_mask].sum()
        modified_total = sample_weights[class_mask].sum()

        sample_weights[class_mask] *= (
            original_total / modified_total
        )

    return sample_weights

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

    train_metadata, val_metadata = get_fold_metadata(
        fold_number
    )

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )

    # Fresh model
    model = resnet18(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 7)

    loss_function = FocalLoss(gamma=2.0)

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=0.0001
    )

    hard_mask = None
    HARD_WEIGHT_MULTIPLIER = 1.5

    history = []

    for epoch in range(1, num_epochs + 1):

        # Epoch 1 = reference WRS
        if hard_mask is None:
            class_counts = count_images_per_class(
                train_dataset
            )

            class_weights = build_class_weights(
                class_counts
            )

            sample_weights = (
                class_weights[train_dataset.targets]
            )

        # Epochs 2-3 = hard-example-focused WRS
        else:
            sample_weights = build_hard_sample_weights(
                train_dataset,
                hard_mask,
                hard_multiplier=HARD_WEIGHT_MULTIPLIER
            )

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

        print(
            f"Fold {fold_number}:",
            epoch_report
        )

        save_json(
            epoch_report,
            f"reports/resnet18_focal_hard15_fold{fold_number}_epoch_{epoch}_report.json"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/resnet18_focal_hard15_fold{fold_number}_epoch_{epoch}.pt"
        )

        # Build sampling strategy for NEXT epoch
        if epoch < num_epochs:
            hard_mask = build_online_hard_mask(
                model,
                train_dataset,
                hard_fraction=0.30
            )

            targets = torch.tensor(
                train_dataset.targets
            )

            print(
                f"\nHard examples after epoch {epoch}:"
            )

            for class_index, class_name in [
                (2, "bkl"),
                (4, "mel"),
                (5, "nv"),
            ]:
                class_mask = targets == class_index

                hard_count = (
                    hard_mask & class_mask
                ).sum().item()

                total_count = (
                    class_mask.sum().item()
                )

                print(
                    f"{class_name}: "
                    f"{hard_count}/{total_count} "
                    f"({hard_count / total_count:.3f})"
                )

    save_json(
        history,
        f"reports/resnet18_focal_hard15_fold{fold_number}_training_history.json"
    )

    return history

if __name__ == "__main__":
    for fold_number in range(2, 6):
        print(f"\n===== FOLD {fold_number} =====")
        run_fold(fold_number, num_epochs=3)
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import copy

import numpy as np

from sklearn.metrics import f1_score

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

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

class SAM:
    def __init__(self, base_optimizer, rho=0.05):
        if rho < 0.0:
            raise ValueError(f"Invalid rho: {rho}")

        self.base_optimizer = base_optimizer
        self.rho = rho

        self.param_groups = base_optimizer.param_groups
        self.state = {}

    @torch.no_grad()
    def first_step(self, amp_scale=1.0, zero_grad=False):
        grad_norm = torch.norm(
            torch.stack([
                (p.grad / amp_scale).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad / amp_scale

                e_w = grad * (
                    self.rho / (grad_norm + 1e-12)
                )

                self.state[p] = {"e_w": e_w}

                # w -> w + epsilon
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def restore(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    continue

                # w + epsilon -> w
                p.sub_(self.state[p]["e_w"])

        self.state.clear()

        if zero_grad:
            self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


TRAIN_SEED = 123
EMA_DECAY = 0.999

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, model_param in zip(
            ema_model.parameters(),
            model.parameters()
        ):
            ema_param.mul_(decay).add_(
                model_param,
                alpha=1.0 - decay
            )

def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

TRAINABLE_FEATURE_MODULES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def run_validation_epoch_with_macro_f1(
    model,
    val_loader,
    loss_function,
    device
):
    model.eval()

    losses = []
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_function(logits, labels)
            losses.append(loss.item())

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    val_loss = sum(losses) / len(losses)
    val_accuracy = correct / total

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(7)),
        average="macro",
        zero_division=0
    )

    return val_loss, val_accuracy, val_macro_f1

def apply_cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    labels_a = labels
    labels_b = labels[index]

    _, _, height, width = images.shape

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_w // 2, 0, width)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    y2 = np.clip(cy + cut_h // 2, 0, height)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = \
        images[index, :, y1:y2, x1:x2]

    lam = 1.0 - (
        (x2 - x1) * (y2 - y1)
        / (width * height)
    )

    return mixed_images, labels_a, labels_b, lam


def run_convnext_train_epoch(
    model,
    ema_model,
    train_loader,
    loss_function,
    optimizer,
    device
):
    
    model.eval()

    model.features[-TRAINABLE_FEATURE_MODULES:].train()
    model.classifier.train()

    losses = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        loss = run_training_step(
            model,
            images,
            labels,
            loss_function,
            optimizer
        )

        update_ema(
            ema_model,
            model,
            EMA_DECAY
        )

        losses.append(loss)

    return sum(losses) / len(losses)

def run_training_step(
    model,
    images,
    labels,
    loss_function,
    optimizer,
    cutmix_alpha=1.0,
    cutmix_prob=0.5,
):
    optimizer.zero_grad()

    use_cutmix = np.random.rand() < cutmix_prob

    if use_cutmix:
        images, labels_a, labels_b, lam = apply_cutmix(
            images,
            labels,
            alpha=cutmix_alpha,
        )

    def compute_loss():
        logits = model(images)

        if use_cutmix:
            return (
                lam * loss_function(logits, labels_a)
                + (1.0 - lam) * loss_function(logits, labels_b)
            )

        return loss_function(logits, labels)

    # SAM pass 1
    loss = compute_loss()
    loss.backward()

    optimizer.first_step(
        amp_scale=1.0,
        zero_grad=True
    )

    # SAM pass 2
    second_loss = compute_loss()
    second_loss.backward()

    # Return from w + epsilon -> w
    optimizer.restore(zero_grad=False)

    # One real Adam update
    optimizer.base_optimizer.step()

    optimizer.zero_grad()

    return loss.item()

def run_fold(fold_number, num_epochs=15):
    set_training_seed(TRAIN_SEED)

    weights = ConvNeXt_Tiny_Weights.DEFAULT
    transform = weights.transforms()

    train_metadata, val_metadata = get_fold_metadata(fold_number)

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

    model = convnext_tiny(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-TRAINABLE_FEATURE_MODULES:].parameters():
        parameter.requires_grad = True

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        7
    )

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(DEVICE)

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    for parameter in ema_model.parameters():
        parameter.requires_grad = False

    loss_function = FocalLoss(gamma=2.0)


    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    base_optimizer = torch.optim.Adam(
    trainable_params,
    lr=1e-4
)

    optimizer = SAM(
        base_optimizer,
        rho=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    history = []

    best_macro_f1 = -float("inf")
    best_epoch = None

    patience = 4
    epochs_without_improvement = 0

    

    for epoch in range(1, num_epochs + 1):
        train_loss = run_convnext_train_epoch(
            model,
            ema_model,
            train_loader,
            loss_function,
            optimizer,
            DEVICE
        )

        val_loss, val_accuracy, val_macro_f1 = (
            run_validation_epoch_with_macro_f1(
                ema_model,
                val_loader,
                loss_function,
                DEVICE
            )
        )

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_report = build_epoch_report(
            epoch,
            train_loss,
            val_loss,
            val_accuracy
        )

        epoch_report["val_macro_f1"] = val_macro_f1
        epoch_report["learning_rate"] = current_lr

        history.append(epoch_report)

        print(
            f"Fold {fold_number}: "
            f"epoch={epoch}, "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"val_macro_f1={val_macro_f1:.4f}, "
        )

        
        save_json(
            epoch_report,
            f"reports/convnext_tiny_finetune_v14_sam_ema_seed123_"
            f"fold{fold_number}_epoch_{epoch}_report.json"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                ema_model,
                optimizer,
                epoch,
                epoch_report,
                f"checkpoints/convnext_tiny_finetune_v14_sam_ema_seed123_"
                f"fold{fold_number}_best.pt"
            )

            print(
                f"  -> New best checkpoint: "
                f"macro F1={best_macro_f1:.4f}"
            )
        
        
        else:
            epochs_without_improvement += 1

            print(
                f"  -> No improvement "
                f"({epochs_without_improvement}/{patience})"
            )
        scheduler.step()

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping fold {fold_number}. "
                f"Best epoch={best_epoch}, "
                f"best macro F1={best_macro_f1:.4f}"
            )
            break

    save_json(
        history,
        f"reports/convnext_tiny_finetune_v14_sam_ema_seed123_"
        f"fold{fold_number}_training_history.json"
    )

    best_summary = {
        "fold": fold_number,
        "train_seed": TRAIN_SEED,
        "ema_decay": EMA_DECAY,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "sam_rho": 0.05,
        "optimizer": "SAM(Adam)",
    }

    save_json(
        best_summary,
        f"reports/convnext_tiny_finetune_v14_sam_ema_seed123_"
        f"fold{fold_number}_best_summary.json"
    )

    return history

if __name__ == "__main__":
    start_time = time.perf_counter()

    for fold_number in range(2, 6):
        print(f"\n===== FOLD {fold_number} =====")
        run_fold(fold_number, num_epochs=15)

    elapsed = time.perf_counter() - start_time

    print(
        f"\nV14 sanity-check time: "
        f"{elapsed / 60:.1f} minutes"
    )
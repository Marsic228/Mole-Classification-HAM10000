import copy
import json
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from core.cross_validation_split import get_fold_metadata
from experiments.metadata.metadata_features import MetadataFeatureEncoder
from core.utils import build_class_weights
from experiments.metadata.convnext_tiny_metadata_fusion import FrozenImageMetadataResidual


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_SEED = 123
NUM_CLASSES = 7
BATCH_SIZE = 8
NUM_EPOCHS = 15
PATIENCE = 4


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_metadata(metadata, image_ids):
    return (
        metadata
        .set_index("image_id")
        .loc[image_ids]
        .reset_index()
    )


def build_fold_datasets(fold_number):
    cache = torch.load(
        f"cache/convnext_v13_features_fold{fold_number}.pt",
        map_location="cpu",
    )

    train_metadata, val_metadata = get_fold_metadata(
        fold_number
    )

    train_metadata = align_metadata(
        train_metadata,
        cache["train_image_ids"],
    )

    val_metadata = align_metadata(
        val_metadata,
        cache["val_image_ids"],
    )

    encoder = MetadataFeatureEncoder()
    encoder.fit(train_metadata)

    train_metadata_features = encoder.transform(
        train_metadata
    )

    val_metadata_features = encoder.transform(
        val_metadata
    )

    train_dataset = TensorDataset(
        cache["train_image_features"].float(),
        train_metadata_features.float(),
        cache["train_labels"].long(),
    )

    val_dataset = TensorDataset(
        cache["val_image_features"].float(),
        val_metadata_features.float(),
        cache["val_labels"].long(),
    )

    return train_dataset, val_dataset, encoder


def build_v16c_from_v13(fold_number):
    checkpoint = torch.load(
        "checkpoints/"
        "convnext_tiny_finetune_v13_ema_seed123_"
        f"fold{fold_number}_best.pt",
        map_location="cpu",
    )

    state_dict = checkpoint["model_state_dict"]

    model = FrozenImageMetadataResidual(
        image_dim=768,
        metadata_dim=20,
        num_classes=NUM_CLASSES,
    )

    with torch.no_grad():
        model.image_classifier.weight.copy_(
            state_dict["classifier.2.weight"]
        )

        model.image_classifier.bias.copy_(
            state_dict["classifier.2.bias"]
        )

    return model.to(DEVICE)


def build_loaders(
    train_dataset,
    val_dataset,
    seed=TRAIN_SEED,
    batch_size=BATCH_SIZE,
):
    train_labels = train_dataset.tensors[2]

    class_counts = torch.bincount(
        train_labels,
        minlength=NUM_CLASSES,
    )

    class_weights = build_class_weights(
        class_counts.tolist()
    )

    sample_weights = class_weights[train_labels]

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        class_counts,
        class_weights,
    )


def run_validation(
    model,
    loader,
    loss_function,
):
    model.eval()

    losses = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for (
            image_features,
            metadata_features,
            labels,
        ) in loader:
            image_features = image_features.to(DEVICE)
            metadata_features = metadata_features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                image_features,
                metadata_features,
            )

            loss = loss_function(
                logits,
                labels,
            )

            losses.append(loss.item())

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    mean_loss = sum(losses) / len(losses)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    accuracy = (
        torch.tensor(all_predictions)
        == torch.tensor(all_labels)
    ).float().mean().item()

    return float(mean_loss), float(accuracy), float(macro_f1)


def run_train_epoch(
    model,
    loader,
    loss_function,
    optimizer,
):
    model.train()

    losses = []

    for (
        image_features,
        metadata_features,
        labels,
    ) in loader:
        image_features = image_features.to(DEVICE)
        metadata_features = metadata_features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(
            image_features,
            metadata_features,
        )

        loss = loss_function(
            logits,
            labels,
        )

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


def train_v16c(
    fold_number,
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
):
    loss_function = FocalLoss(gamma=2.0)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    baseline_loss, baseline_accuracy, baseline_macro_f1 = (
        run_validation(
            model,
            val_loader,
            loss_function,
        )
    )

    best_macro_f1 = baseline_macro_f1
    best_accuracy = baseline_accuracy
    best_val_loss = baseline_loss
    best_epoch = 0

    best_state = copy.deepcopy(
        model.state_dict()
    )

    history = [
        {
            "epoch": 0,
            "train_loss": None,
            "val_loss": baseline_loss,
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
            "lr": None,
        }
    ]

    epochs_without_improvement = 0

    print(f"\n===== FOLD {fold_number} | v16c =====")
    print(
        f"Epoch 0 | "
        f"val_loss={baseline_loss:.4f} | "
        f"acc={baseline_accuracy:.4f} | "
        f"macro_f1={baseline_macro_f1:.4f}"
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = run_train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
        )

        val_loss, val_accuracy, val_macro_f1 = (
            run_validation(
                model,
                val_loader,
                loss_function,
            )
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_accuracy,
                "macro_f1": val_macro_f1,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch:2d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc={val_accuracy:.4f} | "
            f"macro_f1={val_macro_f1:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    model.load_state_dict(best_state)

    delta_macro_f1 = (
        best_macro_f1 - baseline_macro_f1
    )

    print(
        f"Best Fold {fold_number}: "
        f"epoch={best_epoch}, "
        f"macro_f1={best_macro_f1:.4f}, "
        f"delta={delta_macro_f1:+.4f}"
    )

    return {
        "fold": fold_number,
        "baseline_val_loss": baseline_loss,
        "baseline_accuracy": baseline_accuracy,
        "baseline_macro_f1": baseline_macro_f1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "delta_macro_f1": delta_macro_f1,
        "history": history,
        "model": model,
    }


def save_fold_checkpoint(
    fold_result,
    encoder,
    output_dir,
):
    fold_number = fold_result["fold"]

    checkpoint = {
        "model_state_dict": fold_result["model"].state_dict(),
        "fold": fold_number,
        "best_epoch": fold_result["best_epoch"],
        "baseline_macro_f1": fold_result["baseline_macro_f1"],
        "best_macro_f1": fold_result["best_macro_f1"],
        "delta_macro_f1": fold_result["delta_macro_f1"],
        "age_median": float(encoder.age_median),
        "age_mean": float(encoder.age_mean),
        "age_std": float(encoder.age_std),
    }

    checkpoint_path = (
        output_dir
        / f"convnext_tiny_v16c_metadata_residual_fold{fold_number}_best.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path


def summarize_results(fold_results):
    baseline_f1 = [
        result["baseline_macro_f1"]
        for result in fold_results
    ]

    best_f1 = [
        result["best_macro_f1"]
        for result in fold_results
    ]

    delta_f1 = [
        result["delta_macro_f1"]
        for result in fold_results
    ]

    baseline_accuracy = [
        result["baseline_accuracy"]
        for result in fold_results
    ]

    best_accuracy = [
        result["best_accuracy"]
        for result in fold_results
    ]

    improved_folds = sum(
        delta > 0.0
        for delta in delta_f1
    )

    print("\n===== v16c 5-FOLD SUMMARY =====")

    for result in fold_results:
        print(
            f"Fold {result['fold']}: "
            f"baseline_f1={result['baseline_macro_f1']:.4f} | "
            f"best_epoch={result['best_epoch']} | "
            f"best_f1={result['best_macro_f1']:.4f} | "
            f"delta={result['delta_macro_f1']:+.4f} | "
            f"best_acc={result['best_accuracy']:.4f}"
        )

    print()
    print(
        "Baseline macro F1: "
        f"{mean(baseline_f1):.4f} ± {stdev(baseline_f1):.4f}"
    )
    print(
        "v16c macro F1:     "
        f"{mean(best_f1):.4f} ± {stdev(best_f1):.4f}"
    )
    print(
        "Mean delta F1:     "
        f"{mean(delta_f1):+.4f} ± {stdev(delta_f1):.4f}"
    )
    print(
        "Baseline accuracy: "
        f"{mean(baseline_accuracy):.4f} ± {stdev(baseline_accuracy):.4f}"
    )
    print(
        "v16c accuracy:     "
        f"{mean(best_accuracy):.4f} ± {stdev(best_accuracy):.4f}"
    )
    print(
        "Improved folds:    "
        f"{improved_folds}/5"
    )

    return {
        "baseline_macro_f1_mean": mean(baseline_f1),
        "baseline_macro_f1_std": stdev(baseline_f1),
        "v16c_macro_f1_mean": mean(best_f1),
        "v16c_macro_f1_std": stdev(best_f1),
        "delta_macro_f1_mean": mean(delta_f1),
        "delta_macro_f1_std": stdev(delta_f1),
        "baseline_accuracy_mean": mean(baseline_accuracy),
        "baseline_accuracy_std": stdev(baseline_accuracy),
        "v16c_accuracy_mean": mean(best_accuracy),
        "v16c_accuracy_std": stdev(best_accuracy),
        "improved_folds": improved_folds,
    }


def json_safe_fold_result(result):
    return {
        key: value
        for key, value in result.items()
        if key != "model"
    }


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    set_training_seed(TRAIN_SEED)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir = Path("reports")
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_results = []

    for fold_number in range(1, 6):
        print(f"\n########## FOLD {fold_number} ##########")

        train_dataset, val_dataset, encoder = (
            build_fold_datasets(fold_number)
        )

        train_loader, val_loader, class_counts, class_weights = (
            build_loaders(
                train_dataset,
                val_dataset,
                seed=TRAIN_SEED,
            )
        )

        print("Train samples:", len(train_dataset))
        print("Validation samples:", len(val_dataset))
        print("Class counts:", class_counts)
        print("Class weights:", class_weights)

        model = build_v16c_from_v13(
            fold_number
        )

        trainable_parameters = [
            (name, tuple(parameter.shape))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        print(
            "Trainable parameters:",
            trainable_parameters,
        )

        result = train_v16c(
            fold_number=fold_number,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        checkpoint_path = save_fold_checkpoint(
            result,
            encoder,
            checkpoint_dir,
        )

        print(
            "Saved checkpoint:",
            checkpoint_path,
        )

        fold_results.append(result)

    summary = summarize_results(
        fold_results
    )

    report = {
        "experiment": "v16c_frozen_image_metadata_residual",
        "seed": TRAIN_SEED,
        "num_epochs": NUM_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam(lr=1e-4)",
        "scheduler": "CosineAnnealingLR(T_max=15, eta_min=1e-6)",
        "loss": "FocalLoss(gamma=2.0)",
        "metadata_branch": "Linear(20, 7, bias=False)",
        "folds": [
            json_safe_fold_result(result)
            for result in fold_results
        ],
        "summary": summary,
    }

    report_path = (
        report_dir
        / "convnext_tiny_v16c_metadata_residual_cv_summary.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print("\nSaved report:", report_path)

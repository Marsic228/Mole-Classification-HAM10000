import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from cross_validation_split import get_fold_metadata
from cross_validation_dataset import CrossValidationDataset


HARD_CLASSES = {2, 4, 5}  # bkl, mel, nv


def load_reference_model(checkpoint_path):
    model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 7)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def collect_fold_oof_predictions(fold_number):
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    _, val_metadata = get_fold_metadata(fold_number)

    val_metadata = val_metadata.reset_index(drop=True)

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

    model = load_reference_model(
        f"checkpoints/resnet18_focal_fold{fold_number}_epoch_3.pt"
    )

    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            batch_predictions = logits.argmax(dim=1)

            predictions.extend(
                batch_predictions.tolist()
            )
            targets.extend(
                labels.tolist()
            )

    assert len(predictions) == len(val_metadata)
    assert len(targets) == len(val_metadata)

    fold_results = val_metadata[
        ["image_id", "dx"]
    ].copy()

    fold_results["true_idx"] = targets
    fold_results["pred_idx"] = predictions
    fold_results["source_fold"] = fold_number

    fold_results["is_hard"] = [
        (true_idx in HARD_CLASSES)
        and (pred_idx != true_idx)
        for true_idx, pred_idx in zip(
            targets,
            predictions
        )
    ]

    return fold_results


def main():
    all_folds = []

    for fold_number in range(1, 6):
        print(f"Collecting OOF predictions: Fold {fold_number}")

        fold_results = collect_fold_oof_predictions(
            fold_number
        )

        all_folds.append(fold_results)

    oof_results = pd.concat(
        all_folds,
        ignore_index=True
    )

    print("\nTotal OOF samples:", len(oof_results))
    print(
        "Unique image_ids:",
        oof_results["image_id"].nunique()
    )

    assert len(oof_results) == 8530
    assert oof_results["image_id"].nunique() == 8530

    print("\nHard examples by class:")

    for class_name in ["bkl", "mel", "nv"]:
        class_rows = (
            oof_results["dx"] == class_name
        )

        hard_rows = (
            class_rows
            & oof_results["is_hard"]
        )

        total = class_rows.sum()
        hard = hard_rows.sum()

        print(
            f"{class_name}: "
            f"total={total}, "
            f"hard={hard}, "
            f"ratio={hard / total:.4f}"
        )

    print(
        "\nTotal hard examples:",
        oof_results["is_hard"].sum()
    )

    oof_results.to_csv(
        "reports/resnet18_focal_oof_hard_examples.csv",
        index=False
    )

    print(
        "\nSaved to "
        "reports/resnet18_focal_oof_hard_examples.csv"
    )


if __name__ == "__main__":
    main()
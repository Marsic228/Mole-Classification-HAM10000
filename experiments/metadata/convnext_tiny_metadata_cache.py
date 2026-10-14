import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from core.cross_validation_split import get_fold_metadata
from core.cross_validation_dataset import CrossValidationDataset

from pathlib import Path


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_v13_feature_extractor(
    checkpoint_path,
    num_classes=7
):
    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        in_features,
        num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Remove only the final classification layer.
    # ConvNeXt will now return the 768-d feature vector.
    model.classifier[2] = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    return model 
def collect_outputs(model, loader):
    all_outputs = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):
                outputs = model(images)

            all_outputs.append(
                outputs.cpu()
            )

            all_labels.append(
                labels.clone()
            )

    return (
        torch.cat(all_outputs, dim=0),
        torch.cat(all_labels, dim=0),
    )

if __name__ == "__main__":
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    transform = weights.transforms()

    cache_dir = Path("cache")
    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for fold_number in range(1, 6):
        print(f"\n===== FOLD {fold_number} =====")

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )

        model = load_v13_feature_extractor(
            "checkpoints/"
            "convnext_tiny_finetune_v13_ema_seed123_"
            f"fold{fold_number}_best.pt"
        )

        train_features, train_labels = collect_outputs(
            model,
            train_loader
        )

        val_features, val_labels = collect_outputs(
            model,
            val_loader
        )

        print("Train image features:", train_features.shape)
        print("Validation image features:", val_features.shape)

        assert train_features.shape[0] == len(train_metadata)
        assert val_features.shape[0] == len(val_metadata)

        torch.save(
            {
                "train_image_features": train_features,
                "train_labels": train_labels,
                "train_image_ids": train_metadata["image_id"].tolist(),

                "val_image_features": val_features,
                "val_labels": val_labels,
                "val_image_ids": val_metadata["image_id"].tolist(),
            },
            cache_dir
            / f"convnext_v13_features_fold{fold_number}.pt"
        )

        print(
            f"Fold {fold_number} metadata cache saved."
        )
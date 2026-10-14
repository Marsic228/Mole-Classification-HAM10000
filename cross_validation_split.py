import pandas as pd
from cross_validation_dataset import CrossValidationDataset
from torchvision import transforms

from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import count_images_per_class, build_class_weights

from sklearn.model_selection import StratifiedGroupKFold
from preprocessing.validate_image_paths import load_metadata
from preprocessing.split_config import build_split_config
from preprocessing.split_dataset import (
    group_records_by_lesion_id,
    split_lesion_ids,
    assign_records_to_splits,
)

records = load_metadata("HAM10000_metadata.csv")

config = build_split_config()

grouped = group_records_by_lesion_id(records)
lesion_ids = list(grouped.keys())

split_ids = split_lesion_ids(lesion_ids, config)
splits = assign_records_to_splits(grouped, split_ids)

train_metadata = pd.DataFrame(splits["train"])
train_metadata["source_split"] = "train"

val_metadata = pd.DataFrame(splits["val"])
val_metadata["source_split"] = "val"

metadata = pd.concat(
    [train_metadata, val_metadata],
    ignore_index=True
)

X = metadata[["image_id"]]
y = metadata["dx"]
groups = metadata["lesion_id"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

def get_fold_metadata(fold_number=1):
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):
        if fold == fold_number:
            train_fold_metadata = metadata.iloc[train_idx]
            val_fold_metadata = metadata.iloc[val_idx]

            return train_fold_metadata, val_fold_metadata

if __name__ == "__main__":
    train_transform = transforms.ToTensor()

    special_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    val_transform = transforms.ToTensor()

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1
    ):
        train_metadata = metadata.iloc[train_idx]
        val_metadata = metadata.iloc[val_idx]

        train_lesions = set(train_metadata["lesion_id"])
        val_lesions = set(val_metadata["lesion_id"])

        overlap = train_lesions & val_lesions

        print(f"Fold {fold}")
        print("Train:", len(train_metadata))
        print("Validation:", len(val_metadata))
        print("Lesion overlap:", len(overlap))

        print(
        val_metadata["dx"]
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
    )
        
        assert len(overlap) == 0, f"Leakage detected in fold {fold}"

        if fold == 1:
            train_dataset = CrossValidationDataset(
                train_metadata,
                "data/processed",
                transform=train_transform,
                special_transform=special_train_transform,
                special_classes=["mel"],
            )

            val_dataset = CrossValidationDataset(
                val_metadata,
                "data/processed",
                transform=val_transform,
            )

            image, label = train_dataset[0]

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

            train_images, train_labels = next(iter(train_loader))
            val_images, val_labels = next(iter(val_loader))

            print("Train batch:", train_images.shape, train_labels.shape)
            print("Validation batch:", val_images.shape, val_labels.shape)
            print("Train labels:", train_labels)

            print("Train dataset:", len(train_dataset))
            print("Validation dataset:", len(val_dataset))
            print("Class mapping:", train_dataset.class_to_idx)
            print("Image shape:", image.shape)
            print("Label:", label)
            print("Special class indices:", train_dataset.special_class_indices)

            break
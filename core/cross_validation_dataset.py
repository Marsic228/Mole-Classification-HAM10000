from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import torch


class CrossValidationDataset(Dataset):
    def __init__(
        self,
        metadata,
        processed_dir,
        transform=None,
        special_transform=None,
        special_classes=None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.special_transform = special_transform

        self.classes = sorted(self.metadata["dx"].unique())

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.special_class_indices = {
            self.class_to_idx[class_name]
            for class_name in (special_classes or [])
        }

        self.samples = []

        for _, row in self.metadata.iterrows():
            path = (
                self.processed_dir
                / row["source_split"]
                / row["dx"]
                / f"{row['image_id']}_224x224.jpg"
            )

            label_index = self.class_to_idx[row["dx"]]

            self.samples.append((str(path), label_index))

        self.targets = [
            label_index
            for _, label_index in self.samples
        ]


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label_index = self.samples[index]

        image = Image.open(path).convert("RGB")

        if (
            label_index in self.special_class_indices
            and self.special_transform is not None
        ):
            image = self.special_transform(image)
        elif self.transform is not None:
            image = self.transform(image)

        return image, label_index

def prepare_segmentation_pair(image, mask):
    if image.size != mask.size:
        raise ValueError("Image and mask sizes differ")

    image = image.resize(
        (256, 192),
        Image.Resampling.BILINEAR
    )

    mask = mask.resize(
        (256, 192),
        Image.Resampling.NEAREST
    )

    image_array = (
        np.array(image, dtype=np.float32)
        .transpose(2, 0, 1) / 255.0
    )

    mask_array = (
        np.array(mask) > 127
    ).astype(np.float32)[None, ...]

    return (
        torch.from_numpy(image_array),
        torch.from_numpy(mask_array)
    )

class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata,
        mask_mapping,
        image_root="data/images",
        mask_archive="Masks.zip",
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.mask_mapping = mask_mapping
        self.image_root = Path(image_root)
        self.mask_archive = Path(mask_archive)

        with ZipFile(self.mask_archive) as archive:
            self.mask_files = {
                Path(name).name: name
                for name in archive.namelist()
                if name.lower().endswith(".png")
            }

        missing = (
            set(self.metadata["image_id"])
            - set(self.mask_mapping.index)
        )

        if missing:
            raise ValueError(
                f"Missing mask mappings: {len(missing)}"
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        image_id = self.metadata.iloc[index]["image_id"]

        image_path = self.image_root / f"{image_id}.jpg"
        mask_name = Path(self.mask_mapping.loc[image_id]).name

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with ZipFile(self.mask_archive) as archive:
            with Image.open(
                BytesIO(archive.read(self.mask_files[mask_name]))
            ) as source:
                mask = source.convert("L")

        image_tensor, mask_tensor = prepare_segmentation_pair(
            image, mask
        )

        return image_tensor, mask_tensor, image_id
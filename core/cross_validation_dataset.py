from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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
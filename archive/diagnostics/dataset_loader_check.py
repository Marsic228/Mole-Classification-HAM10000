from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import count_images_per_class, build_class_weights

class AugmentedImageFolder(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        special_transform=None,
        special_classes=None
    ):
        super().__init__(root, transform=transform)

        self.special_transform = special_transform

        if special_classes is None:
            self.special_classes = set()
        else:
            self.special_classes = {
                self.class_to_idx[class_name] if isinstance(class_name, str) else class_name
                for class_name in special_classes
            }

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = self.loader(path)

        if self.special_transform is not None and label in self.special_classes:
            image = self.special_transform(image)
        elif self.transform is not None:
            image = self.transform(image)

        return image, label


def build_image_dataset(
    data_dir,
    transform=None,
    special_transform=None,
    special_classes=None
):
    data_dir = Path(data_dir)

    if special_transform is not None:
        dataset = AugmentedImageFolder(
            data_dir,
            transform=transform,
            special_transform=special_transform,
            special_classes=special_classes
        )
    else:
        dataset = ImageFolder(
            data_dir,
            transform=transform
        )

    return dataset


def inspect_dataset_item(dataset, index):
    image, label_index = dataset[index]
    image_type = str(type(image))
    class_name = dataset.classes[label_index]
    result = {
        "image_type": image_type,
        "label_index": label_index,
        "class_name": class_name
    }
    return result

def build_transforms():
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    special_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    result = {
        "train": train_transform,
        "special_train": special_train_transform,
        "val": val_transform,
        "test": test_transform,
    }

    return result
    


def build_data_loader(dataset, batch_size, shuffle=False, sampler=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler
    )
    return loader


def build_split_datasets(processed_dir, transforms_dict, special_classes=None):
    processed_dir = Path(processed_dir)

    train_dataset = build_image_dataset(
        processed_dir / "train",
        transform=transforms_dict["train"],
        special_transform=transforms_dict["special_train"],
        special_classes=special_classes
    )

    val_dataset = build_image_dataset(
        processed_dir / "val",
        transform=transforms_dict["val"]
    )

    test_dataset = build_image_dataset(
        processed_dir / "test",
        transform=transforms_dict["test"]
    )

    result = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    return result


def build_split_loaders(datasets, batch_size):
    class_counts = count_images_per_class(datasets["train"])
    class_weights = build_class_weights(class_counts)
    train_labels = datasets["train"].targets
    sample_weights = class_weights[train_labels]
    train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
    )
    train_loader = build_data_loader(
    datasets["train"],
    batch_size=batch_size,
    sampler=train_sampler
    )
    val_loader = build_data_loader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = build_data_loader(datasets["test"], batch_size=batch_size, shuffle=False)

    result = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return result

    
if __name__ == "__main__":
    transforms_dict = build_transforms()

    datasets = build_split_datasets(
    "data/processed",
    transforms_dict,
    special_classes=["df", "mel"]
    )
    print(datasets["train"].class_to_idx)
    print(datasets["train"].special_classes)

    print(len(datasets["train"]))
    print(len(datasets["val"]))
    print(len(datasets["test"]))
    image, label = datasets["train"][0]

    print(image.shape)
    print(label)
    loaders = build_split_loaders(datasets, batch_size=8)

    train_images, train_labels = next(iter(loaders["train"]))
    val_images, val_labels = next(iter(loaders["val"]))
    test_images, test_labels = next(iter(loaders["test"]))

    print(train_images.shape, train_labels.shape)
    print(val_images.shape, val_labels.shape)
    print(test_images.shape, test_labels.shape)
    train_images, train_labels = next(iter(loaders["train"]))
    print(train_labels)
import torch
from torch import nn
from torchvision import transforms
from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders

def build_baseline_cnn(num_classes):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(32 * 54 * 54, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
    return model

if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)
    train_images, train_labels = next(iter(loaders["train"]))
    model = build_baseline_cnn(num_classes=7)
    outputs = model(train_images)
    predictions = outputs.argmax(dim=1)
    print("Images shape:", train_images.shape)
    print("Labels shape:", train_labels.shape)
    print("Outputs shape:", outputs.shape)
    print("Predictions shape:", predictions.shape)
    print("Predictions:", predictions)
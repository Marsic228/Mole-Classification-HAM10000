import torch
from torchvision import models, transforms
from torch import nn
from dataset_loader_check import build_split_datasets, build_split_loaders



weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 7)

transform = weights.transforms()
resnet_transforms = {
    "train": transform,
    "special_train": None,
    "val": transform,
    "test": transform,
}
print(transform)

datasets = build_split_datasets("data/processed", resnet_transforms)
loaders = build_split_loaders(datasets, batch_size=8)

images, labels = next(iter(loaders["train"]))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
model.eval()

with torch.no_grad():
    logits = model(images)
    predictions = logits.argmax(dim=1)

print("Logits shape:", logits.shape)
print("Predictions shape:", predictions.shape)

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)




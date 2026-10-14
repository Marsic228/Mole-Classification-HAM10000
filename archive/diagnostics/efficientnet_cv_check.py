import torch.nn as nn
import torch
from cross_validation_split import get_fold_metadata
from torch.utils.data import DataLoader
from cross_validation_dataset import CrossValidationDataset
from torch.utils.data import WeightedRandomSampler
from utils import count_images_per_class, build_class_weights

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.features[7].parameters():
    parameter.requires_grad = True

for parameter in model.features[8].parameters():
    parameter.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 7)

model.classifier[1] = nn.Linear(
    in_features,
    7
)


print(model.classifier)

# Get Fold 1 metadata
train_metadata, val_metadata = get_fold_metadata(1)

# EfficientNet preprocessing
model_transform = weights.transforms()

train_dataset = CrossValidationDataset(
    train_metadata,
    "data/processed",
    transform=model_transform
)

# Build validation dataset
val_dataset = CrossValidationDataset(
    val_metadata,
    "data/processed",
    transform=model_transform
)

# Build validation loader
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# Take one real batch
images, labels = next(iter(val_loader))

# Forward pass
model.eval()

with torch.no_grad():
    logits = model(images)

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

print("Images:", images.shape)
print("Labels:", labels.shape)
print("Logits:", logits.shape)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

train_images, train_labels = next(iter(train_loader))

model.eval()
model.classifier.train()

optimizer.zero_grad()

logits = model(train_images)

loss = criterion(logits, train_labels)

loss.backward()

optimizer.step()

print("Training logits:", logits.shape)
print("Training labels:", train_labels.shape)
print("Loss:", loss.item())


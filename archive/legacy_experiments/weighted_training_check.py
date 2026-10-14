# This script checks that class-weighted CrossEntropyLoss works
# for the imbalanced HAM10000 training split.
import torch
from torch import nn
from torchvision import transforms

from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
from training_step_check import build_optimizer, run_training_step


def count_images_per_class(dataset):
    # 1. Create a counter list with one zero per class
    counter_list = [0] * len(dataset.classes)

    # 2. Loop through dataset.samples and count labels
    for image_path, class_index in dataset.samples:
        counter_list[class_index] += 1

    # 3. Return class counts
    return counter_list


def build_class_weights(class_counts):
    class_counts = torch.tensor(class_counts, dtype=torch.float32)

    total_images = class_counts.sum()
    class_weights = torch.sqrt(total_images / class_counts)
    class_weights = class_weights / class_weights.mean()

    return class_weights

def build_weighted_loss_function(class_weights):
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    return loss_function

def build_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function

def main():
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)
    class_counts = count_images_per_class(datasets["train"])
    class_weights = build_class_weights(class_counts)
    train_images, train_labels = next(iter(loaders["train"]))
    model = build_baseline_cnn(num_classes=7)
    loss_function = build_weighted_loss_function(class_weights)
    optimizer = build_optimizer(model, learning_rate=0.001)
    loss = run_training_step(model, train_images, train_labels, loss_function, optimizer)
    
    print("Class names:")
    print(datasets["train"].classes)

    print("Class counts:")
    print(class_counts)

    print("Class weights:")
    print(class_weights)

    print("First weighted loss:")
    print(loss)

if __name__ == "__main__":
    main()
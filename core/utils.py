import torch
from torch import nn
from torchvision import transforms

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
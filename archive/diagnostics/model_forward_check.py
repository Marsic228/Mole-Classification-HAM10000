import torch
from torch import nn
from torchvision import transforms
from dataset_loader_check import build_split_datasets, build_split_loaders


def build_dummy_model(num_classes):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, num_classes)
    )
    return model


def run_forward_pass(model, images):
    outputs = model(images)
    return outputs


def get_predicted_classes(outputs):
    result = outputs.argmax(dim=1)
    return result


if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    train_images, train_labels = next(iter(loaders["train"]))

    model = build_dummy_model(num_classes=7)

    outputs = run_forward_pass(model, train_images)

    print(train_images.shape)
    print(outputs.shape)
    test_outputs = torch.tensor([
    [0.1, 0.9, 0.2],
    [3.0, 1.0, 2.0]
    ])

    predictions = get_predicted_classes(outputs)

    print(outputs.shape)
    print(predictions.shape)
    print(predictions)
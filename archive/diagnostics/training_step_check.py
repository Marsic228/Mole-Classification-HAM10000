import torch
from torch import nn
from torchvision import transforms
from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders
from archive.diagnostics.baseline_cnn_forward_check import build_baseline_cnn

def build_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function

def build_optimizer(model, learning_rate):
    result = torch.optim.Adam(model.parameters() , lr=learning_rate)
    return result

def run_training_step(model, images, labels, loss_function, optimizer):
    optimizer.zero_grad()
    logits = model(images)
    loss = loss_function(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)
    train_images, train_labels = next(iter(loaders["train"]))
    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, 0.001)
    loss = run_training_step(model, train_images, train_labels, loss_function, optimizer)
    print("training loss:", loss)
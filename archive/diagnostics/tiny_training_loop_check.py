from training_step_check import build_loss_function, build_optimizer, run_training_step
from torchvision import transforms
from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn

def run_tiny_training_loop(model, train_loader, loss_function, optimizer, max_batches):
    losses = []

    for batch_index, batch in enumerate(train_loader):
        images, labels = batch

        loss = run_training_step(model, images, labels, loss_function, optimizer)
        losses.append(loss)

        if len(losses) >= max_batches:
            break

    return losses

if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, 0.001)

    losses = run_tiny_training_loop(model, loaders["train"], loss_function, optimizer, 3)
    print("tiny training losses:", losses)
        
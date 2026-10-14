from archive.diagnostics.training_step_check import run_training_step, build_loss_function, build_optimizer
from torchvision import transforms
from archive.diagnostics.dataset_loader_check import build_split_datasets, build_split_loaders
from archive.diagnostics.baseline_cnn_forward_check import build_baseline_cnn
import torch



def run_one_train_epoch(model, train_loader, loss_function, optimizer, max_batches=None):
    model.train()

    losses = []

    for images, labels in train_loader:
        loss = run_training_step(model, images, labels, loss_function, optimizer)
        losses.append(loss)

        if max_batches is not None and len(losses) >= max_batches:
            break

    average_loss = sum(losses) / len(losses)
    return average_loss

def run_validation_epoch(model, val_loader, loss_function, max_batches=None):
   model.eval()

   losses = []
   accuracies = []
   with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images)
        loss = loss_function(logits, labels)
        accuracy = calculate_batch_accuracy(logits, labels)
        losses.append(loss.item())
        accuracies.append(accuracy)

        if max_batches is not None and len(losses) >= max_batches:
            break
       
   average_loss = sum(losses) / len(losses)
   average_accuracy = sum(accuracies) / len(accuracies)
   return average_loss, average_accuracy

def calculate_batch_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    correct_mask = predictions == labels
    correct = correct_mask.sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def inspect_validation_predictions(model, val_loader):
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)
            return predictions, labels

if __name__ == "__main__":
    tensor_transform = transforms.ToTensor()

    datasets = build_split_datasets("data/processed", transform=tensor_transform)
    loaders = build_split_loaders(datasets, batch_size=8)

    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, 0.001)

    average_loss = run_one_train_epoch(
    model,
    loaders["train"],
    loss_function,
    optimizer,
    max_batches=10
    )

    print("Average train loss:", average_loss)
    print("Type:", type(average_loss))
    print("Batches used:", 10)

    val_loss, val_accuracy = run_validation_epoch(
    model,
    loaders["val"],
    loss_function,
    max_batches=10
)

    print("Average val loss:", val_loss)
    print("Average val accuracy:", val_accuracy)
    print("Val loss type:", type(val_loss))
    print("Val accuracy type:", type(val_accuracy))
    print("Batches used:", 10)

    logits = torch.tensor([
    [0.1, 0.9, 0.0],
    [2.0, 0.5, 0.1],
    [0.2, 0.3, 1.5],
    ])

    labels = torch.tensor([1, 0, 2])

    test_accuracy = calculate_batch_accuracy(logits, labels)
    print("Test batch accuracy:", test_accuracy)

    predictions, labels = inspect_validation_predictions(model, loaders["val"])

    print("Validation predictions:", predictions)
    print("Validation labels:", labels)
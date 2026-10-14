from dataset_loader_check import build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
from training_step_check import build_loss_function, build_optimizer
from one_epoch_training_check import run_one_train_epoch, run_validation_epoch
from torchvision import transforms



def run_baseline_check(max_batches):
    transform = transforms.ToTensor()
    datasets = build_split_datasets("data/processed", transform=transform)
    loaders = build_split_loaders(datasets, batch_size=8)
    model = build_baseline_cnn(num_classes=7)
    loss_function = build_loss_function()
    optimizer = build_optimizer(model, learning_rate=0.001)
    
    train_loss = run_one_train_epoch(
        model, 
        loaders["train"], 
        loss_function, 
        optimizer, 
        max_batches=max_batches,
        )
    
    val_loss, val_accuracy = run_validation_epoch(
        model, 
        loaders["val"], 
        loss_function,  
        max_batches=max_batches,
        )


    result = {
        "max_batches": max_batches,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    }

    return result

if __name__ == "__main__":
    metrics = run_baseline_check(max_batches=10)
    print(metrics)
    print(type(metrics["train_loss"]))
    print(type(metrics["val_loss"]))
    print(type(metrics["val_accuracy"]))
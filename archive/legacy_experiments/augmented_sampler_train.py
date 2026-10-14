from dataset_loader_check import build_transforms, build_split_datasets, build_split_loaders
from baseline_cnn_forward_check import build_baseline_cnn
from torch import nn
from training_step_check import build_optimizer
from one_epoch_training_check import run_one_train_epoch, run_validation_epoch
from weighted_train import build_epoch_report, save_json, save_checkpoint

def run_augmented_sampler_training_experiment(num_epochs, train_max_batches=None, val_max_batches=None):
    transforms_dict = build_transforms()
    datasets = build_split_datasets(
        "data/processed",
        transforms_dict,
        special_classes=["df", "mel"]
    )

    loaders = build_split_loaders(
        datasets,
        batch_size=8
    )

    model = build_baseline_cnn(num_classes=7)
    loss_function = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, learning_rate=0.001)

    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_one_train_epoch(
            model,
            loaders["train"],
            loss_function,
            optimizer,
            max_batches=train_max_batches,
        )

        val_loss, val_accuracy = run_validation_epoch(
            model,
            loaders["val"],
            loss_function,
            max_batches=val_max_batches,
        )

        epoch_report = build_epoch_report(epoch, train_loss, val_loss, val_accuracy)
        history.append(epoch_report)

        save_json(epoch_report, f"reports/augmented_sampler_epoch_{epoch}_report.json")
        save_checkpoint(
            model,
            optimizer,
            epoch,
            epoch_report,
            f"checkpoints/augmented_sampler_epoch_{epoch}.pt",
        )
    
    save_json(history, "reports/augmented_sampler_training_history.json")
    
    return history
    
if __name__ == "__main__":
    history = run_augmented_sampler_training_experiment(
       num_epochs=3,
       train_max_batches=None,
       val_max_batches=None,
    )

    print(history)
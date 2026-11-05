import pandas as pd
import torch

from torch.utils.data import DataLoader
from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import SegmentationDataset
from core.segmentation_unet import SmallUNet
from core.segmentation_losses import segmentation_loss
from evaluation.segmentation_validation import evaluate_segmentation
from time import perf_counter
import json
from pathlib import Path


torch.manual_seed(123)

train, val = get_fold_metadata(3)

mapping = pd.read_csv(
    "Masks_to_IsicId_mapping.csv"
).set_index("ISIC_ID")["Filename"]

train_dataset = SegmentationDataset(train, mapping)
val_dataset = SegmentationDataset(val, mapping)

train_loader = DataLoader(
    train_dataset, batch_size=4,
    shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=4,
    shuffle=False, num_workers=0
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SmallUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_images = 0

    for batch_idx, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = segmentation_loss(logits, masks)

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_images += batch_size

        if batch_idx % 250 == 0 or batch_idx == len(loader):
            print(
                f"Batch {batch_idx}/{len(loader)} | "
                f"mean loss: {total_loss / total_images:.4f}"
            )

    return total_loss / total_images

report_dir = Path("reports/segmentation/small_unet_fold3_seed123")
checkpoint_path = Path(
    "checkpoints/segmentation/small_unet_fold3_seed123_best.pt"
)

history_path = report_dir / "history.json"

if history_path.exists() or checkpoint_path.exists():
    raise FileExistsError(
        "Experiment outputs already exist. Do not overwrite them."
    )

report_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

best_dice = -1.0
best_epoch = None
epochs_without_improvement = 0

history = []
start = perf_counter()

for epoch in range(1, 13):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, device
    )

    summary, records = evaluate_segmentation(
        model, val_loader, device
    )

    assert summary["n_images"] == len(val_dataset)
    assert len({r["image_id"] for r in records}) == len(val_dataset)

    current_dice = summary["mean_dice"]

    epoch_report = {
        "epoch": epoch,
        "train_loss": train_loss,
        **summary,
    }
    history.append(epoch_report)

    print(
        f"\nEpoch {epoch}: "
        f"loss={train_loss:.4f}, "
        f"Dice={current_dice:.4f}, "
        f"IoU={summary['mean_iou']:.4f}, "
        f"empty={summary['empty_predictions']}"
    )

    if current_dice > best_dice + 1e-4:
        best_dice = current_dice
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "fold": 3,
                "seed": 123,
                "mean_dice": best_dice,
                "config": {
                    "input_size": [192, 256],
                    "batch_size": 4,
                    "learning_rate": 1e-3,
                    "loss": "BCE + soft Dice",
                    "threshold": 0.5,
                },
            },
            checkpoint_path,
        )

        pd.DataFrame(records).to_csv(
            report_dir / "best_validation_per_image.csv",
            index=False,
        )

        print("New best checkpoint saved.")
    else:
        epochs_without_improvement += 1

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    if epochs_without_improvement >= 4:
        print("Early stopping triggered.")
        break

print("\nFinal Fold 3 summary:")
print("Best epoch:", best_epoch)
print("Best Dice:", round(best_dice, 4))
print("Elapsed minutes:", round((perf_counter() - start) / 60, 2))
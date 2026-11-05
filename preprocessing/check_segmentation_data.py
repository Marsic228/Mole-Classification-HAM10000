from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from core.segmentation_unet import SmallUNet

from torch.utils.data import DataLoader
from evaluation.segmentation_validation import evaluate_segmentation

from core.cross_validation_dataset import (
    SegmentationDataset,
    prepare_segmentation_pair,
)

from core.canonical_cv import get_fold_metadata
from evaluation.segmentation_metrics import segmentation_metrics

with ZipFile("ISIC2018_Task1_Training_GroundTruth.zip") as z:
    expert_ids = {
        Path(name).stem.rsplit("_", 1)[0]
        for name in z.namelist()
        if name.lower().endswith("_segmentation.png")
    }

mask_ids = set(
    pd.read_csv("Masks_to_IsicId_mapping.csv")["ISIC_ID"]
)

local_ids = {
    path.stem
    for path in Path("data/images").glob("*.jpg")
}

print("Expert masks:", len(expert_ids))
print("Reviewed-mask collection:", len(mask_ids))
print("Expert / reviewed-mask overlap:", len(expert_ids & mask_ids))
print("Expert images available locally:", len(expert_ids & local_ids))
print("Reviewed masks with local images:", len(mask_ids & local_ids))


train, val = get_fold_metadata(1)

mask_mapping = pd.read_csv(
    "Masks_to_IsicId_mapping.csv"
).set_index("ISIC_ID")["Filename"]


with ZipFile("Masks.zip") as archive:
    files = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".png")
    }

    for split_name, subset in [("train", train), ("val", val)]:
        print(f"\n{split_name}: {len(subset)} images")

        for row in subset.sample(n=3, random_state=42).itertuples():
            image_path = Path("data/images") / f"{row.image_id}.jpg"
            mask_name = Path(mask_mapping.loc[row.image_id]).name

            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image_size = image.size

            with Image.open(
                BytesIO(archive.read(files[mask_name]))
            ) as source:
                mask = np.asarray(source.convert("L")) > 127

            assert mask.shape == (image_size[1], image_size[0])

            mask_image = Image.fromarray(
                mask.astype(np.uint8) * 255
            )

            image_tensor, mask_tensor = prepare_segmentation_pair(
                image, mask_image
            )

            assert image_tensor.shape == (3, 192, 256)
            assert mask_tensor.shape == (1, 192, 256)

            assert 0 <= image_tensor.min() <= image_tensor.max() <= 1
            assert set(torch.unique(mask_tensor).tolist()) <= {0.0, 1.0}

            print(
                row.image_id,
                "image:", tuple(image_tensor.shape),
                "mask:", tuple(mask_tensor.shape),
                "foreground:", round(mask_tensor.mean().item(), 4)
            )

            print(
                row.image_id,
                "size:", image_size,
                "mask area:", round(float(mask.mean()), 4)
            )

train_dataset = SegmentationDataset(
    train, mask_mapping
)

val_dataset = SegmentationDataset(
    val, mask_mapping
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

images, masks, image_ids = next(iter(train_loader))

assert images.shape == (4, 3, 192, 256)
assert masks.shape == (4, 1, 192, 256)

print("\nSegmentation DataLoader:")
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Image batch:", tuple(images.shape))
print("Mask batch:", tuple(masks.shape))
print("Image IDs:", image_ids)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SmallUNet().to(device)
model.eval()

with torch.inference_mode():
    logits = model(images.to(device))

assert logits.shape == masks.shape
assert torch.isfinite(logits).all()

print("\nU-Net forward-pass check:")
print("Device:", device)
print("Output shape:", tuple(logits.shape))
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

if device.type == "cuda":
    print(
        "GPU memory allocated (MB):",
        round(torch.cuda.memory_allocated() / 1024**2, 1)
    )

model.train()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

batch_images = images.to(device)
batch_masks = masks.to(device)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

optimizer.zero_grad(set_to_none=True)

logits = model(batch_images)
loss = criterion(logits, batch_masks)

assert torch.isfinite(loss).item()

loss.backward()

gradients = [
    p.grad for p in model.parameters()
    if p.grad is not None
]

assert gradients
assert all(torch.isfinite(g).all().item() for g in gradients)

weights_before = model.head.weight.detach().clone()

optimizer.step()

weights_updated = not torch.equal(
    weights_before,
    model.head.weight
)

assert weights_updated

print("\nOne-batch training test:")
print("Loss:", round(loss.item(), 4))
print("Gradients finite: True")
print("Weights updated:", weights_updated)

if device.type == "cuda":
    print(
        "Peak GPU memory (MB):",
        round(
            torch.cuda.max_memory_allocated(device) / 1024**2,
            1
        )
    )

test_logits = torch.tensor([[[[
    10.0, -10.0
], [
    -10.0, 10.0
]]]])

test_target = torch.tensor([[[[
    1.0, 1.0
], [
    0.0, 0.0
]]]])

dice, iou = segmentation_metrics(test_logits, test_target)

assert torch.allclose(dice, torch.tensor([0.5]))
assert torch.allclose(iou, torch.tensor([1 / 3]))

print("\nSegmentation metrics test:")
print("Dice:", round(dice.item(), 4))
print("IoU:", round(iou.item(), 4))

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

summary, records = evaluate_segmentation(
    model,
    val_loader,
    device,
    max_batches=2,
)

assert summary["n_images"] == 8
assert len({r["image_id"] for r in records}) == 8
assert 0 <= summary["mean_dice"] <= 1
assert 0 <= summary["mean_iou"] <= 1

print("\nValidation smoke test:")
for key, value in summary.items():
    print(f"{key}: {value}")

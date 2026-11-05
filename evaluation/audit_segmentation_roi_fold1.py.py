from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import SegmentationDataset
from core.segmentation_unet import SmallUNet
from evaluation.segmentation_metrics import segmentation_metrics
from preprocessing.roi_crop import crop_lesion_with_margin

_, val = get_fold_metadata(1)

mapping = pd.read_csv(
    "Masks_to_IsicId_mapping.csv"
).set_index("ISIC_ID")["Filename"]

loader = DataLoader(
    SegmentationDataset(val, mapping),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "checkpoints/segmentation/small_unet_fold1_seed123_best.pt",
    map_location=device,
    weights_only=True,
)

model = SmallUNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

records = []

with torch.inference_mode():
    for images, targets, image_ids in loader:
        logits = model(images.to(device))

        dice, iou = segmentation_metrics(
            logits, targets.to(device)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).cpu().numpy()

        for j, image_id in enumerate(image_ids):
            reference = targets[j, 0].numpy() >= 0.5
            predicted = predictions[j, 0]

            ref_pil = Image.fromarray(
                reference.astype(np.uint8) * 255
            )
            pred_pil = Image.fromarray(
                predicted.astype(np.uint8) * 255
            )

            cropped, status = crop_lesion_with_margin(
                ref_pil, pred_pil
            )

            reference_pixels = reference.sum()
            retained_pixels = (
                np.asarray(cropped) > 127
            ).sum()

            retention = (
                retained_pixels / reference_pixels
                if reference_pixels else float("nan")
            )

            crop_fraction = (
                cropped.width * cropped.height
                / (ref_pil.width * ref_pil.height)
            )

            records.append({
                "image_id": image_id,
                "dice": dice[j].item(),
                "iou": iou[j].item(),
                "reference_area": reference.mean(),
                "predicted_area": predicted.mean(),
                "status": status,
                "retention": retention,
                "crop_fraction": crop_fraction,
            })

df = pd.DataFrame(records)

assert len(df) == 1707
assert df["image_id"].is_unique

output = Path(
    "reports/segmentation/small_unet_fold1_seed123/"
    "roi_retention_audit.csv"
)

if output.exists():
    raise FileExistsError(output)

df.to_csv(output, index=False)

roi = df[df["status"] == "roi"]

print("\nROI audit — Fold 1")
print("Images:", len(df))
print("\nCrop statuses:")
print(df["status"].value_counts())

print("\nROI images:", len(roi))
print("ROI retention < 0.95:", (roi["retention"] < 0.95).sum())
print("ROI retention < 0.80:", (roi["retention"] < 0.80).sum())
print("ROI crop area >= 80%:", (roi["crop_fraction"] >= 0.80).sum())

print(
    "Large reference masks (>=60%):",
    (df["reference_area"] >= 0.60).sum()
)

print("\nWorst ROI retention:")
print(
    roi.nsmallest(10, "retention")[
        ["image_id", "dice", "retention", "crop_fraction"]
    ].to_string(index=False)
)
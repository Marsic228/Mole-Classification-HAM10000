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
from io import BytesIO
from zipfile import ZipFile

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

with ZipFile("Masks.zip") as archive, torch.inference_mode():
    for images, targets, image_ids in loader:
        logits = model(images.to(device))

        dice, iou = segmentation_metrics(
            logits, targets.to(device)
        )

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).cpu().numpy()

        for j, image_id in enumerate(image_ids):

            predicted = predictions[j, 0]

            mask_name = Path(mapping.loc[image_id]).name

            with Image.open(
                BytesIO(archive.read(f"Masks/{mask_name}"))
            ) as source:
                ref_pil = source.convert("L")

            image_path = Path("data/images") / f"{image_id}.jpg"

            with Image.open(image_path) as source:
                assert source.size == ref_pil.size

            pred_pil = Image.fromarray(
                predicted.astype(np.uint8) * 255
            ).resize(
                ref_pil.size,
                Image.Resampling.NEAREST
            )

            reference = np.asarray(ref_pil) > 127
            predicted_original = np.asarray(pred_pil) > 127

            _, reference_status = crop_lesion_with_margin(
                ref_pil, ref_pil
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
                "predicted_area": predicted_original.mean(),
                "reference_status": reference_status,
                "status": status,
                "retention": retention,
                "crop_fraction": crop_fraction,
            })

df = pd.DataFrame(records)

assert len(df) == 1707
assert df["image_id"].is_unique

output = Path(
    "reports/segmentation/small_unet_fold1_seed123/"
    "roi_retention_audit_original_resolution.csv"
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

print("\nReference vs predicted ROI status:")
print(
    pd.crosstab(
        df["reference_status"],
        df["status"],
        margins=True
    )
)

print("\nSubstantial predicted crops (<80% image area):")
print(
    ((df["status"] == "roi") &
     (df["crop_fraction"] < 0.80)).sum()
)
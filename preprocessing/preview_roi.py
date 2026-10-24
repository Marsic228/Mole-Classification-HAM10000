
import io
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from preprocessing.roi_crop import crop_lesion_with_margin

dev = pd.read_csv("splits/ham10000_dev_cv_v1.csv")
mapping = pd.read_csv("Masks_to_IsicId_mapping.csv")

samples = dev.merge(
    mapping,
    left_on="image_id",
    right_on="ISIC_ID",
    validate="one_to_one"
).groupby("dx").sample(n=1, random_state=42)

fig, axes = plt.subplots(7, 2, figsize=(10, 21))

with ZipFile("Masks.zip") as archive:
    files = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".png")
    }

    for i, row in enumerate(samples.itertuples()):
        image = Image.open(
            f"data/images/{row.image_id}.jpg"
        ).convert("RGB")

        mask = Image.open(io.BytesIO(
            archive.read(files[Path(row.Filename).name])
        )).convert("L")

        cropped, status = crop_lesion_with_margin(
            image, mask
        )

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"{row.dx} — Original")

        axes[i, 1].imshow(cropped)
        axes[i, 1].set_title(f"ROI — {status}")

        for ax in axes[i]:
            ax.axis("off")

        print(
            row.dx,
            status,
            "Original:", image.size,
            "Crop:", cropped.size
        )

plt.tight_layout()
plt.show()


records = []

with ZipFile("Masks.zip") as archive:
    files = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".png")
    }

    for row in dev.merge(
        mapping,
        left_on="image_id",
        right_on="ISIC_ID",
        validate="one_to_one"
    ).itertuples():

        with Image.open(
            f"data/images/{row.image_id}.jpg"
        ) as source:
            image = source.convert("RGB")

        mask = Image.open(io.BytesIO(
            archive.read(files[Path(row.Filename).name])
        )).convert("L")

        cropped, status = crop_lesion_with_margin(
            image, mask
        )

        assert cropped.width > 0 and cropped.height > 0
        assert cropped.width <= image.width
        assert cropped.height <= image.height

        records.append({
            "image_id": row.image_id,
            "dx": row.dx,
            "status": status,
        })

report = pd.DataFrame(records)

print("Processed:", len(report))
print("\nCrop status:")
print(report["status"].value_counts())

print("\nStatus distribution by class (%):")
print(
    (pd.crosstab(
        report["dx"],
        report["status"],
        normalize="index"
    ) * 100).round(1)
)
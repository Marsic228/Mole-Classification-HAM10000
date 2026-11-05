
import io
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from scipy.ndimage import label

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

        binary = np.asarray(mask) > 127

        components, component_count = label(
            binary,
            structure=np.ones((3, 3), dtype=bool)
        )

        sizes = np.bincount(components.ravel())[1:]

        secondary_share = (
            1 - sizes.max() / sizes.sum()
            if component_count > 0 else 0.0
        )

        cropped, status = crop_lesion_with_margin(
            image, mask
        )


        largest_only_status = None
        crop_area_ratio = None

        if component_count > 1:
            largest_id = np.argmax(sizes) + 1

            largest_mask = Image.fromarray(
                ((components == largest_id) * 255).astype(np.uint8)
            )

            alternative_crop, largest_only_status = crop_lesion_with_margin(
                image, largest_mask
            )

            crop_area_ratio = (
                alternative_crop.width * alternative_crop.height
                / (cropped.width * cropped.height)
            )

        assert cropped.width > 0 and cropped.height > 0
        assert cropped.width <= image.width
        assert cropped.height <= image.height

        records.append({
            "image_id": row.image_id,
            "dx": row.dx,
            "status": status,
            "component_count": component_count,
            "secondary_share": secondary_share,
            "largest_only_status": largest_only_status,
            "crop_area_ratio": crop_area_ratio,
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

multi = report[report["component_count"] > 1]

print(f"Multiple components: {len(multi)} / {len(report)}")

print(
    multi[["image_id", "dx", "component_count"]]
    .sort_values("component_count", ascending=False)
    .head(15)
)

ids = [
    "ISIC_0031268",
    "ISIC_0025399",
    "ISIC_0029408",
    "ISIC_0024315",
    "ISIC_0025628",
    "ISIC_0024675",
]

mask_names = mapping.set_index("ISIC_ID")["Filename"]

fig, axes = plt.subplots(len(ids), 4, figsize=(16, 18))

with ZipFile("Masks.zip") as archive:
    files = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".png")
    }

    for i, image_id in enumerate(ids):
        with Image.open(f"data/images/{image_id}.jpg") as source:
            image = source.convert("RGB")

        mask_path = files[Path(mask_names.loc[image_id]).name]

        with Image.open(io.BytesIO(archive.read(mask_path))) as source:
            mask = source.convert("L")

        binary = np.asarray(mask) > 127
        components, count = label(
            binary,
            structure=np.ones((3, 3), dtype=bool)
        )

        cropped, status = crop_lesion_with_margin(image, mask)

        sizes = np.bincount(components.ravel())[1:]
        largest_id = np.argmax(sizes) + 1

        largest_mask = Image.fromarray(
            ((components == largest_id) * 255).astype(np.uint8)
        )

        alternative_crop, alternative_status = crop_lesion_with_margin(
            image, largest_mask
        )

        axes[i, 0].imshow(image)
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(
            np.ma.masked_where(components == 0, components),
            cmap="tab10",
            alpha=0.5,
            vmin=0,
            vmax=10,
        )
        axes[i, 2].imshow(cropped)

        axes[i, 3].imshow(alternative_crop)
        axes[i, 3].set_title(f"Largest only — {alternative_status}")

        axes[i, 0].set_title(f"{image_id} — Original")
        axes[i, 1].set_title(f"{count} components")
        axes[i, 2].set_title(f"Crop — {status}")

        for ax in axes[i]:
            ax.axis("off")

plt.tight_layout()
plt.show()

print("\nMulti-component crop statuses:")
print(pd.crosstab(
    report["component_count"] > 1,
    report["status"]
))

print("\nLargest secondary shares:")
print(
    report[report["component_count"] > 1]
    .sort_values("secondary_share", ascending=False)
    [["image_id", "dx", "status", "secondary_share"]]
    .head(15)
    .to_string(index=False)
)

affected = report[report["component_count"] > 1]

print("\nOriginal vs largest-only status:")
print(pd.crosstab(
    affected["status"],
    affected["largest_only_status"]
))

print("\nStatus changes:",
      (affected["status"] != affected["largest_only_status"]).sum())

print("\nLargest crop-area reductions:")
print(
    affected[
        ["image_id", "dx", "status",
         "largest_only_status", "crop_area_ratio"]
    ]
    .sort_values("crop_area_ratio")
    .head(15)
    .to_string(index=False)
)

import math

import numpy as np


def crop_lesion_with_margin(
    image,
    mask,
    margin=0.15,
    min_fraction=0.35,
    max_area=0.60,
):
    if image.size != mask.size:
        raise ValueError("Image and mask sizes differ")

    width, height = image.size
    binary = np.asarray(mask.convert("L")) > 127

    ys, xs = np.where(binary)

    if len(xs) == 0:
        return image.copy(), "fallback_empty"

    area = binary.mean()

    if area > max_area:
        return image.copy(), "fallback_large_area"

    if (
        xs.min() == 0
        or xs.max() == width - 1
        or ys.min() == 0
        or ys.max() == height - 1
    ):
        return image.copy(), "fallback_edge"

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    crop_width = max(
        (x1 - x0) * (1 + 2 * margin),
        width * min_fraction,
    )
    crop_height = max(
        (y1 - y0) * (1 + 2 * margin),
        height * min_fraction,
    )

    # Preserve the original image aspect ratio.
    aspect = width / height
    crop_width = max(crop_width, crop_height * aspect)
    crop_height = crop_width / aspect

    if crop_width > width or crop_height > height:
        return image.copy(), "fallback_large_crop"

    crop_width = math.ceil(crop_width)
    crop_height = math.ceil(crop_height)

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    left = round(cx - crop_width / 2)
    top = round(cy - crop_height / 2)

    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))

    cropped = image.crop((
        left,
        top,
        left + crop_width,
        top + crop_height,
    ))

    return cropped, "roi"


def export_roi_dataset():
    import io
    from pathlib import Path
    from zipfile import ZipFile
    from collections import Counter

    import pandas as pd
    from PIL import Image

    dev = pd.read_csv("splits/ham10000_dev_cv_v1.csv")
    mapping = pd.read_csv("Masks_to_IsicId_mapping.csv")

    data = dev.merge(
        mapping,
        left_on="image_id",
        right_on="ISIC_ID",
        validate="one_to_one"
    )

    output_root = Path("data/processed_roi_margin15_v1")

    if output_root.exists():
        raise FileExistsError(
            f"{output_root} already exists. Inspect it first."
        )

    status_counts = Counter()

    with ZipFile("Masks.zip") as archive:
        files = {
            Path(name).name: name
            for name in archive.namelist()
            if name.lower().endswith(".png")
        }

        for row in data.itertuples(index=False):
            image_path = Path("data/images") / f"{row.image_id}.jpg"
            mask_name = files[Path(row.Filename).name]

            with Image.open(image_path) as source:
                image = source.convert("RGB")

            with Image.open(
                io.BytesIO(archive.read(mask_name))
            ) as source_mask:
                mask = source_mask.convert("L")

            cropped, status = crop_lesion_with_margin(
                image, mask
            )

            output_path = (
                output_root
                / row.source_split
                / row.dx
                / f"{row.image_id}_224x224.jpg"
            )

            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            cropped.resize((224, 224)).save(output_path)
            status_counts[status] += 1

    print("Exported:", sum(status_counts.values()))
    print("Status counts:", dict(status_counts))
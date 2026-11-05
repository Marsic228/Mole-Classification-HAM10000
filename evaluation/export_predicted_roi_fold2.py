from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.canonical_cv import get_fold_metadata
from core.segmentation_unet import SmallUNet
from preprocessing.roi_crop import crop_lesion_with_margin

output_root = Path("data/processed_roi_predicted_fold2_v1")

if output_root.exists():
    raise FileExistsError(
        f"{output_root} already exists. Do not overwrite it."
    )

_, val = get_fold_metadata(2)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "checkpoints/segmentation/small_unet_fold2_seed123_best.pt",
    map_location=device,
    weights_only=True,
)

model = SmallUNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

status_counts = Counter()

with torch.inference_mode():
    for row in val.itertuples(index=False):
        image_path = Path("data/images") / f"{row.image_id}.jpg"

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        resized = image.resize(
            (256, 192), Image.Resampling.BILINEAR
        )

        array = (
            np.asarray(resized, dtype=np.float32)
            .transpose(2, 0, 1) / 255.0
        )

        tensor = torch.from_numpy(
            array.copy()
        ).unsqueeze(0).to(device)

        predicted = (
            torch.sigmoid(model(tensor))[0, 0] >= 0.5
        ).to(torch.uint8).cpu().numpy()

        mask = Image.fromarray(
            predicted * 255
        ).resize(
            image.size, Image.Resampling.NEAREST
        )

        cropped, status = crop_lesion_with_margin(
            image, mask
        )

        destination = (
            output_root / "val" / row.dx
            / f"{row.image_id}_224x224.jpg"
        )

        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        cropped.resize((224, 224)).save(destination)

        status_counts[status] += 1

print("Exported:", sum(status_counts.values()))
print("Crop statuses:", dict(status_counts))

assert sum(status_counts.values()) == 1706
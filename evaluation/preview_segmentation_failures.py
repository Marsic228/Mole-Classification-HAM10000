import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import SegmentationDataset
from core.segmentation_unet import SmallUNet
from PIL import Image
from preprocessing.roi_crop import crop_lesion_with_margin

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_, val = get_fold_metadata(1)

mapping = pd.read_csv(
    "Masks_to_IsicId_mapping.csv"
).set_index("ISIC_ID")["Filename"]

dataset = SegmentationDataset(val, mapping)

checkpoint = torch.load(
    "checkpoints/segmentation/small_unet_fold1_seed123_best.pt",
    map_location=device,
    weights_only=True,
)

model = SmallUNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

ids = [
    "ISIC_0032096",  # nv
    "ISIC_0029926",  # nv
    "ISIC_0026343",  # bcc
    "ISIC_0027884",  # akiec
    "ISIC_0027365",  # bcc
    "ISIC_0031261",  # bkl
]

index_by_id = {
    image_id: i
    for i, image_id in enumerate(dataset.metadata["image_id"])
}

fig, axes = plt.subplots(len(ids), 4, figsize=(15, 18))

for i, image_id in enumerate(ids):
    image, target, _ = dataset[index_by_id[image_id]]

    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))
        predicted = (
            torch.sigmoid(logits)[0, 0] >= 0.5
        ).cpu().numpy()

    photo = image.permute(1, 2, 0).numpy()
    reference = target[0].numpy()

    photo_pil = Image.fromarray(
        (photo * 255).astype(np.uint8)
    )

    reference_pil = Image.fromarray(
        (reference * 255).astype(np.uint8)
    )

    predicted_pil = Image.fromarray(
        (predicted.astype(np.uint8) * 255)
    )

    _, reference_status = crop_lesion_with_margin(
        photo_pil, reference_pil
    )

    _, predicted_status = crop_lesion_with_margin(
        photo_pil, predicted_pil
    )

    print(
        f"{image_id}: "
        f"reference_area={reference.mean():.3f}, "
        f"predicted_area={predicted.mean():.3f}, "
        f"reference_status={reference_status}, "
        f"predicted_status={predicted_status}"
    )

    reference_inside_crop, status = crop_lesion_with_margin(
        reference_pil,
        predicted_pil,
    )

    total_reference = np.count_nonzero(reference > 0.5)
    retained_reference = np.count_nonzero(
        np.asarray(reference_inside_crop) > 127
    )

    retention = (
        retained_reference / total_reference
        if total_reference > 0 else float("nan")
    )

    crop_fraction = (
        reference_inside_crop.width * reference_inside_crop.height
        / (reference_pil.width * reference_pil.height)
    )

    print(
        f"{image_id}: "
        f"status={status}, "
        f"reference_retained={retention:.3f}, "
        f"crop_fraction={crop_fraction:.3f}"
    )

    axes[i, 0].imshow(photo)
    axes[i, 1].imshow(reference, cmap="gray", vmin=0, vmax=1)
    axes[i, 2].imshow(predicted, cmap="gray", vmin=0, vmax=1)

    axes[i, 3].imshow(photo)
    axes[i, 3].contour(
        reference, levels=[0.5], colors="lime"
    )
    axes[i, 3].contour(
        predicted.astype(float),
        levels=[0.5], colors="red"
    )

    for j, title in enumerate([
        f"{image_id} — Original",
        "Reference",
        "Prediction",
        "Green: reference | Red: prediction",
    ]):
        axes[i, j].set_title(title, fontsize=9)
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()
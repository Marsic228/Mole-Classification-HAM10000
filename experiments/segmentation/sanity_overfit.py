import pandas as pd
import torch

from torch.utils.data import DataLoader
from core.canonical_cv import get_fold_metadata
from core.cross_validation_dataset import SegmentationDataset
from core.segmentation_unet import SmallUNet
from core.segmentation_losses import segmentation_loss
from evaluation.segmentation_metrics import segmentation_metrics


torch.manual_seed(123)

train, _ = get_fold_metadata(1)

mapping = pd.read_csv(
    "Masks_to_IsicId_mapping.csv"
).set_index("ISIC_ID")["Filename"]

dataset = SegmentationDataset(train, mapping)

loader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0
)

images, masks, image_ids = next(iter(loader))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

images = images.to(device)
masks = masks.to(device)

model = SmallUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1, 51):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = segmentation_loss(logits, masks)

    loss.backward()
    optimizer.step()

    if step in (1, 10, 25, 50):
        with torch.no_grad():
            dice, iou = segmentation_metrics(logits, masks)
            foreground = (
                torch.sigmoid(logits) >= 0.5
            ).float().mean()

        print(
            f"Step {step}: "
            f"loss={loss.item():.4f}, "
            f"dice={dice.mean().item():.4f}, "
            f"iou={iou.mean().item():.4f}, "
            f"predicted_foreground={foreground.item():.4f}"
        )
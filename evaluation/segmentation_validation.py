import torch

from evaluation.segmentation_metrics import segmentation_metrics


def evaluate_segmentation(model, loader, device, max_batches=None):
    model.eval()

    records = []

    with torch.inference_mode():
        for batch_idx, (images, targets, image_ids) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            dice, iou = segmentation_metrics(logits, targets)

            predicted = torch.sigmoid(logits) >= 0.5
            predicted_empty = (
                predicted.flatten(1).sum(dim=1) == 0
            )

            for image_id, d, i, empty in zip(
                image_ids,
                dice.cpu().tolist(),
                iou.cpu().tolist(),
                predicted_empty.cpu().tolist(),
            ):
                records.append({
                    "image_id": image_id,
                    "dice": d,
                    "iou": i,
                    "predicted_empty": empty,
                })

    if not records:
        raise ValueError("Validation produced no records")

    summary = {
        "n_images": len(records),
        "mean_dice": sum(r["dice"] for r in records) / len(records),
        "mean_iou": sum(r["iou"] for r in records) / len(records),
        "empty_predictions": sum(r["predicted_empty"] for r in records),
    }

    return summary, records
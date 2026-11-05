import torch


def segmentation_metrics(logits, targets):
    if logits.shape != targets.shape:
        raise ValueError("Prediction and target shapes differ")

    predicted = torch.sigmoid(logits) >= 0.5
    target = targets >= 0.5

    dims = (1, 2, 3)

    intersection = (predicted & target).sum(dim=dims).float()
    pred_area = predicted.sum(dim=dims).float()
    target_area = target.sum(dim=dims).float()

    total_area = pred_area + target_area
    union = total_area - intersection

    dice = torch.where(
        total_area > 0,
        2 * intersection / total_area.clamp_min(1),
        torch.ones_like(total_area),
    )

    iou = torch.where(
        union > 0,
        intersection / union.clamp_min(1),
        torch.ones_like(union),
    )

    return dice, iou
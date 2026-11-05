import torch
import torch.nn.functional as F


def soft_dice_loss(logits, targets, eps=1e-6):
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    total = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2 * intersection + eps
    ) / (total + eps)

    return 1 - dice.mean()


def segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(
        logits, targets
    )

    dice = soft_dice_loss(logits, targets)

    return bce + dice
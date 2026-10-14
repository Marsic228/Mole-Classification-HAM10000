import torch
import torch.nn as nn
from pathlib import Path

from PIL import Image
from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


def load_convnext_model(checkpoint_path, num_classes):
    model = convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(
        in_features,
        num_classes
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


weights = ConvNeXt_Tiny_Weights.DEFAULT
transform = weights.transforms()


def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = probabilities.max(dim=1)

    return {
        "predicted_class": CLASS_NAMES[predicted_index.item()],
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze(0).tolist(),
    }


if __name__ == "__main__":
    model = load_convnext_model(
        "checkpoints/convnext_tiny_finetune_v3_fold1_best.pt",
        len(CLASS_NAMES)
    )

    image_path = next(Path("data/processed").rglob("*.jpg"))

    print("Test image:", image_path)

    result = predict_image(
        model,
        image_path
    )

    print("Predicted class:", result["predicted_class"])
    print("Confidence:", result["confidence"])

    print("\nProbabilities:")
    for class_name, probability in zip(
        CLASS_NAMES,
        result["probabilities"]
    ):
        print(f"{class_name}: {probability:.4f}")
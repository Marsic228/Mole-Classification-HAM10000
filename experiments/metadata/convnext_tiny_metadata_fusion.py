import torch
import torch.nn as nn


class ImageOnlyClassifier(nn.Module):
    def __init__(
        self,
        image_dim=768,
        num_classes=7,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            image_dim,
            num_classes
        )

    def forward(self, image_features):
        return self.classifier(
            image_features
        )


class MetadataFusionClassifier(nn.Module):
    def __init__(
        self,
        image_dim=768,
        metadata_dim=20,
        num_classes=7,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            image_dim + metadata_dim,
            num_classes
        )

    def forward(
        self,
        image_features,
        metadata_features,
    ):
        combined_features = torch.cat(
            [
                image_features,
                metadata_features,
            ],
            dim=1
        )

        return self.classifier(
            combined_features
        )

class FrozenImageMetadataResidual(nn.Module):
    def __init__(
        self,
        image_dim=768,
        metadata_dim=20,
        num_classes=7,
    ):
        super().__init__()

        self.image_classifier = nn.Linear(
            image_dim,
            num_classes
        )

        self.metadata_branch = nn.Linear(
            metadata_dim,
            num_classes,
            bias=False
        )

        # Metadata starts with zero influence.
        nn.init.zeros_(
            self.metadata_branch.weight
        )

        # Image classifier must never change.
        for parameter in self.image_classifier.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        image_features,
        metadata_features,
    ):
        image_logits = self.image_classifier(
            image_features
        )

        metadata_delta = self.metadata_branch(
            metadata_features
        )

        return image_logits + metadata_delta

if __name__ == "__main__":
    image_features = torch.randn(8, 768)
    metadata_features = torch.randn(8, 20)

    image_model = ImageOnlyClassifier()
    fusion_model = MetadataFusionClassifier()

    image_logits = image_model(
        image_features
    )

    fusion_logits = fusion_model(
        image_features,
        metadata_features
    )

    print(
        "Image features:",
        image_features.shape
    )

    print(
        "Metadata features:",
        metadata_features.shape
    )

    print(
        "Image-only logits:",
        image_logits.shape
    )

    print(
        "Fusion logits:",
        fusion_logits.shape
    )
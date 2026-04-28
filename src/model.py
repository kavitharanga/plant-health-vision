import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Fine-tune EfficientNet-B0 for plant disease classification.

    Args:
        num_classes:     Number of output classes (16 for your dataset)
        freeze_backbone: Freeze pretrained layers initially

    Returns:
        model: EfficientNet-B0 with custom classifier head
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze backbone layers
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    NUM_CLASSES = 15
    model = build_model(num_classes=NUM_CLASSES, freeze_backbone=True)

    # Quick sanity check
    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 images
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expected: torch.Size([4, 16])
    print(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")

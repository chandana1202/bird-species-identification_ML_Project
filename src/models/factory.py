import torch.nn as nn
from torchvision import models

from src.models.baseline_cnn import BaselineCNN


def build_model(name, num_classes=200, pretrained=True):
    """Return (model, head_module)."""

    if name == "baseline":
        model = BaselineCNN(num_classes)
        return model, model.classifier

    if name == "resnet50":
        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(weights=weights)

        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(
                model.fc.in_features,
                num_classes
            )
        )

        return model, model.fc

    if name == "efficientnet_b0":
        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_b0(weights=weights)

        in_f = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )

        return model, model.classifier

    raise ValueError(f"Unknown model: {name}")
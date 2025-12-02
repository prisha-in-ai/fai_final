from typing import Optional

import torch
from torch import nn
from torchvision import models


def build_resnet50_multilabel(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-50 backbone with a small dense head for multi-label prediction.

    The final head is:
        FC(in_features -> 256) -> ReLU -> Dropout(0.3) -> FC(256 -> num_classes)

    The outputs are raw logits (no sigmoid).
    """
    if pretrained:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
        except AttributeError:
            # older torchvision versions
            resnet = models.resnet50(pretrained=True)
    else:
        resnet = models.resnet50(weights=None)

    in_features = resnet.fc.in_features

    resnet.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return resnet


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Optionally freeze all backbone layers except the final classification head.
    """
    if not freeze:
        return

    for name, param in model.named_parameters():
        # keep the head trainable
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = False

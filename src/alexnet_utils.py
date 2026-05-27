from __future__ import annotations

import torch
import torch.nn as nn

ALEXNET_LAYER_MAP = {
    "features.0 Conv1": "features.0",
    "features.1 ReLU1": "features.1",
    "features.2 MaxPool1": "features.2",
    "features.3 Conv2": "features.3",
    "features.5 MaxPool2": "features.5",
    "features.6 Conv3": "features.6",
    "features.8 Conv4": "features.8",
    "features.10 Conv5": "features.10",
    "features.12 MaxPool5": "features.12",
    "classifier.1 FC6": "classifier.1",
    "classifier.4 FC7": "classifier.4",
    "classifier.6 Output": "classifier.6",
}


def load_alexnet(device: str):
    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights).eval().to(device)
    return model, weights

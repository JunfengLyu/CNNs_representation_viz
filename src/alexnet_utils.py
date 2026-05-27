from __future__ import annotations

import torch
import torch.nn as nn

ALEXNET_LAYER_MAP = {
    "features.0 Conv1: broad edge and color filters": "features.0",
    "features.1 ReLU1: positive activations": "features.1",
    "features.2 MaxPool1: smaller spatial map": "features.2",
    "features.3 Conv2: textures and color blobs": "features.3",
    "features.5 MaxPool2: further compression": "features.5",
    "features.6 Conv3: local object parts": "features.6",
    "features.8 Conv4: richer part combinations": "features.8",
    "features.10 Conv5: compact semantic maps": "features.10",
    "features.12 MaxPool5: final spatial compression": "features.12",
    "classifier.1 FC6: abstract image vector": "classifier.1",
    "classifier.4 FC7: class-oriented vector": "classifier.4",
    "classifier.6 Output: ImageNet class evidence": "classifier.6",
}


def load_alexnet(device: str):
    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights).eval().to(device)
    return model, weights

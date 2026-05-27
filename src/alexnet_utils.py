from __future__ import annotations

import torch
import torch.nn as nn

ALEXNET_LAYER_MAP = {
    "features.0 卷积1：大范围边缘和颜色滤波器": "features.0",
    "features.1 ReLU1：保留正向激活": "features.1",
    "features.2 最大池化1：缩小空间尺寸": "features.2",
    "features.3 卷积2：纹理和颜色块": "features.3",
    "features.5 最大池化2：进一步压缩": "features.5",
    "features.6 卷积3：物体局部结构": "features.6",
    "features.8 卷积4：更复杂的局部组合": "features.8",
    "features.10 卷积5：较紧凑的语义图": "features.10",
    "features.12 最大池化5：最后的空间压缩": "features.12",
    "classifier.1 全连接6：抽象图像向量": "classifier.1",
    "classifier.4 全连接7：更接近类别的向量": "classifier.4",
    "classifier.6 输出层：ImageNet 类别证据": "classifier.6",
}


def load_alexnet(device: str):
    from torchvision.models import alexnet, AlexNet_Weights

    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights).eval().to(device)
    return model, weights

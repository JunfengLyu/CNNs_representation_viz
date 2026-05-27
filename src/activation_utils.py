from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps


def capture_activation(model: nn.Module, module_name: str, x: torch.Tensor):
    """Run model once and return logits plus the chosen module output."""
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"Unknown module: {module_name}")
    activation = {}

    def hook(_module, _inp, out):
        activation["value"] = out.detach().cpu()

    handle = modules[module_name].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        handle.remove()
    return logits.detach().cpu(), activation["value"]


def to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in 0..1 to HWC numpy."""
    arr = t.detach().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def preprocess_canvas_rgba(rgba: np.ndarray) -> Tuple[Image.Image, torch.Tensor]:
    """Convert drawable-canvas RGBA image to a 28x28 LeNet input tensor.

    Canvas is configured as white ink on a black background. Alpha and RGB are
    collapsed to a grayscale digit image in [0, 1].
    """
    img = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    gray = img.convert("L")
    bbox = gray.point(lambda p: 255 if p > 20 else 0).getbbox()
    if bbox is not None:
        gray = gray.crop(bbox)
        # Add square padding before resizing, to mimic MNIST-like centered digits.
        w, h = gray.size
        side = max(w, h)
        square = Image.new("L", (side, side), 0)
        square.paste(gray, ((side - w) // 2, (side - h) // 2))
        gray = square
    gray = gray.resize((20, 20), Image.Resampling.BILINEAR)
    canvas28 = Image.new("L", (28, 28), 0)
    canvas28.paste(gray, (4, 4))
    arr = np.asarray(canvas28, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, None, :, :]
    return canvas28, tensor


def preprocess_image_for_alexnet(img: Image.Image, weights) -> Tuple[Image.Image, torch.Tensor]:
    """Use torchvision's official AlexNet preprocessing when weights exist."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    transform = weights.transforms()
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def activation_channel_image(act: torch.Tensor, channel: int) -> Tuple[np.ndarray, str]:
    """Return a 2D visualization image and a short description.

    For conv activations [B,C,H,W], show one channel map.
    For vector activations [B,N], return the whole vector as a 1xN heatmap and
    highlight selected channel in the description.
    """
    a = act[0]
    if a.ndim == 3:
        c = int(np.clip(channel, 0, a.shape[0] - 1))
        return a[c].numpy(), f"Spatial activation map for channel {c} of {a.shape[0]}"
    if a.ndim == 1:
        c = int(np.clip(channel, 0, a.shape[0] - 1))
        return a.numpy()[None, :], f"Vector activation; unit {c} value = {float(a[c]):.4f}"
    return a.squeeze().numpy(), "Activation"


def available_channels(act: torch.Tensor) -> int:
    a = act[0]
    if a.ndim == 3:
        return int(a.shape[0])
    if a.ndim == 1:
        return int(a.shape[0])
    return 1

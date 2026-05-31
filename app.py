from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from src.activation_utils import (
    activation_channel_image,
    available_channels,
    capture_activation,
    normalize01,
    preprocess_canvas_rgba,
    preprocess_image_for_alexnet,
)
from src.alexnet_utils import ALEXNET_LAYER_MAP, load_alexnet
from src.lenet_model import LENET_LAYER_MAP, LeNet5

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

ROOT = Path(__file__).resolve().parent
LENET_WEIGHTS = ROOT / "assets" / "lenet_digits_28.pt"
DEMO_IMAGE = ROOT / "ImageNet_demo.JPEG"
IMAGENET_SAMPLE_REPO = "https://github.com/EliSchwartz/imagenet-sample-images"

st.set_page_config(page_title="CNNs Representation Viz", layout="wide")
st.title("CNNs Representation Viz")
st.caption("Use the controls below to run the interface.")


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource
def load_lenet(device: str):
    model = LeNet5().eval().to(device)
    if LENET_WEIGHTS.exists():
        state = torch.load(LENET_WEIGHTS, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    else:
        st.warning("LeNet weights file not found. The model will run with random weights.")
    return model


@st.cache_resource(show_spinner="Loading AlexNet. The first run may download pretrained weights...")
def cached_alexnet(device: str):
    return load_alexnet(device)


def plot_activation(arr: np.ndarray, title: str, height: float = 3.2):
    fig, ax = plt.subplots(figsize=(height * 1.25, height))
    im = ax.imshow(arr, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def plot_overlay(base_img: Image.Image, act2d: np.ndarray, title: str):
    base = base_img.convert("RGB")
    act = Image.fromarray((normalize01(act2d) * 255).astype(np.uint8)).resize(base.size, Image.Resampling.BILINEAR)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(base)
    ax.imshow(act, cmap="viridis", alpha=0.48)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def topk_text(logits: torch.Tensor, k: int = 5, labels=None):
    prob = F.softmax(logits[0], dim=0)
    vals, idxs = torch.topk(prob, k)
    rows = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        name = str(i) if labels is None else labels[i]
        rows.append({"class": name, "probability": round(float(v), 4)})
    return rows


def lenet_mode(device: str):
    st.header("Mode 1: Handwritten digit + LeNet")
    st.info("How to use: draw a white digit, choose a LeNet layer, then move the channel slider.")
    if st_canvas is None:
        st.error("streamlit-drawable-canvas is missing. Run: pip install streamlit-drawable-canvas")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
        )

    if canvas_result.image_data is None:
        st.info("Draw a digit to begin.")
        return

    digit_img, x = preprocess_canvas_rgba(canvas_result.image_data)
    model = load_lenet(device)
    x = x.to(device)

    layer_label = st.selectbox("Choose a LeNet layer", list(LENET_LAYER_MAP.keys()), index=0)
    module_name = LENET_LAYER_MAP[layer_label]
    logits, act = capture_activation(model, module_name, x)
    n_ch = available_channels(act)
    channel = st.slider("Channel / unit index", 0, max(0, n_ch - 1), 0)
    act2d, desc = activation_channel_image(act, channel)

    with c2:
        st.image(digit_img.resize((140, 140), Image.Resampling.NEAREST), caption="28x28 model input", width=180)
        pred = int(torch.argmax(logits[0]).item())
        st.metric("Predicted digit", pred)
        st.dataframe(topk_text(logits, k=5, labels=[str(i) for i in range(10)]), hide_index=True, use_container_width=True)

    st.subheader("Selected representation")
    v1, v2 = st.columns([1, 1])
    with v1:
        plot_activation(act2d, f"{layer_label}\n{desc}")
    with v2:
        if act2d.ndim == 2 and min(act2d.shape) > 1:
            plot_overlay(digit_img.convert("RGB"), act2d, "Activation overlay on digit input")
        else:
            st.info("Fully connected layers are vectors, not spatial maps. The heatmap shows the full activation vector.")

    st.caption(f"Activation tensor shape: {tuple(act.shape)}")


def alexnet_mode(device: str):
    st.header("Mode 2: Natural image + AlexNet")
    st.info("How to use: keep the demo image or upload an image, choose an AlexNet layer, then move the channel slider.")
    st.markdown(f"Sample ImageNet images: [{IMAGENET_SAMPLE_REPO}]({IMAGENET_SAMPLE_REPO})")
    uploaded = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is None:
        if not DEMO_IMAGE.exists():
            st.info("Upload an image to begin.")
            return
        img = Image.open(DEMO_IMAGE)
        source_caption = "Demo image"
    else:
        img = Image.open(uploaded)
        source_caption = "Uploaded image"

    try:
        model, weights = cached_alexnet(device)
    except Exception as e:
        st.error("AlexNet failed to load. This often means the pretrained-weight download was interrupted.")
        st.code(str(e))
        st.write("For local runs, clear the PyTorch cache and retry:")
        st.code("rm -rf ~/.cache/torch/checkpoints/*\npython -m streamlit run app.py", language="bash")
        return

    original, model_input, x = preprocess_image_for_alexnet(img, weights)
    x = x.to(device)

    a, b = st.columns([1, 1])
    with a:
        st.image(original, caption=source_caption, use_container_width=True)
        st.image(model_input, caption="AlexNet model input (224x224 center crop)", use_container_width=True)
    with b:
        layer_label = st.selectbox("Choose an AlexNet layer", list(ALEXNET_LAYER_MAP.keys()), index=0)
        module_name = ALEXNET_LAYER_MAP[layer_label]
        logits, act = capture_activation(model, module_name, x)
        labels = weights.meta.get("categories")
        st.write("Top predictions")
        st.dataframe(topk_text(logits, k=5, labels=labels), hide_index=True, use_container_width=True)
        n_ch = available_channels(act)
        channel = st.slider("Channel / unit index", 0, max(0, n_ch - 1), 0)

    act2d, desc = activation_channel_image(act, channel)
    st.subheader("Selected representation")
    v1, v2 = st.columns([1, 1])
    with v1:
        plot_activation(act2d, f"{layer_label}\n{desc}")
    with v2:
        if act2d.ndim == 2 and min(act2d.shape) > 1:
            plot_overlay(model_input, act2d, "Activation overlay on AlexNet input")
        else:
            st.info("Fully connected layers are vectors. The heatmap shows the full vector rather than an image-like spatial map.")
    st.caption(f"Activation tensor shape: {tuple(act.shape)}")


def main():
    device = get_device()
    st.sidebar.success(f"Device: {device}")
    st.sidebar.markdown("### How to use")
    st.sidebar.write("1. Choose a mode.")
    st.sidebar.write("2. Provide an input image.")
    st.sidebar.write("3. Select a layer.")
    st.sidebar.write("4. Adjust the channel or unit index.")
    mode = st.sidebar.radio(
        "Choose a mode",
        ["Handwritten digit + LeNet", "Natural image + AlexNet"],
    )
    if mode == "Handwritten digit + LeNet":
        lenet_mode(device)
    else:
        alexnet_mode(device)


if __name__ == "__main__":
    main()

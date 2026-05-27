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

st.set_page_config(page_title="早期 CNN 表征探索实验", layout="wide")
st.title("早期 CNN 表征探索实验")
st.caption("通过 LeNet 手写数字和 AlexNet 自然图像，观察神经网络在不同层级中形成的表征。")


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


@st.cache_resource(show_spinner="正在加载 AlexNet。首次运行会自动下载预训练权重，可能需要一点时间...")
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
        rows.append({"类别": name, "概率": round(float(v), 4)})
    return rows


def lenet_mode(device: str):
    st.header("模式一：手写数字 + LeNet")
    st.write("请在黑色画布上用白色笔迹写一个数字，然后选择网络层和通道，观察模型如何逐层处理这个数字。")
    if st_canvas is None:
        st.error("缺少 streamlit-drawable-canvas 依赖。请运行：pip install streamlit-drawable-canvas")
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
        st.info("写一个数字后即可开始。")
        return

    digit_img, x = preprocess_canvas_rgba(canvas_result.image_data)
    model = load_lenet(device)
    x = x.to(device)

    layer_label = st.selectbox("选择 LeNet 层", list(LENET_LAYER_MAP.keys()), index=0)
    module_name = LENET_LAYER_MAP[layer_label]
    logits, act = capture_activation(model, module_name, x)
    n_ch = available_channels(act)
    channel = st.slider("通道 / 单元编号", 0, max(0, n_ch - 1), 0)
    act2d, desc = activation_channel_image(act, channel)

    with c2:
        st.image(digit_img.resize((140, 140), Image.Resampling.NEAREST), caption="模型看到的 28x28 输入", width=180)
        pred = int(torch.argmax(logits[0]).item())
        st.metric("模型预测的数字", pred)
        st.dataframe(topk_text(logits, k=5, labels=[str(i) for i in range(10)]), hide_index=True, use_container_width=True)

    st.subheader("当前选择的表征")
    v1, v2 = st.columns([1, 1])
    with v1:
        plot_activation(act2d, f"{layer_label}\n{desc}")
    with v2:
        if act2d.ndim == 2 and min(act2d.shape) > 1:
            plot_overlay(digit_img.convert("RGB"), act2d, "激活叠加到数字输入上")
        else:
            st.info("全连接层是向量，没有空间位置图；左侧热图展示的是整条激活向量。")

    st.caption(f"激活张量形状：{tuple(act.shape)}")


def alexnet_mode(device: str):
    st.header("模式二：自然图像 + AlexNet")
    st.write("可以直接使用示例图，也可以上传一张自己的图片，观察 AlexNet 对自然图像的逐层表征。")
    uploaded = st.file_uploader("上传图片（可选）", type=["png", "jpg", "jpeg", "webp"])

    if uploaded is None:
        if not DEMO_IMAGE.exists():
            st.info("请上传一张图片后开始。")
            return
        img = Image.open(DEMO_IMAGE)
        source_caption = "示例图片"
    else:
        img = Image.open(uploaded)
        source_caption = "上传的图片"

    try:
        model, weights = cached_alexnet(device)
    except Exception as e:
        st.error("AlexNet 加载失败。常见原因是首次下载预训练权重时网络中断。")
        st.code(str(e))
        st.write("本地运行时可以删除 PyTorch 缓存后重试：")
        st.code("rm -rf ~/.cache/torch/checkpoints/*\npython -m streamlit run app.py", language="bash")
        return

    original, x = preprocess_image_for_alexnet(img, weights)
    x = x.to(device)

    a, b = st.columns([1, 1])
    with a:
        st.image(original, caption=source_caption, use_container_width=True)
    with b:
        layer_label = st.selectbox("选择 AlexNet 层", list(ALEXNET_LAYER_MAP.keys()), index=0)
        module_name = ALEXNET_LAYER_MAP[layer_label]
        logits, act = capture_activation(model, module_name, x)
        labels = weights.meta.get("categories")
        st.write("模型预测结果")
        st.dataframe(topk_text(logits, k=5, labels=labels), hide_index=True, use_container_width=True)
        n_ch = available_channels(act)
        channel = st.slider("通道 / 单元编号", 0, max(0, n_ch - 1), 0)

    act2d, desc = activation_channel_image(act, channel)
    st.subheader("当前选择的表征")
    v1, v2 = st.columns([1, 1])
    with v1:
        plot_activation(act2d, f"{layer_label}\n{desc}")
    with v2:
        if act2d.ndim == 2 and min(act2d.shape) > 1:
            plot_overlay(original, act2d, "激活叠加到原图上")
        else:
            st.info("全连接层是向量。左侧热图展示的是整条向量，而不是类似图像的空间图。")
    st.caption(f"激活张量形状：{tuple(act.shape)}")


def main():
    device = get_device()
    st.sidebar.success(f"运行设备：{device}")
    st.sidebar.markdown("### 探索提示")
    st.sidebar.write("早期卷积层常保留边缘、笔画和纹理；越深的层通常越抽象，也越接近分类任务。")
    mode = st.sidebar.radio(
        "选择模式",
        ["手写数字 + LeNet", "自然图像 + AlexNet"],
    )
    if mode == "手写数字 + LeNet":
        lenet_mode(device)
    else:
        alexnet_mode(device)


if __name__ == "__main__":
    main()

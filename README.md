# CNNs Representation Viz

这是一个用于期末题自主探索的 Streamlit 在线应用。学生无需安装 Python 或配置深度学习环境，只需要打开网页，就可以观察 LeNet 和 AlexNet 在不同层级中的神经网络表征。

应用包含两个模式：

1. **手写数字 + LeNet**
   - 学生在网页画布上写一个数字。
   - LeNet 风格的卷积神经网络会预测该数字。
   - 学生可以选择不同网络层和通道，观察模型内部激活。

2. **自然图像 + AlexNet**
   - 学生可以使用默认示例图，也可以上传自己的图片。
   - 预训练 AlexNet 会进行 ImageNet 分类。
   - 学生可以选择不同网络层和通道，比较早期、中期和后期表征的变化。

## 适合课堂讨论的问题

- 为什么早期卷积层更像原始图片中的边缘、笔画或纹理？
- 为什么更深层的表征越来越难直接看懂？
- 一个卷积通道可以理解为在寻找什么类型的视觉特征？
- 全连接层为什么更适合看成向量，而不是二维图像？
- 同一张图片如何逐步变成一个分类判断？

## 本地运行

推荐使用 Python 3.10 或更新版本。Streamlit Community Cloud 会使用它当前支持的 Python 版本，因此 `requirements.txt` 中的 PyTorch 版本使用了兼容范围，而不是固定旧版本。

```bash
cd CNNs_representation_viz
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

然后打开终端中显示的本地地址，通常是：

```text
http://localhost:8501
```

## 发布到 GitHub

如果这是一个新项目，可以在本目录中运行：

```bash
git init
git add .
git commit -m "Prepare CNN representation lab for deployment"
git branch -M main
```

然后在 GitHub 网页端新建仓库 `CNNs_representation_viz`。创建后，根据 GitHub 给出的命令添加远端并推送：

```bash
git remote add origin https://github.com/<你的用户名>/CNNs_representation_viz.git
git push -u origin main
```

## 发布为在线网站

推荐使用 Streamlit Community Cloud，因为本项目依赖 Python、PyTorch 和 torchvision，不能直接部署到 GitHub Pages 这种静态网页服务。

部署步骤：

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 使用 GitHub 登录。
3. 点击 **Create app** 或 **New app**。
4. 选择刚刚推送的 GitHub 仓库。
5. Branch 选择 `main`。
6. Main file path 填写 `app.py`。
7. 点击 Deploy。

如果 Cloud 的安装日志显示某个依赖不支持当前 Python 版本，可以在部署页面的 **Advanced settings** 中把 Python version 改成 `3.12`，再重新部署。

部署完成后，Streamlit 会生成一个公开网址。把这个网址发给学生即可。

## 模型权重说明

LeNet 使用仓库中自带的权重文件：

```text
assets/lenet_digits_28.pt
```

AlexNet 使用 torchvision 官方 ImageNet 预训练权重。在线网站首次运行 AlexNet 模式时会自动下载权重，第一次打开可能稍慢。

如果本地运行时 AlexNet 权重下载中断，可以清理 PyTorch 缓存后重试：

```bash
rm -rf ~/.cache/torch/checkpoints/*
python -m streamlit run app.py
```

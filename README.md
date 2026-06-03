# CNNs Representation Viz

This Streamlit app is designed for a final-term assignment. Students can open a website and use the interface without installing Python or configuring a deep learning environment.

The app includes two modes:

1. **Handwritten digit + LeNet**
   - Draw a digit in the browser.
   - A LeNet-style convolutional neural network predicts the digit.
   - Choose a network layer and channel to view internal activations.

2. **Natural image + AlexNet**
   - Use the default image or upload your own image.
   - A pretrained AlexNet runs ImageNet classification.
   - Choose a network layer and channel to view activations.

## Run Locally

Python 3.10 or newer is recommended. Streamlit Community Cloud uses its currently supported Python versions, so the PyTorch dependency uses a compatible version range instead of an old fixed version.

```bash
cd CNNs_representation_viz
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Deploy As A Website

Streamlit Community Cloud is recommended because this project depends on Python, PyTorch, and torchvision.

Deployment steps:

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in.
3. Click **Create app** or **New app**.
4. Select the project repository.
5. Set Branch to `main`.
6. Set Main file path to `app.py`.
7. Click Deploy.

If the Cloud install log says a dependency does not support the current Python version, open **Advanced settings**, set Python version to `3.12`, save, and redeploy.

After deployment, Streamlit will create a public URL. Share that URL with students.

## Model Weights

LeNet uses the bundled weight file:

```text
assets/lenet_digits_28.pt
```

AlexNet uses the official ImageNet pretrained weights from torchvision. The first online run may take a little longer because the weights need to download.

If the AlexNet download is interrupted during local development, clear the PyTorch cache and retry:

```bash
rm -rf ~/.cache/torch/checkpoints/*
python -m streamlit run app.py
```

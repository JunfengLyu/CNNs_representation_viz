"""Optional: retrain the bundled LeNet weights from sklearn's built-in digits dataset.
This does not require internet. It is included only for transparency/reproducibility.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image
from src.lenet_model import LeNet5

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets" / "lenet_digits_28.pt"

def upscale_digit(arr8):
    # sklearn digits are bright digits on dark background, intensity 0..16
    img = Image.fromarray((arr8 / 16.0 * 255).astype(np.uint8), mode="L")
    img = img.resize((20, 20), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, (4, 4))
    return np.asarray(canvas, dtype=np.float32) / 255.0

def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    digits = load_digits()
    X = np.stack([upscale_digit(x) for x in digits.images])[:, None, :, :]
    y = digits.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    loader = DataLoader(train, batch_size=64, shuffle=True)
    model = LeNet5()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(12):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        if epoch % 5 == 0 or epoch == 11:
            model.eval()
            with torch.no_grad():
                xt, yt = test.tensors
                pred = model(xt).argmax(1)
                acc = (pred == yt).float().mean().item()
            print(f"epoch={epoch:02d} loss={total/len(train):.4f} test_acc={acc:.3f}")
    OUT.parent.mkdir(exist_ok=True, parents=True)
    torch.save({"model_state_dict": model.state_dict(), "source": "sklearn.datasets.load_digits upscaled to 28x28", "test_accuracy": acc}, OUT)
    print(f"saved {OUT}")

if __name__ == "__main__":
    main()

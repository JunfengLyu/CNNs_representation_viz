import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """A small LeNet-style CNN for 28x28 handwritten digit images."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LENET_LAYER_MAP = {
    "conv1": "conv1",
    "pool1": "pool1",
    "conv2": "conv2",
    "pool2": "pool2",
    "fc1": "fc1",
    "fc2": "fc2",
    "output layer": "fc3",
}

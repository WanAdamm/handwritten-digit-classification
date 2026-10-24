"""Neural Network architectures for MNIST classification in CST435 Parallel Processing.

Provides:
- MNISTCNN: Standard 2-layer Convolutional Neural Network (high compute-to-communication ratio).
- MNISTMLP: 3-layer Multi-Layer Perceptron matching standard feed-forward benchmark baselines.
- get_model(): Factory function to instantiate models by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    """
    2-Layer Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
        Input: (B, 1, 28, 28)
        Conv1: (B, 32, 28, 28) -> ReLU -> MaxPool(2,2) -> (B, 32, 14, 14)
        Conv2: (B, 64, 14, 14) -> ReLU -> MaxPool(2,2) -> (B, 64, 7, 7)
        Flatten: (B, 3136)
        FC1: (B, 128) -> ReLU -> Dropout(0.25)
        FC2 / Logits: (B, 10)
    """

    def __init__(self, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # Fully connected block
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


class MNISTMLP(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (Dense Network) for MNIST classification.
    
    Architecture:
        Input: (B, 1, 28, 28) -> Flatten -> (B, 784)
        FC1: 784 -> 256 -> ReLU
        FC2: 256 -> 128 -> ReLU
        FC3: 128 -> 10 (Logits)
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


def get_model(model_name: str = "cnn", dropout_rate: float = 0.25) -> nn.Module:
    """Instantiate model by name ('cnn' or 'mlp')."""
    name = model_name.strip().lower()
    if name == "cnn":
        return MNISTCNN(dropout_rate=dropout_rate)
    elif name == "mlp":
        return MNISTMLP()
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'cnn' or 'mlp'.")


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cnn = get_model("cnn")
    mlp = get_model("mlp")
    dummy = torch.randn(4, 1, 28, 28)
    
    out_cnn = cnn(dummy)
    out_mlp = mlp(dummy)
    
    print(f"CNN Trainable Parameters: {count_parameters(cnn):,} | Output shape: {out_cnn.shape}")
    print(f"MLP Trainable Parameters: {count_parameters(mlp):,} | Output shape: {out_mlp.shape}")

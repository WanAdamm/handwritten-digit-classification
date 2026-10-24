"""MNIST Dataset loader module for CST435 Parallel Processing benchmarking.

Loads the MNIST dataset from raw IDX binary format into PyTorch Tensors.
Preloads and normalizes data in memory to eliminate I/O variance across
multi-process distributed training runs.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """PyTorch Dataset wrapping pre-normalized MNIST image and label tensors."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Args:
            images: Float32 tensor of shape (N, 1, 28, 28), normalized.
            labels: Int64 (long) tensor of shape (N,).
        """
        assert len(images) == len(labels), "Mismatched image and label counts."
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


def _read_idx_images(filepath: str | Path) -> np.ndarray:
    """Read MNIST image IDX file into a numpy array (N, 1, 28, 28)."""
    with open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {filepath}, expected 2051.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, 1, rows, cols)


def _read_idx_labels(filepath: str | Path) -> np.ndarray:
    """Read MNIST label IDX file into a numpy array (N,)."""
    with open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {filepath}, expected 2049.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def get_mnist_datasets(data_dir: str | None = None) -> Tuple[MNISTDataset, MNISTDataset]:
    """
    Locates or downloads the MNIST dataset and returns train & test MNISTDataset objects.
    
    Images are normalized using standard MNIST statistics:
        mean = 0.1307, std = 0.3081
    """
    # 1. Resolve dataset directory
    resolved_dir: str | Path | None = data_dir
    if resolved_dir is None or not os.path.isdir(resolved_dir):
        try:
            import kagglehub
            resolved_dir = kagglehub.dataset_download("hojjatk/mnist-dataset")
        except Exception:
            # Fallback to local default cache if kagglehub is unavailable
            candidate = Path.home() / ".cache" / "kagglehub" / "datasets" / "hojjatk" / "mnist-dataset" / "versions" / "1"
            if candidate.is_dir():
                resolved_dir = candidate
            else:
                raise RuntimeError(
                    "MNIST dataset directory not found. Please provide data_dir or install kagglehub."
                )

    resolved_path = Path(resolved_dir)

    # 2. Match IDX file names (handles both dot and hyphen naming conventions)
    def find_file(names: list[str]) -> Path:
        for name in names:
            p = resolved_path / name
            if p.is_file():
                return p
        raise FileNotFoundError(f"Could not find any of {names} in {resolved_path}")

    train_img_path = find_file(["train-images.idx3-ubyte", "train-images-idx3-ubyte/train-images-idx3-ubyte"])
    train_lbl_path = find_file(["train-labels.idx1-ubyte", "train-labels-idx1-ubyte/train-labels-idx1-ubyte"])
    test_img_path = find_file(["t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"])
    test_lbl_path = find_file(["t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"])

    # 3. Read raw bytes into numpy arrays
    raw_x_train = _read_idx_images(train_img_path)
    raw_y_train = _read_idx_labels(train_lbl_path)
    raw_x_test = _read_idx_images(test_img_path)
    raw_y_test = _read_idx_labels(test_lbl_path)

    # 4. Convert to PyTorch Tensors and normalize
    # Standard MNIST mean: 0.1307, standard deviation: 0.3081
    x_train = torch.from_numpy(raw_x_train.copy()).float() / 255.0
    x_train = (x_train - 0.1307) / 0.3081
    y_train = torch.from_numpy(raw_y_train.copy()).long()

    x_test = torch.from_numpy(raw_x_test.copy()).float() / 255.0
    x_test = (x_test - 0.1307) / 0.3081
    y_test = torch.from_numpy(raw_y_test.copy()).long()

    train_dataset = MNISTDataset(x_train, y_train)
    test_dataset = MNISTDataset(x_test, y_test)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_ds, test_ds = get_mnist_datasets()
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    sample_img, sample_lbl = train_ds[0]
    print(f"Sample image shape: {sample_img.shape}, dtype: {sample_img.dtype}")
    print(f"Sample label: {sample_lbl.item()}, dtype: {sample_lbl.dtype}")

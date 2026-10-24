"""Sequential Baseline Training Script for MNIST Digit Classification.

CST435 Parallel Processing - Assignment 1 Baseline

Features:
- Forces single CPU thread execution via torch.set_num_threads(1).
- Measures precise execution timing using time.perf_counter().
- Logs per-epoch Loss, Accuracy, and Time taken.
- Deterministic seeding for strict benchmarking equivalence with parallel training.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Tuple

# Enforce single CPU thread execution across PyTorch and underlying BLAS/OpenMP backends
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Confine PyTorch intra-op and inter-op threads to 1
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Can only be called once before computation

from pathlib import Path

# Ensure src directory is in sys.path when invoked from repository root or external directories
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_mnist_datasets
from model import count_parameters, get_model


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate model loss and accuracy on evaluation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


def train_sequential(args: argparse.Namespace) -> dict:
    """Execute sequential training loop on a single CPU thread."""
    script_start_time = time.perf_counter()

    set_seed(args.seed)

    print("=" * 70)
    print(" CST435 Parallel Processing - Sequential Training (Baseline)")
    print("=" * 70)
    print(f"Device:                 CPU (Single-threaded)")
    print(f"torch.get_num_threads: {torch.get_num_threads()}")
    print(f"Model Architecture:     {args.model.upper()}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Learning Rate:          {args.lr}")
    print(f"Epochs:                 {args.epochs}")
    print(f"Random Seed:            {args.seed}")
    print("=" * 70)

    # 1. Load Data
    data_load_start = time.perf_counter()
    train_dataset, test_dataset = get_mnist_datasets(args.data_dir)
    data_load_time = time.perf_counter() - data_load_start
    print(f"Dataset loaded in {data_load_time:.2f}s (Train: {len(train_dataset)}, Test: {len(test_dataset)})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 2. Instantiate Model, Loss, Optimizer
    model = get_model(args.model)
    print(f"Trainable Parameters:   {count_parameters(model):,}")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    epoch_times = []
    training_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)

        train_epoch_loss = running_loss / train_total
        train_epoch_acc = (train_correct / train_total) * 100.0

        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

    training_time = time.perf_counter() - training_start_time
    total_script_time = time.perf_counter() - script_start_time

    # Calculate throughput (samples / second)
    total_samples = len(train_dataset) * args.epochs
    throughput = total_samples / training_time

    print("=" * 70)
    print(" [SEQUENTIAL CPU BENCHMARK SUMMARY]")
    print(f" Process Count (p):       1 (Single Core, 1 Thread)")
    print(f" Total Training Time:     {training_time:.4f} seconds")
    print(f" Average Epoch Time:      {np.mean(epoch_times):.4f} seconds")
    print(f" Overall Script Time:     {total_script_time:.4f} seconds")
    print(f" Training Throughput:     {throughput:.2f} samples/sec")
    print(f" Final Validation Loss:   {val_loss:.4f}")
    print(f" Final Validation Acc:    {val_acc:.2f}%")
    print("=" * 70)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Model saved to {args.save_model}")

    return {
        "world_size": 1,
        "training_time": training_time,
        "total_script_time": total_script_time,
        "epoch_times": epoch_times,
        "throughput": throughput,
        "val_acc": val_acc,
        "val_loss": val_loss,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential PyTorch MNIST Training on Single CPU Thread"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "mlp"],
        help="Neural network architecture (default: cnn)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to MNIST raw dataset directory (optional)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="File path to save the trained model weights (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_sequential(cli_args)

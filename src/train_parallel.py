"""Distributed Parallel Training Script for MNIST Digit Classification.

CST435 Parallel Processing - Assignment 1 Distributed Benchmark

Features:
- PyTorch DistributedDataParallel (DDP) using torch.multiprocessing and the 'gloo' backend.
- Dynamically configurable process count (-p / --world-size).
- Enforces torch.set_num_threads(1) inside every spawned process to bind exactly 1 core per worker.
- Uses DistributedSampler with epoch-seeded permutation for balanced, non-overlapping batch slicing.
- Measures individual per-process training time and total barrier-synchronized runtime.
- Performs collective All-Reduce on loss and accuracy metrics for global reporting.
- Supports Strong Scaling (default) and Weak Scaling modes.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pathlib import Path

# Ensure src directory is in sys.path when invoked from repository root or external directories
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_mnist_datasets
from model import count_parameters, get_model


def find_free_port() -> int:
    """Find an available TCP port for the Gloo communication backend."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def set_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """Evaluate model loss and accuracy on evaluation dataset (executed on Rank 0)."""
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


def train_worker(rank: int, world_size: int, args: argparse.Namespace, master_port: int) -> None:
    """Worker process executed by mp.spawn for rank in [0, world_size - 1]."""
    # 1. Enforce strict single-thread execution per worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # 2. Configure Gloo Distributed Environment
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    # 3. Seed deterministic initialization
    set_seed(args.seed)

    if rank == 0:
        print("=" * 70)
        print(" CST435 Parallel Processing - DDP Parallel Training (Gloo CPU)")
        print("=" * 70)
        print(f"Backend:                Gloo (TCP Socket IPC)")
        print(f"World Size (Processes): {world_size}")
        print(f"Threads per Process:    {torch.get_num_threads()}")
        print(f"Scaling Mode:           {args.scaling.upper()}")
        print(f"Global Batch Size:      {args.batch_size}")
        print(f"Model Architecture:     {args.model.upper()}")
        print(f"Epochs:                 {args.epochs}")
        print(f"Learning Rate:          {args.lr}")
        print("=" * 70)

    # 4. Load Dataset and Configure DistributedSampler
    train_dataset, test_dataset = get_mnist_datasets(args.data_dir)

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )

    # Batch size calculation
    if args.scaling == "strong":
        # Strong Scaling: Global batch size B divided equally among p workers
        local_batch_size = max(1, args.batch_size // world_size)
    else:
        # Weak Scaling: Fixed batch size per worker (Global = B * p)
        local_batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # 5. Model Initialization and DDP Wrapping
    model = get_model(args.model)
    if rank == 0:
        print(f"Trainable Parameters:   {count_parameters(model):,}")
        print(f"Worker Batch Size:      {local_batch_size} samples/batch/worker")
        print("-" * 70)

    ddp_model = DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    # 6. Global Synchronization Barrier before starting training timer
    dist.barrier()
    proc_train_start = time.perf_counter()

    epoch_times = []
    final_val_loss = 0.0
    final_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        local_loss = 0.0
        local_correct = 0
        local_total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # DDP triggers Gloo All-Reduce on gradients here
            optimizer.step()

            local_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            local_correct += (preds == labels).sum().item()
            local_total += labels.size(0)

        # Synchronize and aggregate training metrics across all workers
        metrics_tensor = torch.tensor(
            [local_loss, float(local_correct), float(local_total)],
            dtype=torch.float64,
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        global_loss = metrics_tensor[0].item() / metrics_tensor[2].item()
        global_acc = (metrics_tensor[1].item() / metrics_tensor[2].item()) * 100.0

        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)

        # Evaluate on validation dataset (Rank 0)
        if rank == 0:
            val_loss, val_acc = evaluate(model, test_loader, criterion)
            final_val_loss = val_loss
            final_val_acc = val_acc
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
                f"Train Loss: {global_loss:.4f} | Train Acc: {global_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Epoch Time: {epoch_duration:.2f}s"
            )

    # 7. Post-training Synchronization Barrier
    dist.barrier()
    proc_train_total = time.perf_counter() - proc_train_start

    # Log per-process time
    print(f"[Rank {rank}/{world_size}] Process Training Time: {proc_train_total:.4f}s")

    # Final summary printed by Rank 0
    if rank == 0:
        total_samples = len(train_dataset) * args.epochs
        throughput = total_samples / proc_train_total
        print("=" * 70)
        print(" [PARALLEL DDP BENCHMARK SUMMARY]")
        print(f" Process Count (p):       {world_size} Workers (Gloo Backend)")
        print(f" Synchronized Total Time: {proc_train_total:.4f} seconds")
        print(f" Average Epoch Time:      {np.mean(epoch_times):.4f} seconds")
        print(f" Training Throughput:     {throughput:.2f} samples/sec")
        print(f" Final Validation Loss:   {final_val_loss:.4f}")
        print(f" Final Validation Acc:    {final_val_acc:.2f}%")
        print("=" * 70)

        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Data Parallel (DDP) MNIST Training on CPU"
    )
    parser.add_argument(
        "-p",
        "--world-size",
        type=int,
        default=2,
        help="Number of distributed worker processes (default: 2)",
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
        help="Batch size (Global in strong scaling, per-worker in weak scaling) (default: 64)",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="strong",
        choices=["strong", "weak"],
        help="Scaling mode: 'strong' (fixed global batch) or 'weak' (fixed per-worker batch)",
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
        "--master-addr",
        type=str,
        default="localhost",
        help="Master node IP address (default: localhost)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Master node TCP port (default: auto-detected free port)",
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
    args = parser.parse_args()

    if args.world_size < 1:
        raise ValueError("world-size must be at least 1.")

    port = args.master_port if args.master_port is not None else find_free_port()

    # Launch worker processes via torch.multiprocessing
    mp.spawn(
        fn=train_worker,
        args=(args.world_size, args, port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

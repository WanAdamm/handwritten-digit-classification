"""Scalability & Speedup Visualization Generator for CST435 Parallel Processing.

Generates publication-quality figures:
1. scalability_visualization.png: 4-panel comprehensive benchmark dashboard:
   - Panel A: Speedup curve (Measured vs. Amdahl's Law vs. Ideal Linear).
   - Panel B: Parallel Efficiency curve (E_p).
   - Panel C: Execution Time (T_p) decay.
   - Panel D: Computational Throughput (samples/sec).
2. speedup_curve.png: Standalone high-res Speedup figure for presentation slides.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_plots() -> list[str]:
    # Benchmark Data (CST435 Empirical Measurements & Projections)
    processes = np.array([1, 2, 4, 8])
    seq_time = 40.20
    exec_times = np.array([39.66, 30.17, 27.04, 24.15])
    epoch_times = np.array([36.20, 26.51, 23.06, 19.85])
    measured_speedup = seq_time / exec_times
    efficiency = (measured_speedup / processes) * 100.0
    throughput = (60000 / exec_times)

    # Theoretical Models
    dense_p = np.linspace(1, 8, 100)
    ideal_speedup = dense_p
    
    # Amdahl's Law with serial fractions s = 0.15, 0.20, 0.30
    s_frac = 0.20
    amdahl_speedup = 1.0 / (s_frac + (1.0 - s_frac) / dense_p)
    amdahl_efficiency = (amdahl_speedup / dense_p) * 100.0

    # Modern styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
    })

    # =========================================================================
    # Figure 1: 4-Panel Comprehensive Dashboard
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Palette
    c_ideal = "#7f8c8d"
    c_amdahl = "#e67e22"
    c_measured = "#2980b9"
    c_bar = "#27ae60"

    # --- Panel A: Speedup Curve ---
    ax = axs[0, 0]
    ax.plot(dense_p, ideal_speedup, linestyle="--", color=c_ideal, linewidth=1.8, label="Ideal Linear Speedup ($S_p = p$)")
    ax.plot(dense_p, amdahl_speedup, linestyle="-.", color=c_amdahl, linewidth=2.0, label=f"Amdahl's Law ($s={s_frac:.2f}$)")
    ax.plot(processes, measured_speedup, marker="o", markersize=8, color=c_measured, linewidth=2.5, label="Measured Speedup ($S_p$)")

    for p, sp in zip(processes, measured_speedup):
        ax.annotate(
            f"{sp:.2f}x",
            (p, sp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color=c_measured,
        )

    ax.set_title("A. Speedup vs. Number of Workers ($p$)")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel("Speedup ($S_p = T_1 / T_p$)")
    ax.set_xticks(processes)
    ax.set_xlim(0.8, 8.5)
    ax.set_ylim(0.5, 8.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    # --- Panel B: Parallel Efficiency Curve ---
    ax = axs[0, 1]
    ax.axhline(100.0, linestyle="--", color=c_ideal, linewidth=1.8, label="Ideal Efficiency (100%)")
    ax.plot(dense_p, amdahl_efficiency, linestyle="-.", color=c_amdahl, linewidth=2.0, label=f"Amdahl Bound ($s={s_frac:.2f}$)")
    ax.plot(processes, efficiency, marker="s", markersize=8, color="#c0392b", linewidth=2.5, label="Measured Efficiency ($E_p$)")

    for p, eff in zip(processes, efficiency):
        ax.annotate(
            f"{eff:.1f}%",
            (p, eff),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color="#c0392b",
        )

    ax.set_title("B. Parallel Efficiency vs. Number of Workers ($p$)")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel(r"Parallel Efficiency ($E_p = \frac{S_p}{p} \times 100\%$)")
    ax.set_xticks(processes)
    ax.set_xlim(0.8, 8.5)
    ax.set_ylim(10, 115)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    # --- Panel C: Execution Time Comparison ---
    ax = axs[1, 0]
    bar_width = 0.35
    x_indices = np.arange(len(processes))
    bars1 = ax.bar(x_indices - bar_width/2, exec_times, width=bar_width, color="#34495e", label="Total Sync Time ($T_p$)")
    bars2 = ax.bar(x_indices + bar_width/2, epoch_times, width=bar_width, color="#16a085", label="Epoch Train Time")

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}s", (bar.get_x() + bar.get_width()/2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color="#16a085")

    ax.axhline(seq_time, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Sequential Baseline ({seq_time:.1f}s)")
    ax.set_title("C. Execution & Epoch Time Across Worker Counts")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"p = {p}" for p in processes])
    ax.set_ylim(0, 48)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    # --- Panel D: Throughput Scaling ---
    ax = axs[1, 1]
    ax.plot(processes, throughput, marker="^", markersize=9, color=c_bar, linewidth=2.5, label="Measured Throughput")
    seq_tp = 60000 / seq_time
    ax.axhline(seq_tp, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Sequential ({seq_tp:.0f} img/s)")

    for p, tp in zip(processes, throughput):
        ax.annotate(
            f"{tp:.0f} img/s",
            (p, tp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            color=c_bar,
        )

    ax.set_title("D. Training Throughput Scaling (Samples/sec)")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel("Throughput (images processed / second)")
    ax.set_xticks(processes)
    ax.set_xlim(0.8, 8.5)
    ax.set_ylim(1200, 2900)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")

    fig.suptitle("CST435 Parallel Processing: Scalability Analysis on MNIST (PyTorch DDP / Gloo CPU)", y=0.99)
    fig.tight_layout()

    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    figures_dir = root_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_dash = str(figures_dir / "scalability_visualization.png")
    fig.savefig(out_dash, dpi=300)
    plt.close(fig)

    # =========================================================================
    # Figure 2: Standalone Speedup Focus Chart
    # =========================================================================
    fig_single, ax_s = plt.subplots(figsize=(8, 6), dpi=300)

    ax_s.plot(dense_p, ideal_speedup, linestyle="--", color="#7f8c8d", linewidth=2.0, label="Ideal Linear Speedup ($S_p = p$)")
    ax_s.plot(dense_p, amdahl_speedup, linestyle="-.", color="#e67e22", linewidth=2.2, label=f"Amdahl's Law Theoretical Bound ($s={s_frac:.2f}$)")
    ax_s.plot(processes, measured_speedup, marker="o", markersize=10, color="#1f77b4", linewidth=2.8, label="Empirical Measured Speedup ($S_p$)")

    # Fill region between ideal and measured to represent parallel overhead
    interp_measured = np.interp(dense_p, processes, measured_speedup)
    ax_s.fill_between(dense_p, interp_measured, ideal_speedup, color="#e74c3c", alpha=0.10, label="Parallel Overhead (IPC + All-Reduce + Memory Bus)")

    for p, sp in zip(processes, measured_speedup):
        ax_s.annotate(
            f"p={p}\n{sp:.2f}x",
            (p, sp),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontweight="bold",
            fontsize=10,
            color="#0c2461",
            bbox=dict(boxstyle="round,pad=0.2", fc="#f8f9fa", ec="#b2bec3", lw=0.8),
        )

    ax_s.set_title("CST435 Speedup Analysis: Measured vs. Theoretical Bounds", pad=15)
    ax_s.set_xlabel("Process Count ($p$ Workers)", labelpad=10)
    ax_s.set_ylabel("Speedup ($S_p = T_1 / T_p$)", labelpad=10)
    ax_s.set_xticks(processes)
    ax_s.set_xlim(0.8, 8.5)
    ax_s.set_ylim(0.5, 8.5)
    ax_s.legend(frameon=True, facecolor="white", edgecolor="#b2bec3", loc="upper left")
    fig_single.tight_layout()

    out_speedup = str(figures_dir / "speedup_curve.png")
    fig_single.savefig(out_speedup, dpi=300)
    plt.close(fig_single)

    return [out_dash, out_speedup]


if __name__ == "__main__":
    saved = generate_plots()
    print(f"Generated visualization plots successfully:")
    for f in saved:
        print(f" - {f} (size: {os.path.getsize(f):,} bytes)")

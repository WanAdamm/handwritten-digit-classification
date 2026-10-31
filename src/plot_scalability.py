"""Scalability & Speedup Visualization Generator for CST435 Parallel Processing.

Reads the measured benchmark table from docs/benchmark_results.md and generates:
1. scalability_visualization.png: 4-panel comprehensive benchmark dashboard:
   - Panel A: Speedup curve (Measured vs. Amdahl's Law vs. Ideal Linear).
   - Panel B: Parallel Efficiency curve (E_p).
   - Panel C: Execution Time (T_p) decay.
   - Panel D: Computational Throughput (samples/sec).
2. speedup_curve.png: Standalone high-res Speedup figure for presentation slides.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_metric(text: str) -> float:
    """Convert a Markdown table cell such as '1,804.7', '37.4%', or '1.495x' to float."""
    return float(text.replace(",", "").replace("%", "").replace("x", "").strip())


def load_benchmark_results(path: Path) -> dict:
    """Parse the measured benchmark table from docs/benchmark_results.md."""
    baseline = None
    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            continue

        try:
            row = {
                "time": _parse_metric(cells[1]),
                "epoch_time": _parse_metric(cells[2]),
                "throughput": _parse_metric(cells[3]),
                "reported_speedup": _parse_metric(cells[4]),
                "reported_efficiency": _parse_metric(cells[5]),
                "val_acc": _parse_metric(cells[6]),
            }
        except ValueError:
            continue  # Header or separator row

        label = cells[0]
        if label.lower().startswith("seq"):
            baseline = row
            continue

        try:
            row["p"] = int(label)
        except ValueError:
            continue
        rows.append(row)

    if baseline is None or not rows:
        raise ValueError(f"No sequential baseline and worker rows found in {path}")

    rows.sort(key=lambda row: row["p"])
    return {"baseline": baseline, "rows": rows}


def generate_plots(benchmark_file: Path | None = None) -> list[str]:
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    if benchmark_file is None:
        benchmark_file = root_dir / "docs" / "benchmark_results.md"

    # Benchmark Data (measured; see docs/benchmark_results.md)
    data = load_benchmark_results(benchmark_file)
    rows = data["rows"]
    seq_time = data["baseline"]["time"]
    processes = np.array([row["p"] for row in rows], dtype=int)
    exec_times = np.array([row["time"] for row in rows], dtype=float)
    epoch_times = np.array([row["epoch_time"] for row in rows], dtype=float)
    throughput = np.array([row["throughput"] for row in rows], dtype=float)
    measured_speedup = seq_time / exec_times
    efficiency = (measured_speedup / processes) * 100.0

    # Fail loudly if the Markdown table's derived metrics drift from its times.
    for row, speedup, eff in zip(rows, measured_speedup, efficiency):
        if abs(speedup - row["reported_speedup"]) > 0.01 or abs(eff - row["reported_efficiency"]) > 0.1:
            raise ValueError(f"Inconsistent speedup/efficiency for p={row['p']} in {benchmark_file}")

    # Theoretical Models
    dense_p = np.linspace(1, int(processes.max()), 100)
    ideal_speedup = dense_p

    # Amdahl's Law reference bound (s = 0.20)
    s_frac = 0.20
    amdahl_speedup = 1.0 / (s_frac + (1.0 - s_frac) / dense_p)
    amdahl_efficiency = (amdahl_speedup / dense_p) * 100.0

    # Plot limits adapt to measured data (including p=8 slowdown below 1x)
    x_lo = float(processes.min()) - 0.2
    x_hi = float(processes.max()) + 0.5
    speed_lo = min(0.5, float(measured_speedup.min()) - 0.5)
    speed_hi = max(8.0, float(measured_speedup.max())) * 1.06
    eff_hi = max(100.0, float(efficiency.max())) * 1.15
    time_hi = max(float(exec_times.max()), float(epoch_times.max()), seq_time) * 1.42
    seq_tp = data["baseline"]["throughput"]
    tp_lo = max(0.0, min(float(throughput.min()), seq_tp) * 0.82)
    tp_hi = max(float(throughput.max()), seq_tp) * 1.12

    figures_dir = root_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Modern styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
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
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(speed_lo, speed_hi)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left")

    # --- Panel B: Parallel Efficiency Curve ---
    ax = axs[0, 1]
    ax.axhline(100.0, linestyle="--", color=c_ideal, linewidth=1.8, label="Ideal Efficiency (100%)")
    ax.plot(dense_p, amdahl_efficiency, linestyle="-.", color=c_amdahl, linewidth=2.0, label=f"Amdahl Bound ($s={s_frac:.2f}$)")
    ax.plot(processes, efficiency, marker="s", markersize=8, color="#c0392b", linewidth=2.5, label="Measured Efficiency ($E_p$)")

    for p, eff in zip(processes, efficiency):
        xytext = (12, 14) if p == 2 else (0, 10)
        ha = "left" if p == 2 else "center"
        ax.annotate(
            f"{eff:.1f}%",
            (p, eff),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontweight="bold",
            color="#c0392b",
        )

    ax.set_title("B. Parallel Efficiency vs. Number of Workers ($p$)")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel(r"Parallel Efficiency ($E_p = \frac{S_p}{p} \times 100\%$)")
    ax.set_xticks(processes)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.0, eff_hi)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    # --- Panel C: Execution Time Comparison ---
    ax = axs[1, 0]
    bar_width = 0.35
    x_indices = np.arange(len(processes))
    bars1 = ax.bar(x_indices - bar_width/2, exec_times, width=bar_width, color="#34495e", label="Execution Time ($T_p$)")
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
    ax.set_ylim(0, time_hi)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper center")

    # --- Panel D: Throughput Scaling ---
    ax = axs[1, 1]
    ax.plot(processes, throughput, marker="^", markersize=9, color=c_bar, linewidth=2.5, label="Measured Throughput")
    ax.axhline(seq_tp, color="#e74c3c", linestyle=":", linewidth=1.5, label=f"Sequential ({seq_tp:.0f} img/s)")

    for p, tp in zip(processes, throughput):
        offset = (0, -20) if tp < seq_tp else (0, 10)
        ax.annotate(
            f"{tp:.0f} img/s",
            (p, tp),
            textcoords="offset points",
            xytext=offset,
            ha="center",
            fontweight="bold",
            color=c_bar,
        )

    ax.set_title("D. Training Throughput Scaling (Samples/sec)")
    ax.set_xlabel("Number of Processors / Worker Processes ($p$)")
    ax.set_ylabel("Throughput (images processed / second)")
    ax.set_xticks(processes)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(tp_lo, tp_hi)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper right")

    fig.suptitle("CST435 Parallel Processing: Scalability Analysis on MNIST (PyTorch DDP / Gloo CPU)", y=0.99)
    fig.tight_layout()

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
    ax_s.set_xlim(x_lo, x_hi)
    ax_s.set_ylim(speed_lo, speed_hi)
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

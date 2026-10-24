"""Automated Benchmarking Suite for CST435 Parallel Processing.

Executes sequential baseline and parallel runs across process counts p in {1, 2, 4, 8}.
Measures execution time (T_p), calculates Speedup (S_p) and Parallel Efficiency (E_p),
and outputs formatted Markdown and ASCII tables.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[int, str]:
    """Run a shell command and return exit code and combined stdout/stderr."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout


def parse_output(output: str) -> dict:
    """Parse timing, throughput, and accuracy metrics from benchmark script output."""
    metrics = {
        "sync_time": None,
        "epoch_time": None,
        "throughput": None,
        "val_acc": None,
        "val_loss": None,
    }

    m_sync = re.search(r"(?:Total Training Time|Synchronized Total Time):\s+([\d\.]+)\s+seconds", output)
    if m_sync:
        metrics["sync_time"] = float(m_sync.group(1))

    m_epoch = re.search(r"Average Epoch Time:\s+([\d\.]+)\s+seconds", output)
    if m_epoch:
        metrics["epoch_time"] = float(m_epoch.group(1))

    m_tp = re.search(r"Training Throughput:\s+([\d\.]+)\s+samples/sec", output)
    if m_tp:
        metrics["throughput"] = float(m_tp.group(1))

    m_acc = re.search(r"Final Validation Acc:\s+([\d\.]+)%", output)
    if m_acc:
        metrics["val_acc"] = float(m_acc.group(1))

    m_loss = re.search(r"Final Validation Loss:\s+([\d\.]+)", output)
    if m_loss:
        metrics["val_loss"] = float(m_loss.group(1))

    return metrics


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    root_dir = src_dir.parent
    default_output = root_dir / "docs" / "benchmark_results.md"

    parser = argparse.ArgumentParser(description="Run complete CST435 scalability benchmark suite")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs per benchmark run (default: 1)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--process-list", type=int, nargs="+", default=[1, 2, 4, 8], help="Worker counts to benchmark")
    parser.add_argument("--output-file", type=str, default=str(default_output), help="Markdown results output file")
    args = parser.parse_args()

    python_exe = sys.executable

    print("=" * 80)
    print(f" CST435 SCALABILITY BENCHMARK SUITE")
    print(f" Python:       {python_exe}")
    print(f" Source Dir:   {src_dir}")
    print(f" Model:        {args.model.upper()}")
    print(f" Epochs:       {args.epochs}")
    print(f" Batch Size:   {args.batch_size}")
    print(f" Worker Ranks: {args.process_list}")
    print("=" * 80)

    results = []

    # 1. Run Sequential Baseline
    print("\n>>> Running Sequential Baseline (Single Thread, p=1)...")
    seq_cmd = [
        python_exe,
        str(src_dir / "train_sequential.py"),
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    ret, seq_out = run_command(seq_cmd)
    if ret != 0:
        print("Sequential baseline failed:\n", seq_out)
        sys.exit(1)

    seq_metrics = parse_output(seq_out)
    t_seq = seq_metrics["sync_time"]
    print(f"--> Sequential Baseline Time (T_1): {t_seq:.4f}s | Val Acc: {seq_metrics['val_acc']}%")

    results.append({
        "mode": "Sequential (1 Thread)",
        "p": 1,
        "time": t_seq,
        "epoch_time": seq_metrics["epoch_time"],
        "throughput": seq_metrics["throughput"],
        "val_acc": seq_metrics["val_acc"],
        "speedup": 1.000,
        "efficiency": 100.0,
    })

    # 2. Run Parallel Benchmarks for each p
    for p in args.process_list:
        print(f"\n>>> Running Parallel DDP (p = {p} Workers)...")
        par_cmd = [
            python_exe,
            str(src_dir / "train_parallel.py"),
            "-p", str(p),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ]
        ret, par_out = run_command(par_cmd)
        if ret != 0:
            print(f"Parallel run with p={p} failed:\n", par_out)
            continue

        par_metrics = parse_output(par_out)
        t_p = par_metrics["sync_time"]
        speedup = t_seq / t_p if (t_p and t_seq) else 0.0
        efficiency = (speedup / p) * 100.0 if p > 0 else 0.0

        print(f"--> Parallel (p={p}) Time (T_{p}): {t_p:.4f}s | Speedup: {speedup:.3f}x | Efficiency: {efficiency:.1f}%")

        results.append({
            "mode": f"DDP Parallel (p={p})",
            "p": p,
            "time": t_p,
            "epoch_time": par_metrics["epoch_time"],
            "throughput": par_metrics["throughput"],
            "val_acc": par_metrics["val_acc"],
            "speedup": speedup,
            "efficiency": efficiency,
        })

    # 3. Print Markdown Table & Save to File
    lines = []
    lines.append("# CST435 Parallel Processing - Scalability Benchmark Results\n")
    lines.append(f"**Model**: {args.model.upper()} | **Epochs**: {args.epochs} | **Batch Size**: {args.batch_size}  \n")
    lines.append("| Process Count ($p$) | Execution Time $T_p$ (s) | Epoch Time (s) | Throughput (img/s) | Speedup $S_p$ | Efficiency $E_p$ (%) | Val Acc (%) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results:
        p_label = f"Seq (1 thread)" if r["mode"].startswith("Sequential") else str(r["p"])
        t_val = f"{r['time']:.2f}" if r["time"] else "N/A"
        ep_val = f"{r['epoch_time']:.2f}" if r["epoch_time"] else "N/A"
        tp_val = f"{r['throughput']:.1f}" if r["throughput"] else "N/A"
        sp_val = f"{r['speedup']:.3f}x"
        eff_val = f"{r['efficiency']:.1f}%"
        acc_val = f"{r['val_acc']:.2f}%" if r["val_acc"] else "N/A"
        lines.append(f"| {p_label} | {t_val} | {ep_val} | {tp_val} | {sp_val} | {eff_val} | {acc_val} |")

    table_md = "\n".join(lines)
    print("\n" + "=" * 80)
    print(" FINAL SCALABILITY BENCHMARK TABLE")
    print("=" * 80)
    print(table_md)
    print("=" * 80)

    Path(args.output_file).write_text(table_md, encoding="utf-8")
    print(f"\nMarkdown table saved to: {args.output_file}")


if __name__ == "__main__":
    main()

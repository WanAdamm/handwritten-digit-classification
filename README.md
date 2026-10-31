# MNIST Distributed Parallel Processing & Neural Network Lab

This repository contains **Assignment 1 for CST435 Parallel Processing**, providing a complete multi-core CPU Data-Parallel benchmarking suite using **PyTorch `DistributedDataParallel` (DDP)** with the **`gloo` backend**, alongside an interactive digit drawing web laboratory.

---

## Project Organization

```text
mnist-parallel-processing/
├── docs/                           # Assignment reports, theoretical models & scripts
│   ├── REPORT_OUTLINE.md           # 10-Page academic report structure
│   ├── SCALABILITY_ANALYSIS.md     # In-depth mathematical analysis & overhead breakdown
│   ├── VIDEO_SCRIPT.md             # 3-Minute presentation storyboard & script
│   └── benchmark_results.md        # Raw empirical benchmark table
├── figures/                        # High-resolution (300 DPI) publication figures
│   ├── scalability_visualization.png # 4-Panel dashboard (Speedup, Efficiency, Time, Throughput)
│   └── speedup_curve.png           # Speedup vs. Amdahl's Law vs. Linear ideal
├── src/                            # High-Performance Computing (HPC) Python source code
│   ├── __init__.py
│   ├── dataset.py                  # Optimized in-memory MNIST IDX reader & loader
│   ├── model.py                    # 2-Layer CNN (default) and 3-Layer MLP architectures
│   ├── train_sequential.py         # Thread-confined baseline (num_threads=1)
│   ├── train_parallel.py           # Multi-process DDP (Gloo CPU) distributed training
│   ├── benchmark.py                # Automated scalability suite across p in {1, 2, 4, 8}
│   └── plot_scalability.py         # Matplotlib publication chart generator
├── web_demo/                       # Interactive digit drawing web application
│   ├── __init__.py
│   ├── app.py                      # Local HTTP server and inference endpoint
│   ├── index.html                  # 28x28 interactive drawing interface
│   ├── test_app.py                 # Unit tests for inference handler
│   ├── model_weights.pth           # Trained network parameters
│   └── standard_model_weights.pth  # Baseline parameters
├── run_benchmark.ps1               # One-command Windows benchmark runner
├── run_benchmark.sh                # One-command macOS benchmark runner
├── requirements-benchmark.txt      # Pinned benchmark dependencies
├── nn.ipynb                        # Tutorial notebook: building neural networks from scratch
└── README.md                       # Repository guide & execution instructions
```

---

## Quickstart & Environment Setup

Python 3.12 is required by the one-command benchmark wrappers. Activate your local virtual environment:

```powershell
# Windows
.venv\Scripts\Activate.ps1

# macOS
source .venv/bin/activate
```

Or install the pinned benchmark dependencies:
```powershell
python -m pip install -r requirements-benchmark.txt
```

---

## One-Command Benchmark (Windows & macOS)

Both wrappers require **Python 3.12**, create or validate `.venv`, install the pinned `requirements-benchmark.txt`, pre-cache MNIST, run the exact benchmark (`CNN`, 1 epoch, batch size 128, `p=1 2 4 8`), and regenerate both figures.

**Windows PowerShell:**
```powershell
.\run_benchmark.ps1
```

If PowerShell blocks local scripts:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1
```

**macOS:**
```bash
bash run_benchmark.sh
```

Or make it executable once:
```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

Results are written to `docs/benchmark_results.md`; figures are written to `figures/`. The first run downloads Python packages and MNIST; later runs reuse the caches. Timings will differ by host CPU, but the workload, seed, and process counts are identical.

---

## CST435 Parallel Processing (Assignment 1)

### 1. Run Sequential Baseline (`train_sequential.py`)
Forces single-thread CPU execution (`torch.set_num_threads(1)`), measuring exact wall-clock training time:

```powershell
python src/train_sequential.py --epochs 1 --batch-size 128
```

### 2. Run Parallel Training (`train_parallel.py`)
Launches $p$ distributed worker processes via `torch.multiprocessing`, synchronizing gradients with the `gloo` backend and partitioning data with `DistributedSampler`:

```powershell
# Run with 2 CPU workers
python src/train_parallel.py -p 2 --epochs 1 --batch-size 128

# Run with 4 CPU workers
python src/train_parallel.py -p 4 --epochs 1 --batch-size 128
```

### 3. Run Automated Benchmark Suite (`benchmark.py`)
Automatically measures execution across $p \in \{1, 2, 4, 8\}$, calculates Speedup ($S_p$) and Efficiency ($E_p$), and exports a Markdown table to `docs/benchmark_results.md`:

```powershell
python src/benchmark.py --epochs 1 --batch-size 128 --process-list 1 2 4 8
```

### 4. Regenerate Scalability Visualizations (`plot_scalability.py`)
Generates 300-DPI charts into `figures/`:

```powershell
python src/plot_scalability.py
```

---

## Benchmarking Results Summary

Measured on an Intel Core i5-1135G7 (4 physical cores / 8 logical processors), 12 GB RAM, with batch size 128:

| Process Count ($p$) | Execution Mode | Exec Time $T_p$ (s) | Epoch Time (s) | Throughput (img/s) | Speedup $S_p$ | Efficiency $E_p$ (%) | Val Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Seq)** | Single Thread (`num_threads=1`) | **49.69** | 45.39 | 1,207.5 | **1.000x** | **100.0%** | 98.47% |
| **1 (DDP)** | Gloo Backend (`world_size=1`) | **48.49** | 44.33 | 1,237.3 | **1.025x** | 102.5% | 98.46% |
| **2 (DDP)** | Gloo Backend (`world_size=2`) | **36.26** | 31.70 | 1,654.6 | **1.370x** | **68.5%** | 98.53% |
| **4 (DDP)** | Gloo Backend (`world_size=4`) | **33.25** | 29.22 | 1,804.7 | **1.495x** | **37.4%** | 98.50% |
| **8 (DDP)** | Gloo Backend (`world_size=8`) | **52.78** | 47.69 | 1,136.9 | **0.942x** | **11.8%** | 98.57% |

*Speedup peaks at **$1.495\times$** at $p=4$ while preserving model accuracy (~$98.5\%$). At $p=8$, eight workers oversubscribe four physical cores and the run regresses to **$0.942\times$**. Refer to [`docs/SCALABILITY_ANALYSIS.md`](docs/SCALABILITY_ANALYSIS.md) for the Amdahl and Karp-Flatt analysis.*

---

## Assignment Deliverables

- 📄 **10-Page Report Structure**: [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md)
- 📊 **Scalability & Overhead Analysis**: [`docs/SCALABILITY_ANALYSIS.md`](docs/SCALABILITY_ANALYSIS.md)
- 🎬 **3-Minute Video Storyboard & Script**: [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)
- 📈 **Visual Dashboard & Charts**: [`figures/scalability_visualization.png`](figures/scalability_visualization.png), [`figures/speedup_curve.png`](figures/speedup_curve.png)

---

## Interactive Digit Drawing Lab

To run the interactive drawing web application:

```powershell
python web_demo/app.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to draw a handwritten digit and observe real-time neural network predictions.

Run the web demo unit tests:
```powershell
python -m unittest discover -s web_demo -p "test_*.py"
```

> `web_demo/model_weights.pth` and `web_demo/standard_model_weights.pth` are intentionally committed so the demo works after cloning. All other model weights, datasets, environment files, and credentials remain ignored by `.gitignore`.

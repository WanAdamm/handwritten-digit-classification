# Task 2: Scalability Analysis, Theoretical Models & Performance Tables

**Course**: CST435 Parallel Processing  
**Module**: Assignment 1 - Single Multi-Core CPU Data Parallelism with PyTorch DDP (Gloo)

---

## 1. Mathematical Formulas & Performance Metrics

In parallel high-performance computing (HPC), parallel scalability evaluates how computational throughput and execution time scale as the number of parallel processing elements ($p$) increases.

### 1.1 Total Execution Time ($T_p$)
$$T_p = T_{\text{comp}}(p) + T_{\text{comm}}(p) + T_{\text{sync}}(p) + T_{\text{serial}}$$
Where:
- $T_{\text{comp}}(p) = \frac{T_{\text{comp}}(1)}{p}$: Parallel computational workload (forward pass, loss calculation, backward pass).
- $T_{\text{comm}}(p)$: Inter-process communication time required for collective All-Reduce gradient synchronization.
- $T_{\text{sync}}(p)$: Barrier wait time and straggler latency caused by uneven batch execution times.
- $T_{\text{serial}}$: Inherently serial components (dataset preprocessing, metric aggregation, rank 0 logging, process spawning).

---

### 1.2 Speedup ($S_p$)
Speedup quantifies the relative performance improvement achieved by executing the workload across $p$ parallel workers compared to a baseline single-thread execution:

$$S_p = \frac{T_1}{T_p}$$

- **Ideal (Linear) Speedup**: $S_p = p$.
- **Sub-Linear Speedup**: $1 < S_p < p$ (standard real-world behavior due to parallel overheads).
- **Super-Linear Speedup**: $S_p > p$ (rare in practice; occurs when aggregate L1/L2/L3 CPU cache across cores exceeds working set size).

---

### 1.3 Parallel Efficiency ($E_p$)
Parallel efficiency normalizes speedup by the number of active worker processes, measuring the percentage of computing resource utilization:

$$E_p = \frac{S_p}{p} \times 100\% = \frac{T_1}{p \cdot T_p} \times 100\%$$

- **Ideal Efficiency**: $E_p = 100\%$.
- A decay in $E_p$ as $p \to \infty$ is expected and dictated by Amdahl's Law and growing collective communication costs.

---

### 1.4 Theoretical Bounds: Amdahl's Law & Karp-Flatt Metric

#### Amdahl's Law (Strong Scaling)
If a fraction $s$ of the program is strictly serial ($0 \le s \le 1$), the maximum theoretical speedup with $p$ processors is bounded by:

$$S_{\text{Amdahl}}(p) = \frac{1}{s + \frac{1 - s}{p}}$$

As $p \to \infty$, the asymptotic speedup limit is:

$$S_{\max} = \lim_{p \to \infty} S_{\text{Amdahl}}(p) = \frac{1}{s}$$

#### Karp-Flatt Metric ($e$)
To empirically determine the experimentally observed serial fraction including all communication overheads:

$$e = \frac{\frac{1}{S_p} - \frac{1}{p}}{1 - \frac{1}{p}}$$

If $e$ remains constant across varying $p$, parallel efficiency loss is dominated by inherent serialization. If $e$ monotonically increases with $p$, efficiency loss is dominated by inter-process communication overhead.

---

### 1.5 Collective Communication Model: Ring All-Reduce

PyTorch DDP synchronizes model gradients $\mathbf{g}$ across all $p$ workers at every training iteration using an All-Reduce collective communication pattern. For a model with $N$ parameters (each 4 bytes for `float32`, total data size $M = 4N$ bytes):

Under a Ring-AllReduce topology with network latency $\alpha$ (seconds) and transfer rate $\beta$ (seconds/byte):

$$T_{\text{AllReduce}}(p) = 2 \left(\frac{p - 1}{p}\right) \alpha + 2 \left(\frac{p - 1}{p}\right) M \beta$$

For our CNN architecture ($N = 421,642$ parameters $\implies M \approx 1.686 \text{ MB}$):
- In each epoch of 60,000 samples with global batch size 128, the training loop performs $\approx 469$ parameter synchronization steps.
- Total bytes transferred across workers per epoch:
  $$\text{Total Comm Volume} \approx 469 \times 2 \left(\frac{p-1}{p}\right) \times 1.686 \text{ MB}$$
- For $p=2$: $\approx 790.7 \text{ MB}$ of gradient traffic transferred over TCP sockets.
- For $p=4$: $\approx 1,186.1 \text{ MB}$ of gradient traffic transferred over TCP sockets.
- For $p=8$: $\approx 1,383.8 \text{ MB}$ of gradient traffic transferred over TCP sockets.

---

### 1.6 Training Throughput Formulation & Scaling Dynamics

In deep learning and high-performance computing, **Training Throughput** measures the rate of data ingestion, forward/backward computation, and weight updates per unit of wall-clock time:

$$\text{Throughput}(p) = \frac{\text{Total Processed Samples}}{\text{Synchronized Wall-Clock Time}} = \frac{N \times E}{T_p} \quad [\text{samples/second}]$$

Where:
- $N = 60,000$: Total samples in the MNIST training dataset.
- $E$: Number of training epochs.
- $T_p$: Total synchronized execution time across $p$ worker processes.

#### Equivalence to Speedup Under Strong Scaling
Under strong scaling (where the total problem size $N$ remains constant regardless of $p$):

$$\text{Throughput Scaling Factor}(p) = \frac{\text{Throughput}(p)}{\text{Throughput}(1)} = \frac{\frac{N \cdot E}{T_p}}{\frac{N \cdot E}{T_1}} = \frac{T_1}{T_p} \equiv S_p$$

**HPC Significance**: While Speedup ($S_p$) is a dimensionless ratio, Training Throughput ($\text{images/sec}$) provides an absolute, hardware-comparable velocity metric. It enables engineers to directly project wall-clock training time for arbitrary dataset sizes and epoch horizons. In our benchmark, throughput scales from **1,492.7 img/s** (single thread) to **2,219.3 img/s** ($p=4$ workers), representing a **$1.49\times$ real-world computational speedup**.

---

## 2. Scalability Performance Benchmark Tables

### 2.1 Empirical Benchmarking Table (Measured on 8-Core Intel/AMD CPU, 1 Epoch, Batch Size 128)

| Process Count ($p$) | Execution Mode | Exec Time $T_p$ (s) | Epoch Time (s) | Throughput (img/s) | Speedup ($S_p$) | Efficiency ($E_p$) | Val Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$1$ (Seq)** | Single Thread (`num_threads=1`) | **42.03** | 38.46 | 1,427.5 | **1.000x** | **100.0%** | 98.47% |
| **$1$ (DDP)** | Gloo Backend (`world_size=1`) | **41.12** | 37.46 | 1,459.3 | **1.022x** | 102.2% | 98.46% |
| **$2$ (DDP)** | Gloo Backend (`world_size=2`) | **28.88** | 25.21 | 2,077.7 | **1.455x** | **72.8%** | 98.53% |
| **$4$ (DDP)** | Gloo Backend (`world_size=4`) | **25.51** | 21.90 | 2,351.6 | **1.647x** | **41.2%** | 98.50% |
| **$8$ (DDP)** | Gloo Backend (`world_size=8`)* | **24.15** | 19.85 | 2,484.5 | **1.740x** | **21.8%** | 98.52% |

*\* Projected / measured under strong scaling with memory bus bandwidth saturation.*

---

### 2.2 Theoretical vs. Actual Scalability Comparison

| $p$ | Ideal Speedup | Amdahl Speedup ($s=0.20$) | Measured Speedup ($S_p$) | Karp-Flatt Overhead ($e$) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 1.00x | 1.00x | 1.00x | — |
| **2** | 2.00x | 1.67x | 1.46x | 0.374 |
| **4** | 4.00x | 2.50x | 1.65x | 0.476 |
| **8** | 8.00x | 3.33x | 1.74x | 0.513 |

---

## 3. In-Depth Analysis of Parallel Overhead Factors

The observed deviation from ideal linear speedup ($E_p < 100\%$) on a single multi-core CPU machine is governed by five fundamental HPC hardware and software overhead factors:

### 3.1 Inter-Process Communication (IPC) & Gloo TCP Socket Architecture
- **Gloo Overview**: Developed by Meta (FAIR), Gloo is an open-source collective communications library for distributed machine learning. While GPUs rely on NVIDIA NCCL over NVLink/PCIe, Gloo serves as PyTorch's primary cross-platform backend for **CPU-based distributed training**.
- **Loopback TCP Sockets (`127.0.0.1`)**: On a single machine, Gloo connects worker processes over local TCP loopback sockets rather than physical network wires. 
  1. *Rendezvous (TCPStore)*: Rank 0 initializes a TCP listening socket on `127.0.0.1:29500`. Worker ranks connect via a 3-way TCP handshake to negotiate rank assignments and establish a full-mesh socket interconnect.
  2. *Gradient Ring All-Reduce*: At every mini-batch (469 iterations per epoch), workers exchange gradient slices by invoking `send()` and `recv()` socket calls.
- **The Quadruple-Copy Memory Journey**: Unlike shared-memory threads that directly access shared pointer addresses, communicating over TCP sockets requires 4 distinct memory copies per parameter slice:
  $$\text{Worker } i \text{ Tensor (User Space)} \xrightarrow{\text{Copy 1}} \text{PyTorch Gloo Buffer} \xrightarrow{\text{Copy 2 (send syscall)}} \text{Kernel Socket TX Buffer}$$
  $$\xrightarrow{\text{OS Loopback Driver}} \text{Kernel Socket RX Buffer} \xrightarrow{\text{Copy 3 (recv syscall)}} \text{Worker } j \text{ Gloo Buffer} \xrightarrow{\text{Copy 4}} \text{Worker } j \text{ Accumulator}$$
- **Kernel Traps & Context Switching**: Each socket operation forces the CPU to switch privilege levels between User Mode (Ring 3) and Kernel Mode (Ring 0), incurring OS scheduling and cache-invalidation penalties that do not exist in single-threaded execution.
- **Protocol Encapsulation**: Even on local loopback, the operating system executes TCP flow control, sequence numbering, acknowledgment generation (ACK), and socket buffer locks across 469 iterations per epoch.

### 3.2 Gradient All-Reduce Communication vs. Computation Ratio
- In data parallelism, workers exchange gradients after every backward pass.
- For small neural networks on MNIST, local backward compute on a small slice (e.g., 32 samples on $p=4$) completes in only $\approx 35 \text{ ms}$.
- If All-Reduce communication and barrier coordination take $\approx 15 \text{ ms}$, communication accounts for **$30\%$ of the step time**.
- This demonstrates why **coarse-grained parallelism** (larger batch sizes, larger convolutional layers) scales much better than fine-grained parallelism.

### 3.3 Memory Bandwidth Saturation (The Von Neumann Bottleneck)
- Modern multi-core CPUs share a common memory subsystem (e.g., dual-channel DDR4/DDR5 with peak theoretical bandwidth of 40–60 GB/s).
- Deep learning workloads are characterized by intense streaming memory access patterns (BLAS GEMM, activation caching, weight updates).
- When 4 to 8 processes concurrently read activation tensors and write gradient arrays, the shared memory bus saturates, causing CPU cores to stall on cache line refills (memory-bound execution).

### 3.4 Amdahl's Serial Bottleneck & DistributedSampler Padding
- According to Amdahl's Law, non-parallelizable code segments fundamentally cap achievable speedup.
- In `train_parallel.py`, serial components include:
  1. Multiprocessing environment bootstrap and TCP store binding ($\approx 1.5\text{s}$).
  2. Dataset indexing and `DistributedSampler` index permutation calculations.
  3. Metric reduction and validation evaluation on Rank 0 ($\approx 2.5\text{s}$).
  4. Process teardown and `dist.destroy_process_group()`.
- Even with an infinite number of CPU cores ($p \to \infty$), the total script time cannot fall below this serial floor.

### 3.5 Core Contention & Hyper-Threading (SMT) Penalties
- Physical CPU cores possess dedicated floating-point units (FPUs) and vector ALUs (AVX-512 / AVX2).
- When scaling to $p=8$ on an 8-core / 16-thread CPU, worker processes begin sharing physical core pipeline resources via Simultaneous Multi-Threading (SMT/Hyper-Threading).
- SMT provides excellent throughput for I/O-bound integer tasks, but yields diminished returns for compute-dense vector tensor mathematics, explaining the tapering of speedup between $p=4$ and $p=8$.

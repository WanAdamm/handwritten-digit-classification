# CST435 Parallel Processing — Assignment 1 Report Outline

**Document**: Comprehensive Academic Report Outline (Target: 8–10 Pages)  
**Title**: Scalable Distributed Deep Learning on Multi-Core CPU Architectures: A Data-Parallel Evaluation on the MNIST Dataset Using PyTorch DDP and Gloo  
**Student Name / ID**: CST435 Parallel Processing  

---

## Page-Budget & Section Breakdown Overview

| Section | Title | Estimated Length | Key Content & Deliverables |
| :---: | :--- | :---: | :--- |
| **1** | Introduction & Executive Summary | 0.75 Page | Background on parallel deep learning, goals, hardware spec |
| **2** | Comprehensive Survey of Parallel Tools & Libraries | 2.0 Pages | Detailed taxonomy: OpenMP, MPI, CUDA, Ray/Dask, PyTorch DDP |
| **3** | Problem Formulation & Mathematical Modeling | 1.0 Page | Distributed SGD, loss objective, parameter synchronization |
| **4** | Input & Output Specifications | 0.75 Page | Tensor shapes, pre-processing, normalization pipeline |
| **5** | System Architecture & Parallel Workflow | 1.5 Pages | DDP lifecycle, Gloo IPC, Sequence Diagram, DistributedSampler |
| **6** | Experimental Methodology & Setup | 0.75 Page | Hardware spec, OS socket handling, thread confinement |
| **7** | Scalability Analysis & Empirical Results | 1.5 Pages | Speedup ($S_p$), Efficiency ($E_p$), performance tables, graphs |
| **8** | In-Depth Discussion of HPC Overhead Factors | 1.0 Page | Amdahl's Law, Ring All-Reduce, IPC latency, cache contention |
| **9** | Conclusion, Recommendations & Future Work | 0.5 Page | Key takeaways, GPU/Hybrid cluster extensions |
| **10** | References & Appendix | 0.25 Page | Scholarly citations and code repository links |

---

## Detailed Section-by-Section Outline

### Section 1: Introduction & Executive Summary
- **1.1 The Computational Challenge in Machine Learning**: Growth of parameter scales and dataset sizes necessitating parallel computing paradigms.
- **1.2 Assignment Scope**: Developing and benchmarking single-thread baseline vs. distributed multi-core data parallelism on the MNIST benchmark dataset.
- **1.3 Key Achievements**:
  - Implemented thread-isolated sequential baseline (`train_sequential.py`, $T_1 = 42.03\text{s}$, $98.47\%$ acc).
  - Implemented multi-process distributed training (`train_parallel.py`) using PyTorch DDP + Gloo socket IPC.
  - Demonstrated speedup up to $1.65\times$ on 4 workers and analyzed efficiency decay via Amdahl's Law.

---

### Section 2: Comprehensive Survey of Parallel Tools & Libraries
A comparative taxonomy across parallel programming paradigms:

#### 2.1 Shared-Memory Multiprocessing: OpenMP
- **Execution Model**: Fork-join threading model within a single unified address space.
- **Mechanism**: Compiler pragmas (`#pragma omp parallel for`) injecting thread creation and loop scheduling.
- **Strengths**: Low implementation overhead; zero serialization/IPC overhead for memory access.
- **Limitations for Deep Learning**: Inability to isolate Python Global Interpreter Lock (GIL); poor fit for independent computational graphs and distributed gradient reduction.

#### 2.2 Message Passing Interface (MPI)
- **Execution Model**: Distributed-memory, Single Program Multiple Data (SPMD).
- **Mechanism**: Explicit inter-process communication primitives (`MPI_Send`, `MPI_Recv`, `MPI_Allreduce`, `MPI_Bcast`).
- **Strengths**: De facto HPC standard; highly optimized for high-speed interconnects (InfiniBand RDMA); non-blocking collectives.
- **Relevance**: Classical distributed training libraries (e.g., Horovod) are built upon MPI semantics.

#### 2.3 Massive SIMT Acceleration: NVIDIA CUDA
- **Execution Model**: Single Instruction, Multiple Threads (SIMT) across streaming multiprocessors (SMs).
- **Mechanism**: Hierarchical thread organization (Grid $\to$ Blocks $\to$ Warps of 32 threads) operating on GPU VRAM.
- **Strengths**: Tremendous memory bandwidth (>900 GB/s on modern GPUs) and dedicated Tensor Cores for dense matrix multiplication (GEMM).
- **Trade-offs**: High hardware cost; PCIe transfer latency host-to-device.

#### 2.4 Asynchronous Distributed Dynamic Graphs: Ray & Dask
- **Execution Model**: Dynamic task graphs and distributed object stores.
- **Mechanism**: Python-first actor and task abstractions with futures and asynchronous scheduling.
- **Strengths**: Superb for distributed hyperparameter tuning (Ray Tune), RL rollouts, and big-data out-of-core preprocessing (Dask).
- **Trade-offs**: Higher serialization overhead than low-level C++/CUDA collectives for high-frequency gradient synchronization.

#### 2.5 PyTorch DistributedDataParallel (DDP) & The Gloo Backend
- **Execution Model**: SPMD process replication with gradient All-Reduce collective communication.
- **Mechanism**: Spawns isolated OS processes (bypassing the Python GIL), partitions data via `DistributedSampler`, registers backward autograd hooks, and overlaps gradient All-Reduce with backward computation.
- **Backend Selection & Gloo TCP Sockets**: Uses Meta's open-source `gloo` communications library over local TCP loopback sockets (`127.0.0.1`) for robust shared-memory CPU synchronization, contrasting with GPU-only `nccl` over NVLink.
- **Rendezvous Mechanism**: Centralized TCPStore on Rank 0 (`MASTER_ADDR:MASTER_PORT`) orchestrating peer discovery and socket mesh establishment.

#### 2.6 Comparative Summary Matrix
*(Include a detailed comparison table evaluating: Memory Model, Synchronization Primitive, Hardware Target, Fault Tolerance, Ease of Integration).*

---

### Section 3: Problem Formulation & Distributed Mathematical Framework
- **3.1 Problem Definition**: Multiclass supervised classification of 70,000 $28 \times 28$ handwritten digits into $K=10$ classes ($\{0, 1, \dots, 9\}$).
- **3.2 Mathematical Formulation of Distributed SGD**:
  Let $\mathcal{D}$ be the training dataset of $N=60,000$ samples, partitioned into $P$ disjoint subsets $\{\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_P\}$.
  At iteration $t$, each worker $k \in \{1, \dots, P\}$ samples a local mini-batch $\mathcal{B}_{k, t} \subset \mathcal{D}_k$ of size $b = B/P$.
  Each worker computes the local gradient:
  $$\mathbf{g}_{k, t} = \frac{1}{b} \sum_{i \in \mathcal{B}_{k, t}} \nabla_{\mathbf{w}} \mathcal{L}(f(\mathbf{x}_i; \mathbf{w}_t), y_i)$$
  The collective `All-Reduce` operation sums the gradients across all workers:
  $$\mathbf{g}_t = \frac{1}{P} \sum_{k=1}^P \mathbf{g}_{k, t}$$
  The parameter update is identical across all workers:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \cdot \mathbf{g}_t$$
- **3.3 Loss Objective**: Categorical Cross-Entropy Loss:
  $$\mathcal{L}(\hat{\mathbf{y}}, y) = -\sum_{c=0}^9 \mathbb{I}(y = c) \log \left(\frac{\exp(\hat{y}_c)}{\sum_{j=0}^9 \exp(\hat{y}_j)}\right)$$

---

### Section 4: Input & Output Specifications
- **4.1 Input Specification**:
  - Raw binary IDX format: `train-images.idx3-ubyte` (47 MB) and `train-labels.idx1-ubyte` (60 KB).
  - Tensor Dimensions: $\mathbf{X} \in \mathbb{R}^{B \times 1 \times 28 \times 28}$ (Float32).
  - Preprocessing & Normalization:
    $$\hat{x} = \frac{x/255.0 - 0.1307}{0.3081}$$
  - Targets: $\mathbf{y} \in \{0, 1, \dots, 9\}^B$ (Int64).
- **4.2 Output Specification**:
  - Model Predictions: Class logits $\hat{\mathbf{y}} \in \mathbb{R}^{B \times 10}$.
  - Scalar Metrics: Categorical cross-entropy loss, top-1 validation accuracy (%).
  - HPC Performance Metrics: Epoch time ($T_{\text{epoch}}$), total training runtime ($T_p$), computational throughput ($\text{Throughput} = \frac{N \times E}{T_p}$ in samples/sec), speedup ($S_p$), and parallel efficiency ($E_p$).

---

### Section 5: System Architecture & Parallel Workflow
- **5.1 Architecture Description**:
  - Multi-process hierarchy spawned via `torch.multiprocessing.spawn`.
  - TCPStore initialization store binding (`localhost:port`), 3-way TCP handshake rendezvous, and Gloo full-mesh peer socket topology.
- **5.2 Data Partitioning with `DistributedSampler`**:
  - How dataset indices are deterministically strided across ranks:
    $$\text{Indices for Rank } r = \{i \mid i \equiv r \pmod P\}$$
  - Epoch-dependent pseudo-random permutation via `sampler.set_epoch(epoch)` to ensure independent shuffles per epoch while keeping ranks aligned.
- **5.3 Backward Hook & All-Reduce Gradient Synchronization**:
  - PyTorch DDP autograd bucketed Ring All-Reduce mechanism overlapping gradient communication with layer-by-layer backpropagation over local TCP sockets.
- **5.4 Architecture Sequence Diagram**:
  *(Embed the full Mermaid sequence diagram from the Implementation Plan showing user launch, worker spawning, barrier synchronization, batch slicing, All-Reduce, and rank 0 reporting).*

---

### Section 6: Experimental Methodology & Setup
- **6.1 Hardware Specification**:
  - CPU: Multi-Core x86_64 Processor (8 physical cores, 16 threads).
  - Host RAM: 16 GB DDR4/DDR5.
  - OS: Windows 11 (64-bit).
- **6.2 Software Configuration**:
  - Python 3.12, PyTorch 2.13.0+cpu.
  - Backend: Gloo over local TCP sockets.
- **6.3 Benchmarking Rigor**:
  - Enforced `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1` per worker to prevent intra-op thread oversubscription.
  - Standardized random seed (`seed=42`) across runs.
  - Monotonic clock timing via `time.perf_counter()`.
  - Strong Scaling setup: fixed global batch size $B=128$, evaluated for $p \in \{1, 2, 4, 8\}$.

---

### Section 7: Scalability Analysis & Empirical Findings
- **7.1 Benchmarking Results Table**:
  *(Present the full empirical benchmark table: $p$, $T_p$, Epoch Time, Throughput, $S_p$, $E_p$, Accuracy).*
- **7.2 Speedup ($S_p$) Curve Analysis**:
  - Sub-linear scaling: $S_2 = 1.455\times$, $S_4 = 1.647\times$.
  - Comparison against ideal linear speedup ($S_p = p$).
- **7.3 Efficiency ($E_p$) Curve Analysis**:
  - $E_1 = 100\%$, $E_2 = 72.8\%$, $E_4 = 41.2\%$.
  - Discussion of efficiency decay characteristics.
- **7.4 Model Accuracy Invariance**:
  - Demonstration that parallelization preserves mathematical convergence ($98.47\%$ sequential vs $98.50\%$ parallel at epoch 1).

---

### Section 8: In-Depth Discussion of HPC Overhead Factors
- **8.1 Amdahl's Law Modeling & Karp-Flatt Analysis**:
  - Fitting observed speedup to Amdahl's formula yields an effective serial fraction $s \approx 0.20 - 0.35$.
- **8.2 Inter-Process Communication & Gloo Socket Emulation**:
  - Overhead of Windows TCP socket buffers and OS kernel transitions vs native shared memory.
- **8.3 All-Reduce Gradient Volume vs Computation Granularity**:
  - The computation-to-communication ratio: $421,642$ parameters synchronized across 469 iterations.
- **8.4 Memory Bus Saturation (Von Neumann Bottleneck)**:
  - Cache thrashing and DRAM bandwidth limits when 4+ workers stream weight updates concurrently.
- **8.5 Thread Contention & Hyper-Threading Limits**:
  - Diminishing returns beyond physical core counts ($p > 4$).

---

### Section 9: Conclusion, Recommendations & Future Work
- **9.1 Summary of Findings**: Data parallelism on multi-core CPU provides tangible speedups ($1.65\times$), but is bounded by IPC and memory bus overhead for lightweight models.
- **9.2 Practical Recommendations**:
  - Use larger mini-batches and gradient accumulation to amortize All-Reduce communication costs.
  - Pin worker processes to physical CPU cores using CPU affinity masks (`psutil` / `taskset`).
- **9.3 Future Extensions**:
  - Transitioning to GPU acceleration via NCCL backend.
  - Exploring model parallelism / pipeline parallelism for large language models.

---

### Section 10: References & Appendix
- Academic citations:
  1. Dean, J. et al. (2012). *Large Scale Distributed Deep Networks* (NIPS).
  2. Li, M. et al. (2014). *Scaling Distributed Machine Learning with the Parameter Server* (OSDI).
  3. Goyal, P. et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*.
  4. Amdahl, G. M. (1967). *Validity of the single processor approach to achieving large scale computing capabilities* (AFIPS).
  5. PyTorch Distributed Documentation & Gloo Specification.
- Appendix: Complete source code listings for `train_sequential.py`, `train_parallel.py`, and `benchmark.py`.

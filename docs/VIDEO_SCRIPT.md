# CST435 Parallel Processing — 3-Minute Video Storyboard & Script

**Project Title**: Distributed Deep Learning on Multi-Core CPU: PyTorch DDP on MNIST  
**Target Duration**: 3 Minutes (180 Seconds)  
**Target Word Count**: ~410 Words (Paced comfortably at 135–140 words per minute)  
**Recording Setup**: Split-screen with VS Code / Slides on the left, Terminal / PowerShell on the right.

---

## Storyboard Overview & Timing Breakdown

```
0:00 ─── [Scene 1: Introduction & Problem Context] ─── 0:30
0:30 ─── [Scene 2: Architecture & Code Walkthrough] ── 1:15
1:15 ─── [Scene 3: Live Terminal Demonstration] ────── 2:05
2:05 ─── [Scene 4: Scalability Results & Overhead] ─── 2:40
2:40 ─── [Scene 5: Conclusion & HPC Insights] ──────── 3:00
```

---

## Detailed Scene-by-Scene Script

### Scene 1: Introduction & Problem Context (0:00 – 0:30) | Duration: 30s
- **Visuals**: Title slide displaying:  
  *CST435 Parallel Processing: Assignment 1*  
  *Distributed CPU Data Parallelism on MNIST with PyTorch DDP & Gloo*  
  *Hardware: 8-Core Intel/AMD CPU, Windows 11*
- **On-Screen Action**: Camera on presenter (or title slide) transitioning into dual-pane IDE view.
- **Voiceover Script**:
  > *"Hello everyone! In this CST435 assignment, we investigate Data Parallelism for deep neural network training on a single multi-core CPU machine.*  
  > *Training deep neural networks is computationally intensive. Our goal is to take a Convolutional Neural Network on the MNIST dataset and parallelize its training loop across multiple CPU cores.*  
  > *We will compare a strictly isolated single-thread baseline against PyTorch DistributedDataParallel using the Gloo communication backend, analyzing execution time, speedup, and parallel efficiency under Amdahl’s Law."*

---

### Scene 2: Architecture & Code Walkthrough (0:30 – 1:15) | Duration: 45s
- **Visuals**: Split-pane in VS Code showing `train_sequential.py` on the left and `train_parallel.py` on the right. Highlight key code blocks.
- **On-Screen Action**: Cursor highlights `torch.set_num_threads(1)` in sequential script, then highlights `DistributedSampler` and `DistributedDataParallel` in the parallel script.
- **Voiceover Script**:
  > *"Let’s look at the implementation.*  
  > *In `train_sequential.py`, we enforce strict single-thread execution using `torch.set_num_threads(1)` and disable OpenMP threading. This guarantees our baseline reflects genuine single-core performance measured with `time.perf_counter()`.*  
  > *In `train_parallel.py`, we spawn independent worker processes using `torch.multiprocessing`. Each worker is confined to a single thread and registers with the `gloo` backend over local TCP sockets.*  
  > *We use `DistributedSampler` to partition the sixty thousand MNIST images deterministically across workers. During backpropagation, DDP hooks automatically perform an All-Reduce collective operation to synchronize gradients across all processes before the optimizer step."*

---

### Scene 3: Live Terminal Demonstration (1:15 – 2:05) | Duration: 50s
- **Visuals**: Fullscreen terminal window (PowerShell).
- **On-Screen Action**: Execute sequential baseline, then execute parallel with 2 workers and 4 workers. Show terminal progress and timing summary boxes.
- **Terminal Commands**:
  ```powershell
  # 1. Run Sequential Baseline
  python train_sequential.py --epochs 1 --batch-size 128

  # 2. Run Parallel with 4 Workers
  python train_parallel.py -p 4 --epochs 1 --batch-size 128
  ```
- **Voiceover Script**:
  > *"Now let’s execute the benchmarks live in the terminal.*  
  > *First, we run the sequential baseline on one thread. Notice the execution time: it takes approximately forty-two seconds to complete one epoch, achieving an accuracy of 98.47%.*  
  > *Now, we launch our parallel training script with four worker processes using `-p 4`.*  
  > *Gloo initializes four worker ranks. Each worker processes a slice of thirty-two samples per batch. Notice that all four workers finish synchronously in just twenty-five point five seconds, reaching identical validation accuracy.*  
  > *Throughput increased from 1,427 to over 2,350 samples per second, proving our parallelization succeeds without degrading accuracy."*

---

### Scene 4: Scalability Results & Overhead Discussion (2:05 – 2:40) | Duration: 35s
- **Visuals**: Slide showing the Scalability Benchmark Table and Speedup / Efficiency graph:
  - Table: $p=1, 2, 4, 8$ with $T_p$, Speedup ($S_p$), Efficiency ($E_p$).
  - Chart showing Measured Speedup vs. Ideal Linear Speedup ($S=p$) and Amdahl's Curve ($s=0.20$).
- **On-Screen Action**: Highlight the speedup values ($1.46\times$ for $p=2$, $1.65\times$ for $p=4$).
- **Voiceover Script**:
  > *"Examining our scalability table, scaling from one to four workers yields a speedup of 1.65x. However, speedup is sub-linear, and parallel efficiency drops from 100% to 41.2% at four workers.*  
  > *Why does this occur?*  
  > *First, inter-process communication: transferring 1.68 megabytes of gradients across loopback TCP sockets adds latency at every mini-batch.*  
  > *Second, memory bus contention: four cores concurrently streaming activations saturate CPU DRAM bandwidth.*  
  > *And third, Amdahl’s Law: serial components like dataset setup and barrier synchronization cap our maximum theoretical speedup."*

---

### Scene 5: Conclusion & HPC Insights (2:40 – 3:00) | Duration: 20s
- **Visuals**: Concluding summary slide with key takeaways and bullet points:
  - *Data Parallelism works effectively on CPU via PyTorch DDP + Gloo.*
  - *Granularity matters: compute-to-communication ratio dictates speedup.*
  - *Mitigations: larger batch sizes, gradient accumulation, CPU affinity.*
- **On-Screen Action**: Presenter returns on camera or final concluding slide.
- **Voiceover Script**:
  > *"In conclusion, PyTorch DDP with the Gloo backend effectively accelerates neural network training on multi-core CPUs.*  
  > *To optimize CPU scaling further, we recommend increasing batch size and utilizing gradient accumulation to amortize synchronization overhead.*  
  > *Thank you for watching!"*

---

## Recording Tips for Maximum Marks

1. **Terminal Font & Colors**:
   - Use a clean terminal font (Consolas, Cascadia Code, or Fira Code) at font size 16pt so viewers can clearly read epoch times.
   - Use PowerShell with standard dark background.
2. **Pre-caching Dataset**:
   - Run the script once before recording so the dataset is already cached and loads in 1-2 seconds without downloading during the recording.
3. **Pacing**:
   - Speak clearly and deliberately. The script is structured with natural pauses between terminal outputs and slides.

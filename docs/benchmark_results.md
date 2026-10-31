# CST435 Parallel Processing - Scalability Benchmark Results

**Model**: CNN | **Epochs**: 1 | **Batch Size**: 128 (global, strong scaling)

- **Host**: Intel Core i5-1135G7 (4 physical cores / 8 logical processors), 12 GB RAM, Windows 64-bit
- **Software**: Python 3.12.5, PyTorch 2.13.0+cpu, Gloo backend over loopback TCP
- **Timing**: Execution Time is the synchronized training-loop time measured with `time.perf_counter()`.

| Process Count ($p$) | Execution Time $T_p$ (s) | Epoch Time (s) | Throughput (img/s) | Speedup $S_p$ | Efficiency $E_p$ (%) | Val Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seq (1 thread) | 49.69 | 45.39 | 1207.5 | 1.000x | 100.0% | 98.47% |
| 1 | 48.49 | 44.33 | 1237.3 | 1.025x | 102.5% | 98.46% |
| 2 | 36.26 | 31.70 | 1654.6 | 1.370x | 68.5% | 98.53% |
| 4 | 33.25 | 29.22 | 1804.7 | 1.495x | 37.4% | 98.50% |
| 8 | 52.78 | 47.69 | 1136.9 | 0.942x | 11.8% | 98.57% |

**Measured result:** Speedup peaks at **1.495x** with `p=4` workers. At `p=8`, every logical processor is occupied, but the eight workers share only four physical cores; SMT contention plus higher collective participation makes the run slower than the sequential baseline (`0.942x`). This is a measured result, not a projection.

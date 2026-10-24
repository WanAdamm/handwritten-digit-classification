# CST435 Parallel Processing - Scalability Benchmark Results

**Model**: CNN | **Epochs**: 1 | **Batch Size**: 128  

| Process Count ($p$) | Execution Time $T_p$ (s) | Epoch Time (s) | Throughput (img/s) | Speedup $S_p$ | Efficiency $E_p$ (%) | Val Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seq (1 thread) | 40.20 | 36.72 | 1492.7 | 1.000x | 100.0% | 98.47% |
| 1 | 39.66 | 36.20 | 1512.9 | 1.014x | 101.4% | 98.46% |
| 2 | 30.17 | 26.51 | 1989.0 | 1.332x | 66.6% | 98.53% |
| 4 | 27.04 | 23.06 | 2219.3 | 1.487x | 37.2% | 98.50% |
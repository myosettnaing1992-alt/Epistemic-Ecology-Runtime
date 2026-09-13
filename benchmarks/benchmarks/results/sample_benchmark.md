# Benchmark Summary (Schematic Sample)

> **Note:** These numbers are illustrative and hardware-dependent. Run
> `python benchmarks/run_benchmarks.py` to reproduce on your machine.

| Graph | n | Method | Wall-clock (s) | Updates | Speedup |
|---|---|---|---|---|---|
| MVE | 32 | Cyclic GS | 0.012 ± 0.000 | 384 | 1.00x |
| MVE | 32 | Random priority | 0.008 ± 0.000 | 478 | 1.50x |
| MVE | 32 | Hybrid priority | 0.009 ± 0.000 | 352 | 1.33x |
| BA | 500 | Cyclic GS | 0.245 ± 0.000 | 15,500 | 1.00x |
| BA | 500 | Random priority | 0.198 ± 0.000 | 14,200 | 1.24x |
| BA | 500 | Hybrid priority | 0.062 ± 0.000 | 1,403 | **3.95x** |
| BA | 10000 | Cyclic GS | 12.340 ± 0.000 | 8,420 | 1.00x |
| BA | 10000 | Random priority | 13.210 ± 0.000 | 9,100 | 0.93x |
| BA | 10000 | Hybrid priority | 7.890 ± 0.000 | 5,230 | **1.56x** |
| ER | 10000 | Cyclic GS | 11.870 ± 0.000 | 7,890 | 1.00x |
| ER | 10000 | Random priority | 12.450 ± 0.000 | 8,200 | 0.95x |
| ER | 10000 | Hybrid priority | 11.920 ± 0.000 | 7,950 | 1.00x |

## Observations

1. **Topology-dependent speedup.** The hybrid scheduler achieves its
   largest gain (3.95x on BA-500) when the residual mass concentrates on
   a small set of high-degree hubs. On ER-10^4, where residual mass is
   uniformly distributed, the aging term is dominated by the residual
   term and the scheduler behaves like random priority.

2. **Graph-size effect.** On BA-500, the hybrid scheduler achieves
   3.95x; on BA-10^4, only 1.56x. Larger graphs dilute the residual
   concentration that drives aging-based prioritization.

3. **Update-count vs wall-clock.** Coordinate updates are not directly
   comparable across methods because the hybrid scheduler performs one
   cyclic sweep per `M` priority updates. Wall-clock time is the primary
   metric.

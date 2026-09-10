#!/usr/bin/env python3
"""Automated Benchmark Suite for Epistemic Ecology Runtime (EER).

Compares projected Cyclic Gauss-Seidel (CGS) against the Numba hybrid
priority coordinate-descent scheduler on sparse epistemic graphs.

Run:
    python benchmarks/run_benchmarks.py

Output:
    benchmarks/results/benchmark_summary.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from eer.core_graph import EpistemicGraph
from eer.hessian_builder import (
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
)
from eer.schedulers import run_hybrid_priority_scheduler_optimized


# ---------------------------------------------------------------------------
# Projected Cyclic Gauss-Seidel baseline
# ---------------------------------------------------------------------------

def run_cyclic_gauss_seidel(H_csr, b, max_sweeps=2000, tol=1e-6):
    """Projected CGS with box constraints on [0, 1]."""
    n = H_csr.shape[0]
    x = np.full(n, 0.5, dtype=np.float64)
    diag = H_csr.diagonal()

    start = time.perf_counter()
    sweeps = 0
    for _ in range(max_sweeps):
        max_diff = 0.0
        for i in range(n):
            row = slice(H_csr.indptr[i], H_csr.indptr[i + 1])
            cols = H_csr.indices[row]
            vals = H_csr.data[row]
            s = np.dot(vals, x[cols]) - diag[i] * x[i]
            x_new = (b[i] - s) / diag[i]
            x_new = min(1.0, max(0.0, x_new))
            diff = abs(x_new - x[i])
            if diff > max_diff:
                max_diff = diff
            x[i] = x_new
        sweeps += 1
        if max_diff < tol:
            break

    return sweeps, time.perf_counter() - start, x


# ---------------------------------------------------------------------------
# EER hybrid priority solver
# ---------------------------------------------------------------------------

def run_eer_solver(H_csr, b, max_sweeps=2000, tol=1e-6):
    """Numba hybrid priority coordinate-descent solver."""
    H_csc = H_csr.tocsc()
    x0 = np.full(H_csr.shape[0], 0.5, dtype=np.float64)

    start = time.perf_counter()
    x_star, total_updates, _ = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0, M=H_csr.shape[0], epsilon=1e-3,
        max_sweeps=max_sweeps, tol=tol,
    )
    return total_updates, time.perf_counter() - start, x_star


# ---------------------------------------------------------------------------
# JIT warm-up
# ---------------------------------------------------------------------------

def warmup_jit():
    n = 50
    g = EpistemicGraph(n)
    for i in range(n - 1):
        g.add_support_edge(i, i + 1, 1.0)
    H = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
    _ = run_eer_solver(H, g.b, max_sweeps=5, tol=1e-2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Warming up Numba JIT...")
    warmup_jit()
    print("JIT compilation complete.\n")

    node_scales = [1000, 10000, 50000]
    n_seeds = 3
    records = []

    for n in node_scales:
        cgs_times, eer_times = [], []
        cgs_sweeps_list, eer_updates_list = [], []

        for seed in range(42, 42 + n_seeds):
            rng = np.random.default_rng(seed)
            graph = EpistemicGraph(n)
            graph.b = rng.uniform(0.0, 1.0, size=n)
            graph.lambda_vec = rng.uniform(0.5, 2.0, size=n)

            for _ in range(n * 2):
                u, v = rng.integers(0, n, size=2)
                if u != v:
                    graph.add_support_edge(
                        int(u), int(v), float(rng.uniform(0.5, 1.5))
                    )

            Q_cyc = build_cycle_matrix_fundamental(graph)
            H = assemble_extended_hessian(
                graph, gamma=0.05, Q_cycle=Q_cyc
            )

            cgs_sw, cgs_t, _ = run_cyclic_gauss_seidel(H, graph.b)
            eer_upd, eer_t, _ = run_eer_solver(H, graph.b)

            cgs_sweeps_list.append(cgs_sw)
            eer_updates_list.append(eer_upd)
            cgs_times.append(cgs_t)
            eer_times.append(eer_t)

        mean_cgs_t, std_cgs_t = np.mean(cgs_times), np.std(cgs_times)
        mean_eer_t, std_eer_t = np.mean(eer_times), np.std(eer_times)
        speedup = mean_cgs_t / max(mean_eer_t, 1e-9)

        records.append({
            "Nodes": n,
            "CGS_Sweeps": int(np.mean(cgs_sweeps_list)),
            "EER_Updates": int(np.mean(eer_updates_list)),
            "CGS_Time_s": round(mean_cgs_t, 4),
            "CGS_Time_std": round(std_cgs_t, 4),
            "EER_Time_s": round(mean_eer_t, 4),
            "EER_Time_std": round(std_eer_t, 4),
            "Speedup": round(speedup, 2),
        })

    df = pd.DataFrame(records)
    csv_path = results_dir / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)

    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    print(f"[OK] Saved to {csv_path}")
    print("Next: python benchmarks/format_benchmark.py")


if __name__ == "__main__":
    main()

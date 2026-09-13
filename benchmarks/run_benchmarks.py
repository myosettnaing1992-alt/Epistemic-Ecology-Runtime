"""
Benchmark runner for EER.

Compares three schedulers across three graph families:
    - Cyclic Gauss-Seidel (CGS)
    - Random priority
    - Hybrid priority (Algorithm 4)

Outputs a JSON file with wall-clock time, update counts, and residual.

Usage:
    python benchmarks/run_benchmarks.py
    python benchmarks/run_benchmarks.py --smoke
    python benchmarks/run_benchmarks.py --out results/benchmark.json
    python benchmarks/run_benchmarks.py --graphs ba er mve --n 500 10000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eer import (                       # noqa: E402
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
    run_random_priority_scheduler,
)
from eer.utils import (                 # noqa: E402
    make_ba_graph,
    make_er_graph,
    residual_inf,
    set_random_priors,
)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class BenchmarkRecord:
    graph: str
    n: int
    m_S: int
    method: str
    wallclock_sec: float
    coordinate_updates: int
    final_residual: float
    converged: bool
    seed: int


# ----------------------------------------------------------------------
# Graph builders
# ----------------------------------------------------------------------

def build_mve_graph(n_nodes: int = 32) -> EpistemicGraph:
    """Chain-like MVE graph (paper's small example)."""
    import networkx as nx
    G_nx = nx.path_graph(n_nodes)
    g = EpistemicGraph(num_nodes=n_nodes)
    set_random_priors(g, seed=0)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


def build_graph(kind: str, n: int, seed: int = 0) -> EpistemicGraph:
    if kind == "mve":
        return build_mve_graph(min(n, 32))

    if kind == "ba":
        G_nx = make_ba_graph(n, m=3, seed=seed)
    elif kind == "er":
        G_nx = make_er_graph(n, avg_deg=5.0, seed=seed)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")

    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


# ----------------------------------------------------------------------
# Assemble system once per (graph, n, seed)
# ----------------------------------------------------------------------

def assemble_system(
    kind: str, n: int, seed: int = 0, gamma: float = 0.05
) -> tuple[EpistemicGraph, csr_matrix, csr_matrix]:
    g = build_graph(kind, n, seed=seed)
    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=gamma, Q_cycle=Q_cyc)
    return g, H, Q_cyc


# ----------------------------------------------------------------------
# Single benchmark run
# ----------------------------------------------------------------------

def run_one(
    kind: str,
    n: int,
    seed: int = 0,
    tol: float = 1e-6,
    max_sweeps: int = 100_000,
    verbose: bool = False,
) -> list[BenchmarkRecord]:

    g, H, _ = assemble_system(kind, n, seed=seed)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    b = g.b
    x0 = np.full(n, 0.5, dtype=np.float64)

    m_S = int(len(g._src))

    records: list[BenchmarkRecord] = []

    # ----------------------------------------------------------------
    # 1. Cyclic Gauss-Seidel
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [CGS] n={n}...", end=" ", flush=True)
    t0 = time.perf_counter()
    x, updates, res = run_cyclic_gs_scheduler(
        H, b, x0, tol=tol, max_sweeps=max_sweeps
    )
    t1 = time.perf_counter()
    records.append(BenchmarkRecord(
        graph=kind, n=n, m_S=m_S, method="cyclic_gs",
        wallclock_sec=t1 - t0,
        coordinate_updates=updates,
        final_residual=res,
        converged=res < tol,
        seed=seed,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, updates={updates}, res={res:.2e}")

    # ----------------------------------------------------------------
    # 2. Random priority
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [RND] n={n}...", end=" ", flush=True)
    t0 = time.perf_counter()
    x, updates, res = run_random_priority_scheduler(
        H, b, x0, tol=tol, max_updates=max_sweeps * n, seed=seed
    )
    t1 = time.perf_counter()
    records.append(BenchmarkRecord(
        graph=kind, n=n, m_S=m_S, method="random_priority",
        wallclock_sec=t1 - t0,
        coordinate_updates=updates,
        final_residual=res,
        converged=res < tol,
        seed=seed,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, updates={updates}, res={res:.2e}")

    # ----------------------------------------------------------------
    # 3. Hybrid priority
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [HYP] n={n}...", end=" ", flush=True)
    t0 = time.perf_counter()
    x, updates, res = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0,
        M=n,
        epsilon=1e-3,
        tol=tol,
        max_sweeps=max_sweeps,
    )
    t1 = time.perf_counter()
    records.append(BenchmarkRecord(
        graph=kind, n=n, m_S=m_S, method="hybrid_priority",
        wallclock_sec=t1 - t0,
        coordinate_updates=updates,
        final_residual=res,
        converged=res < tol,
        seed=seed,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, updates={updates}, res={res:.2e}")

    return records


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Fast run: mve only, tiny n.")
    p.add_argument("--graphs", nargs="+", default=["mve", "ba", "er"],
                   choices=["mve", "ba", "er"])
    p.add_argument("--n", nargs="+", type=int, default=[32, 500, 10000])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", type=str, default="results/benchmark.json")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.graphs = ["mve"]
        args.n = [32]
        args.seeds = [0]

    # Filter: skip huge n for smoke / small machines
    configs: list[tuple[str, int]] = []
    for kind in args.graphs:
        for n in args.n:
            if kind == "mve" and n > 32:
                continue
            configs.append((kind, n))

    all_records: list[BenchmarkRecord] = []
    verbose = not args.quiet

    for kind, n in configs:
        for seed in args.seeds:
            if verbose:
                print(f"\n=== Graph={kind}, n={n}, seed={seed} ===")
            try:
                recs = run_one(kind, n, seed=seed,
                               tol=args.tol, verbose=verbose)
                all_records.extend(recs)
            except Exception as e:
                print(f"  [FAIL] {kind} n={n} seed={seed}: {e}",
                      file=sys.stderr)

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "timestamp": time.time(),
        },
        "records": [asdict(r) for r in all_records],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'graph':>6} {'n':>7} {'method':>18} "
          f"{'wallclock':>10} {'updates':>10} {'residual':>12}")
    for r in all_records:
        print(f"{r.graph:>6} {r.n:>7} {r.method:>18} "
              f"{r.wallclock_sec:>10.4f} {r.coordinate_updates:>10} "
              f"{r.final_residual:>12.2e}")


if __name__ == "__main__":
    main()

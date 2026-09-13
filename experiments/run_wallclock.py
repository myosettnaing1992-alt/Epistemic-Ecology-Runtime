"""
Wall-clock timing experiment for Section 7.2.

Reproduces Table 9.1 (coordinate updates) and reports wall-clock time to
reach residual tolerance 1e-6, comparing:

    - Cyclic Gauss-Seidel (CGS)
    - Random priority
    - Hybrid priority (aged residual, Algorithm 4)

Usage:
    python experiments/run_wallclock.py --graph ba --n 500
    python experiments/run_wallclock.py --graph ba --n 10000
    python experiments/run_wallclock.py --all
    python experiments/run_wallclock.py --all --out results/table_9_1.json

Outputs
-------
JSON file at --out (default: results/table_9_1.json) containing per-method
records, plus a printed summary table.
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
    set_random_priors,
)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class TimingResult:
    graph: str
    n: int
    m_S: int
    method: str
    wallclock_sec: float
    coordinate_updates: int
    final_residual: float
    converged: bool
    seed: int
    tol: float


# ----------------------------------------------------------------------
# Graph generators
# ----------------------------------------------------------------------

def make_mve_graph(n_nodes: int = 32) -> EpistemicGraph:
    """Chain-like MVE graph (paper's small example)."""
    import networkx as nx

    G_nx = nx.path_graph(n_nodes)
    g = EpistemicGraph(num_nodes=n_nodes)
    set_random_priors(g, seed=0)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


def make_graph(kind: str, n: int, seed: int = 0) -> EpistemicGraph:
    """Build an EpistemicGraph for the requested family."""
    if kind == "mve":
        return make_mve_graph(min(n, 32))

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
    kind: str, n: int, seed: int = 0, gamma: float = 0.05,
) -> tuple[EpistemicGraph, "csr_matrix"]:
    g = make_graph(kind, n, seed=seed)
    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=gamma, Q_cycle=Q_cyc)
    return g, H


# ----------------------------------------------------------------------
# Single-run benchmark
# ----------------------------------------------------------------------

def run_one(
    kind: str,
    n: int,
    seed: int = 0,
    tol: float = 1e-6,
    max_sweeps: int = 100_000,
    verbose: bool = True,
) -> list[TimingResult]:
    """Run all three schedulers on one (kind, n, seed) configuration."""
    import tracemalloc

    g, H = assemble_system(kind, n, seed=seed)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    b = g.b
    x0 = np.full(n, 0.5, dtype=np.float64)

    m_S = int(len(g._src))
    results: list[TimingResult] = []

    # ----------------------------------------------------------------
    # 1. Cyclic Gauss-Seidel
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [CGS] n={n:>6} seed={seed} ...", end=" ", flush=True)
    tracemalloc.start()
    t0 = time.perf_counter()
    x, updates, res = run_cyclic_gs_scheduler(
        H, b, x0, tol=tol, max_sweeps=max_sweeps,
    )
    t1 = time.perf_counter()
    _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results.append(TimingResult(
        graph=kind, n=n, m_S=m_S, method="cyclic_gs",
        wallclock_sec=t1 - t0,
        coordinate_updates=int(updates),
        final_residual=float(res),
        converged=res < tol,
        seed=seed, tol=tol,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  updates={updates:>8,}  res={res:.2e}")

    # ----------------------------------------------------------------
    # 2. Random priority
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [RND] n={n:>6} seed={seed} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    x, updates, res = run_random_priority_scheduler(
        H, b, x0, tol=tol, max_updates=max_sweeps * n, seed=seed,
    )
    t1 = time.perf_counter()
    results.append(TimingResult(
        graph=kind, n=n, m_S=m_S, method="random_priority",
        wallclock_sec=t1 - t0,
        coordinate_updates=int(updates),
        final_residual=float(res),
        converged=res < tol,
        seed=seed, tol=tol,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  updates={updates:>8,}  res={res:.2e}")

    # ----------------------------------------------------------------
    # 3. Hybrid priority
    # ----------------------------------------------------------------
    if verbose:
        print(f"  [HYP] n={n:>6} seed={seed} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    x, updates, res = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0,
        M=n,            # backbone period = n (Theorem 9.1)
        epsilon=1e-3,   # aging rate
        tol=tol,
        max_sweeps=max_sweeps,
    )
    t1 = time.perf_counter()
    results.append(TimingResult(
        graph=kind, n=n, m_S=m_S, method="hybrid_priority",
        wallclock_sec=t1 - t0,
        coordinate_updates=int(updates),
        final_residual=float(res),
        converged=res < tol,
        seed=seed, tol=tol,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  updates={updates:>8,}  res={res:.2e}")

    return results


# ----------------------------------------------------------------------
# Aggregation across seeds
# ----------------------------------------------------------------------

def aggregate(records: list[TimingResult]) -> dict:
    """Aggregate records across seeds for each (graph, n, method)."""
    from collections import defaultdict

    buckets: dict[tuple, list[TimingResult]] = defaultdict(list)
    for r in records:
        buckets[(r.graph, r.n, r.method)].append(r)

    agg = {}
    for key, rs in buckets.items():
        wall = np.array([r.wallclock_sec for r in rs])
        upd = np.array([r.coordinate_updates for r in rs])
        res = np.array([r.final_residual for r in rs])
        agg[key] = {
            "wallclock_mean": float(wall.mean()),
            "wallclock_std": float(wall.std()),
            "updates_mean": float(upd.mean()),
            "updates_std": float(upd.std()),
            "residual_mean": float(res.mean()),
            "n_seeds": len(rs),
        }
    return agg


# ----------------------------------------------------------------------
# Markdown table
# ----------------------------------------------------------------------

METHOD_LABELS = {
    "cyclic_gs": "Cyclic GS",
    "random_priority": "Random priority",
    "hybrid_priority": "Hybrid priority",
}


def to_markdown(agg: dict) -> str:
    """Reproduce Table 9.1 in markdown format."""
    from collections import defaultdict

    groups: dict[tuple, dict] = defaultdict(dict)
    for (graph, n, method), stats in agg.items():
        groups[(graph, n)][method] = stats

    lines = []
    lines.append("| Graph | n | Method | Wall-clock (s) | Updates | Speedup |")
    lines.append("|---|---|---|---|---|---|")

    for (graph, n), methods in sorted(groups.items()):
        base = methods.get("cyclic_gs")
        if base is None:
            continue
        base_wall = base["wallclock_mean"]
        base_upd = base["updates_mean"]

        for method in ["cyclic_gs", "random_priority", "hybrid_priority"]:
            if method not in methods:
                continue
            s = methods[method]
            wall = s["wallclock_mean"]
            wall_std = s["wallclock_std"]
            upd = int(s["updates_mean"])
            upd_speedup = base_upd / s["updates_mean"] if s["updates_mean"] > 0 else 0.0
            wall_speedup = base_wall / wall if wall > 0 else 0.0

            lines.append(
                f"| {graph.upper()} | {n:,} | {METHOD_LABELS[method]} "
                f"| {wall:.3f} ± {wall_std:.3f} "
                f"| {upd:,} "
                f"| {upd_speedup:.2f}x (wall: {wall_speedup:.2f}x) |"
            )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Wall-clock timing benchmark for EER schedulers.",
    )
    p.add_argument(
        "--graph", choices=["mve", "ba", "er"], default="ba",
        help="Graph family (default: ba).",
    )
    p.add_argument(
        "--n", type=int, default=500,
        help="Number of nodes (default: 500).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Run all standard configurations from the paper.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0],
        help="Random seeds (default: [0]).",
    )
    p.add_argument(
        "--tol", type=float, default=1e-6,
        help="Residual tolerance (default: 1e-6).",
    )
    p.add_argument(
        "--out", type=str, default="results/table_9_1.json",
        help="Output JSON path (default: results/table_9_1.json).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run progress.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.all:
        configs = [
            ("mve", 32),
            ("ba", 500),
            ("ba", 10_000),
            ("er", 10_000),
        ]
    else:
        configs = [(args.graph, args.n)]

    all_records: list[TimingResult] = []
    verbose = not args.quiet

    for kind, n in configs:
        for seed in args.seeds:
            if verbose:
                print(f"\n=== Graph={kind.upper()}, n={n:,}, seed={seed} ===")
            try:
                recs = run_one(
                    kind, n, seed=seed, tol=args.tol, verbose=verbose,
                )
                all_records.extend(recs)
            except Exception as e:
                print(f"  [FAIL] {kind} n={n} seed={seed}: {e}",
                      file=sys.stderr)

    if not all_records:
        print("[error] No records produced.", file=sys.stderr)
        sys.exit(1)

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "timestamp": time.time(),
            "tol": args.tol,
            "seeds": args.seeds,
        },
        "records": [asdict(r) for r in all_records],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Aggregate + markdown
    agg = aggregate(all_records)
    table = to_markdown(agg)
    print("\n=== Table 9.1 ===")
    print(table)

    # Also save markdown
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Table 9.1 — Scheduler comparison\n\n")
        f.write(table + "\n")
    print(f"\nSaved → {md_path}")


if __name__ == "__main__":
    main()

"""
Scalability sweep for Section 7.4.

Measures wall-clock time per iteration and peak memory as a function of
graph size:

    n ∈ {500, 1000, 2500, 5000, 10000}

on BA scale-free graphs, with both CGS and hybrid priority.

Usage:
    python experiments/run_scalability.py
    python experiments/run_scalability.py --sizes 500 1000 2500 5000
    python experiments/run_scalability.py --out results/scalability.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eer import (                       # noqa: E402
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
)
from eer.utils import make_ba_graph, set_random_priors   # noqa: E402


@dataclass
class ScalabilityRecord:
    graph: str
    n: int
    m_S: int
    method: str
    wallclock_sec: float
    coordinate_updates: int
    sweeps: int
    time_per_iter_ms: float
    peak_memory_mb: float
    final_residual: float
    converged: bool


def _make_system(n: int, seed: int = 0):
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc, L_max=3)
    return g, H


def run_one(n: int, seed: int = 0, tol: float = 1e-6, verbose: bool = True):
    g, H = _make_system(n, seed=seed)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    b = g.b
    x0 = np.full(n, 0.5, dtype=np.float64)

    records: list[ScalabilityRecord] = []

    # --- CGS ---
    if verbose:
        print(f"  [CGS] n={n:>6} ...", end=" ", flush=True)
    tracemalloc.start()
    t0 = time.perf_counter()
    _, upd, res = run_cyclic_gs_scheduler(H, b, x0, tol=tol, max_sweeps=200_000)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sweeps = upd // n
    records.append(ScalabilityRecord(
        graph="ba", n=n, m_S=int(len(g._src)), method="cyclic_gs",
        wallclock_sec=t1 - t0, coordinate_updates=int(upd),
        sweeps=sweeps,
        time_per_iter_ms=(t1 - t0) * 1000.0 / max(upd, 1),
        peak_memory_mb=peak / 1e6,
        final_residual=float(res), converged=res < tol,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  iter={upd:,}  res={res:.2e}")

    # --- Hybrid ---
    if verbose:
        print(f"  [HYP] n={n:>6} ...", end=" ", flush=True)
    tracemalloc.start()
    t0 = time.perf_counter()
    _, upd_h, res_h = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0, M=n, epsilon=1e-3, tol=tol, max_sweeps=200_000,
    )
    t1 = time.perf_counter()
    _, peak_h = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    records.append(ScalabilityRecord(
        graph="ba", n=n, m_S=int(len(g._src)), method="hybrid_priority",
        wallclock_sec=t1 - t0, coordinate_updates=int(upd_h),
        sweeps=int(upd_h // n) + 1,
        time_per_iter_ms=(t1 - t0) * 1000.0 / max(upd_h, 1),
        peak_memory_mb=peak_h / 1e6,
        final_residual=float(res_h), converged=res_h < tol,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  iter={upd_h:,}  res={res_h:.2e}")

    return records


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_scaling(records: list[ScalabilityRecord], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for method, style in [
        ("cyclic_gs", {"color": "C0", "label": "Cyclic GS", "marker": "o"}),
        ("hybrid_priority", {"color": "C3", "label": "Hybrid", "marker": "s"}),
    ]:
        subset = [r for r in records if r.method == method]
        subset.sort(key=lambda r: r.n)
        ns = [r.n for r in subset]

        axes[0].loglog(ns, [r.time_per_iter_ms for r in subset],
                       color=style["color"], marker=style["marker"],
                       label=style["label"])
        axes[1].loglog(ns, [r.peak_memory_mb for r in subset],
                       color=style["color"], marker=style["marker"],
                       label=style["label"])

    axes[0].set_xlabel("Number of nodes (n)")
    axes[0].set_ylabel("Time per coordinate update (ms)")
    axes[0].set_title("Per-iteration scaling")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Number of nodes (n)")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("Memory scaling")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[500, 1000, 2500, 5000, 10000])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--out", type=str, default="results/scalability.json")
    p.add_argument("--fig", type=str, default="figures/scaling.pdf")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    all_records: list[ScalabilityRecord] = []
    for n in args.sizes:
        for seed in args.seeds:
            if verbose:
                print(f"\n=== n={n:,}  seed={seed} ===")
            try:
                recs = run_one(n, seed=seed, tol=args.tol, verbose=verbose)
                all_records.extend(recs)
            except Exception as e:
                print(f"  [FAIL] n={n}: {e}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "python": sys.version,
                "platform": platform.platform(),
                "sizes": args.sizes,
                "seeds": args.seeds,
            },
            "records": [asdict(r) for r in all_records],
        }, f, indent=2)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        plot_scaling(all_records, Path(args.fig))


if __name__ == "__main__":
    main()

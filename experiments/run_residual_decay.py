"""
Residual decay experiment for Figure 11.1.

Reproduces the log-residual-vs-sweeps plot on a BA scale-free graph
(n = 500), comparing:

    - Cyclic Gauss-Seidel
    - Random priority
    - Hybrid priority (aged residual, Algorithm 4)

Usage:
    python experiments/run_residual_decay.py
    python experiments/run_residual_decay.py --n 500 --out figures/residual_decay.pdf
    python experiments/run_residual_decay.py --n 500 --no-plot --out results/residual_decay.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eer import (                       # noqa: E402
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
)
from eer.schedulers import (            # noqa: E402
    _residual_inf,
    _update_coord,
)
from eer.utils import make_ba_graph, set_random_priors   # noqa: E402


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class DecayRecord:
    graph: str
    n: int
    method: str
    wallclock_sec: float
    sweeps: int
    updates: int
    residual_history: list[float] = field(default_factory=list)
    final_residual: float = float("inf")
    converged: bool = False


# ----------------------------------------------------------------------
# Instrumented solvers (record residual every full sweep)
# ----------------------------------------------------------------------

def _build_system(n: int, seed: int = 0, gamma: float = 0.05):
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=gamma, Q_cycle=Q_cyc, L_max=3)
    return g, H


def solve_cgs_with_history(
    H, b, x0, tol: float = 1e-6, max_sweeps: int = 5000,
):
    """Cyclic Gauss-Seidel with per-sweep residual recording."""
    n = H.shape[0]
    H_csr = H.tocsr()
    indptr = H_csr.indptr.astype(np.int64)
    indices = H_csr.indices.astype(np.int64)
    data = H_csr.data.astype(np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)

    history: list[float] = []
    updates = 0
    for sweep in range(max_sweeps):
        for i in range(n):
            x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
        updates += n
        res = _residual_inf(x, indptr, indices, data, b)
        history.append(res)
        if res < tol:
            return x, updates, sweep + 1, history, True
    return x, updates, max_sweeps, history, False


def solve_random_with_history(
    H, b, x0, tol: float = 1e-6, max_updates: int = 500_000, seed: int = 0,
):
    """Random priority with per-sweep (every n updates) residual recording."""
    n = H.shape[0]
    H_csr = H.tocsr()
    indptr = H_csr.indptr.astype(np.int64)
    indices = H_csr.indices.astype(np.int64)
    data = H_csr.data.astype(np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)

    rng = np.random.default_rng(seed)
    history: list[float] = []
    updates = 0
    sweeps = 0

    while updates < max_updates:
        i = int(rng.integers(0, n))
        x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
        updates += 1
        if updates % n == 0:
            sweeps += 1
            res = _residual_inf(x, indptr, indices, data, b)
            history.append(res)
            if res < tol:
                return x, updates, sweeps, history, True
    return x, updates, sweeps, history, False


def solve_hybrid_with_history(
    H, b, x0, M: int, epsilon: float = 1e-3,
    tol: float = 1e-6, max_sweeps: int = 5000,
):
    """Hybrid priority with per-sweep residual recording."""
    import heapq

    n = H.shape[0]
    H_csr = H.tocsr()
    indptr = H_csr.indptr.astype(np.int64)
    indices = H_csr.indices.astype(np.int64)
    data = H_csr.data.astype(np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)

    r = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        r[i] = s - b[i]

    tau = np.zeros(n, dtype=np.int64)
    pq = [(-abs(r[i]), i) for i in range(n)]
    heapq.heapify(pq)

    history: list[float] = []
    updates = 0
    sweeps = 0
    k = 0

    while sweeps < max_sweeps:
        if k % M == 0:
            for i in range(n):
                x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
                tau[i] = 0
            for i in range(n):
                s = 0.0
                for kk in range(indptr[i], indptr[i + 1]):
                    s += data[kk] * x[indices[kk]]
                r[i] = s - b[i]
            updates += n
        else:
            if not pq:
                pq = [(-(abs(r[i]) + epsilon * tau[i]), i) for i in range(n)]
                heapq.heapify(pq)
            _, i = heapq.heappop(pq)
            x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
            tau[i] = 0
            updates += 1

            # Recompute residual at i
            s = 0.0
            for kk in range(indptr[i], indptr[i + 1]):
                s += data[kk] * x[indices[kk]]
            r[i] = s - b[i]
            heapq.heappush(pq, (-(abs(r[i]) + epsilon * tau[i]), i))

            # Recompute residual for neighbors (approximate, only push)
            for kk in range(indptr[i], indptr[i + 1]):
                j = int(indices[kk])
                if j == i:
                    continue
                tau[j] += 1
                heapq.heappush(pq, (-(abs(r[j]) + epsilon * tau[j]), j))

        k += 1
        sweeps += 1
        res = float(np.max(np.abs(r)))
        history.append(res)
        if res < tol:
            return x, updates, sweeps, history, True

    return x, updates, sweeps, history, False


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def run_one(n: int, seed: int = 0, tol: float = 1e-6, verbose: bool = True):
    g, H = _build_system(n, seed=seed)
    x0 = np.full(n, 0.5, dtype=np.float64)
    b = g.b

    records: list[DecayRecord] = []

    # --- CGS ---
    if verbose:
        print(f"  [CGS] n={n} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    _, upd, sw, hist, conv = solve_cgs_with_history(H, b, x0, tol=tol)
    t1 = time.perf_counter()
    records.append(DecayRecord(
        graph="ba", n=n, method="cyclic_gs",
        wallclock_sec=t1 - t0, sweeps=sw, updates=upd,
        residual_history=hist, final_residual=hist[-1], converged=conv,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, sweeps={sw}, res={hist[-1]:.2e}")

    # --- Random ---
    if verbose:
        print(f"  [RND] n={n} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    _, upd, sw, hist, conv = solve_random_with_history(H, b, x0, tol=tol, seed=seed)
    t1 = time.perf_counter()
    records.append(DecayRecord(
        graph="ba", n=n, method="random_priority",
        wallclock_sec=t1 - t0, sweeps=sw, updates=upd,
        residual_history=hist, final_residual=hist[-1], converged=conv,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, sweeps={sw}, res={hist[-1]:.2e}")

    # --- Hybrid ---
    if verbose:
        print(f"  [HYP] n={n} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    _, upd, sw, hist, conv = solve_hybrid_with_history(
        H, b, x0, M=n, epsilon=1e-3, tol=tol,
    )
    t1 = time.perf_counter()
    records.append(DecayRecord(
        graph="ba", n=n, method="hybrid_priority",
        wallclock_sec=t1 - t0, sweeps=sw, updates=upd,
        residual_history=hist, final_residual=hist[-1], converged=conv,
    ))
    if verbose:
        print(f"{t1 - t0:.3f}s, sweeps={sw}, res={hist[-1]:.2e}")

    return records


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_decay(records: list[DecayRecord], out_path: Path) -> None:
    """Plot residual decay (Figure 11.1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    styles = {
        "cyclic_gs":       {"color": "C0", "linestyle": "-",  "label": "Cyclic GS"},
        "random_priority": {"color": "C2", "linestyle": "--", "label": "Random priority"},
        "hybrid_priority": {"color": "C3", "linestyle": "-.", "label": "Hybrid priority"},
    }

    for r in records:
        s = styles.get(r.method, {})
        ax.semilogy(
            range(1, len(r.residual_history) + 1),
            r.residual_history,
            **s,
        )

    ax.set_xlabel("Effective sweeps")
    ax.set_ylabel(r"$\|H_{\mathrm{ext}} x - b\|_\infty$")
    ax.set_title(f"Residual decay on BA scale-free graph (n={records[0].n})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

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
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--out", type=str, default="figures/residual_decay.pdf")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"=== Residual decay on BA graph (n={args.n}) ===")
    records = run_one(args.n, seed=args.seed, tol=args.tol)

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "python": sys.version,
                "platform": platform.platform(),
                "n": args.n,
                "seed": args.seed,
                "tol": args.tol,
            },
            "records": [asdict(r) for r in records],
        }, f, indent=2)
    print(f"Saved JSON  → {json_path}")

    if not args.no_plot:
        plot_decay(records, out_path)


if __name__ == "__main__":
    main()

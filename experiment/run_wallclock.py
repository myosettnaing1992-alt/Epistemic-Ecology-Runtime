"""
Wall-clock timing experiment for Section 7.2.

Reproduces Table 9.1 (coordinate updates) and reports wall-clock time
to reach residual tolerance 1e-6, comparing:
    - Cyclic Gauss-Seidel (PCGS)
    - Random priority
    - Hybrid priority (aged residual)

Usage:
    python experiments/run_wallclock.py --graph ba --n 10000
    python experiments/run_wallclock.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

# Adjust to your repo layout
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cycle_basis import (                       # noqa: E402
    build_fundamental_cycle_basis,
    build_cycle_precision_matrix,
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
    sweeps: int
    coordinate_updates: int
    wallclock_sec: float
    peak_memory_mb: float
    converged: bool


# ----------------------------------------------------------------------
# Graph generators
# ----------------------------------------------------------------------

def make_graph(kind: str, n: int, seed: int = 0) -> nx.Graph:
    if kind == "mve":
        # Chain of 32 nodes (as in paper)
        return nx.path_graph(min(n, 32))
    if kind == "ba":
        m = max(2, int(round(3)))       # ~3 edges per new node
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if kind == "er":
        p = 5.0 / n                     # average degree ~5
        return nx.erdos_renyi_graph(n, p, seed=seed)
    raise ValueError(f"Unknown graph kind: {kind}")


# ----------------------------------------------------------------------
# Build H_ext and b for a synthetic EER instance
# ----------------------------------------------------------------------

def build_eer_system(
    G: nx.Graph,
    alpha: float = 0.5,
    gamma: float = 0.5,
    lambda_self: float = 1.0,
    seed: int = 0,
):
    """
    Construct H_base, K_cascade, K_cycle, and b for a synthetic instance.

    - Support graph = G itself.
    - Contradiction edges = random 1% of edges.
    - Derived_from DAG = a random orientation of G (skip cycles).
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    node_list = list(G.nodes())
    idx = {v: i for i, v in enumerate(node_list)}

    # Support Laplacian: L_S = D_S - A_S (weight 1 for now)
    A_S = nx.to_scipy_sparse_array(G, nodelist=node_list, format="csr",
                                    dtype=np.float64)
    deg_S = np.asarray(A_S.sum(axis=1)).ravel()
    L_S = csr_matrix(
        (np.concatenate([deg_S, -A_S.data]),
         (np.concatenate([np.arange(n), A_S.indices]),
          np.concatenate([np.arange(n), A_S.indices]))),
        shape=(n, n),
    )
    # Simpler: use scipy's laplacian
    from scipy.sparse.csgraph import laplacian
    L_S = laplacian(A_S, normed=False).tocsr()

    # Derived-from: same as L_S but on a DAG. For synthetic purposes,
    # we approximate L_D ≈ 0.1 * L_S (paper's Remark 2.1).
    L_D = 0.1 * L_S

    # Self-consistency precision
    Lambda = lambda_self * identity(n, format="csr")

    # Base Hessian
    H_base = (Lambda + L_S + L_D).tocsr()

    # Cycle basis → K_cycle
    basis = build_fundamental_cycle_basis(G, K_max=1000)
    K_cycle = build_cycle_precision_matrix(basis, gamma=gamma)

    # Cascade: cheap proxy (identity-weighted); in production use Eq. (5.4)
    K_cascade = (0.05 * identity(n, format="csr")).tocsr()

    H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()

    # RHS: random but reproducible
    b = rng.standard_normal(n)

    return H_ext, b, K_cascade, K_cycle


# ----------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------

def solve_cyclic_gs(H, b, x0, tol=1e-6, max_sweeps=200_000):
    """
    Projected cyclic Gauss-Seidel with box constraints [-5, 5].
    Returns (x, sweeps, coord_updates, converged).
    """
    n = H.shape[0]
    x = x0.copy()
    lo, hi = -5.0, 5.0
    H_diag = H.diagonal()
    H_csr = H.tocsr()

    sweeps = 0
    for k in range(max_sweeps):
        for i in range(n):
            row_start = H_csr.indptr[i]
            row_end = H_csr.indptr[i + 1]
            s = 0.0
            for idx in range(row_start, row_end):
                j = H_csr.indices[idx]
                if j != i:
                    s += H_csr.data[idx] * x[j]
            xi_new = (b[i] - s) / H_diag[i]
            x[i] = min(hi, max(lo, xi_new))
        sweeps += 1
        r = H @ x - b
        if np.max(np.abs(r)) < tol:
            return x, sweeps, sweeps * n, True

    return x, sweeps, sweeps * n, False


def solve_random_priority(H, b, x0, tol=1e-6, max_updates=200_000, seed=0):
    """
    Random priority queue (baseline for ablation).
    Pops a uniformly-random node each iteration.
    """
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    x = x0.copy()
    lo, hi = -5.0, 5.0
    H_diag = H.diagonal()
    H_csr = H.tocsr()
    r = H @ x - b

    updates = 0
    for k in range(max_updates):
        i = int(rng.integers(0, n))
        row_start = H_csr.indptr[i]
        row_end = H_csr.indptr[i + 1]
        s = 0.0
        for idx in range(row_start, row_end):
            j = H_csr.indices[idx]
            if j != i:
                s += H_csr.data[idx] * x[j]
        x[i] = min(hi, max(lo, (b[i] - s) / H_diag[i]))
        r[i] = (H_csr.data[H_csr.indptr[i]:H_csr.indptr[i+1]] *
                x[H_csr.indices[H_csr.indptr[i]:H_csr.indptr[i+1]]]).sum() - b[i]

        # Recompute residual for neighbors lazily (approximate)
        if np.max(np.abs(r)) < tol:
            return x, updates // n + 1, updates, True
        updates += 1

    return x, updates // n + 1, updates, False


def solve_hybrid_priority(
    H, b, x0, tol=1e-6, M=50, eps=1e-4, max_updates=200_000
):
    """
    Hybrid Priority EER (Algorithm 4).
    Alternates cyclic sweep every M iterations with aging-based priority.
    """
    import heapq
    n = H.shape[0]
    x = x0.copy()
    lo, hi = -5.0, 5.0
    H_diag = H.diagonal()
    H_csr = H.tocsr()
    r = H @ x - b

    tau = np.zeros(n, dtype=np.int64)

    # Priority queue: (-priority, node)
    pq = [(-abs(r[i]), i) for i in range(n)]
    heapq.heapify(pq)

    def residual_of(i):
        s = 0.0
        for idx in range(H_csr.indptr[i], H_csr.indptr[i + 1]):
            j = H_csr.indices[idx]
            s += H_csr.data[idx] * x[j]
        return s - b[i]

    updates = 0
    k = 0
    while updates < max_updates:
        if k % M == 0:
            # Cyclic sweep
            for i in range(n):
                s = 0.0
                for idx in range(H_csr.indptr[i], H_csr.indptr[i + 1]):
                    j = H_csr.indices[idx]
                    if j != i:
                        s += H_csr.data[idx] * x[j]
                x[i] = min(hi, max(lo, (b[i] - s) / H_diag[i]))
                tau[i] = 0
            r = H @ x - b
            updates += n
        else:
            # Priority pop
            if not pq:
                pq = [(-abs(r[i]) + eps * tau[i], i) for i in range(n)]
                heapq.heapify(pq)
            _, i = heapq.heappop(pq)
            s = 0.0
            for idx in range(H_csr.indptr[i], H_csr.indptr[i + 1]):
                j = H_csr.indices[idx]
                if j != i:
                    s += H_csr.data[idx] * x[j]
            x[i] = min(hi, max(lo, (b[i] - s) / H_diag[i]))
            tau[i] = 0
            updates += 1

            # Push back with updated priority
            r[i] = residual_of(i)
            heapq.heappush(pq, (-abs(r[i]), i))

            # Push neighbors
            for idx in range(H_csr.indptr[i], H_csr.indptr[i + 1]):
                j = H_csr.indices[idx]
                if j != i:
                    r[j] = residual_of(j)
                    tau[j] += 1
                    heapq.heappush(pq, (-(abs(r[j]) + eps * tau[j]), j))

        k += 1

        if np.max(np.abs(r)) < tol:
            return x, updates // n + 1, updates, True

    return x, updates // n + 1, updates, False


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def run_one(kind: str, n: int, seed: int = 0, verbose: bool = True) -> list[TimingResult]:
    import tracemalloc

    G = make_graph(kind, n, seed=seed)
    H_ext, b, _, _ = build_eer_system(G, seed=seed)
    x0 = np.zeros(n, dtype=np.float64)

    results: list[TimingResult] = []

    for method_name, solver in [
        ("Cyclic GS", solve_cyclic_gs),
        ("Random priority", solve_random_priority),
        ("Hybrid priority", solve_hybrid_priority),
    ]:
        if verbose:
            print(f"  [{method_name}] running...")

        tracemalloc.start()
        t0 = time.perf_counter()
        x, sweeps, updates, converged = solver(H_ext, b, x0)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append(TimingResult(
            graph=kind, n=n, m_S=G.number_of_edges(),
            method=method_name,
            sweeps=sweeps, coordinate_updates=updates,
            wallclock_sec=t1 - t0,
            peak_memory_mb=peak / 1e6,
            converged=converged,
        ))

        if verbose:
            print(f"    sweeps={sweeps}, wallclock={t1 - t0:.3f}s, "
                  f"converged={converged}, peak_mem={peak/1e6:.1f}MB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=["mve", "ba", "er"], default="ba")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--all", action="store_true",
                        help="Run all three graph families.")
    parser.add_argument("--out", type=str, default="results/wallclock.json")
    args = parser.parse_args()

    if args.all:
        configs = [("mve", 32), ("ba", 500), ("ba", 10_000), ("er", 10_000)]
    else:
        configs = [(args.graph, args.n)]

    all_results: list[TimingResult] = []
    for kind, n in configs:
        print(f"\n=== Graph: {kind}, n = {n} ===")
        all_results.extend(run_one(kind, n, verbose=True))

    # Print summary table
    print("\n=== Summary (reproduces Table 9.1) ===")
    print(f"{'graph':>6} {'n':>6} {'method':>18} "
          f"{'sweeps':>10} {'updates':>10} {'wallclock(s)':>14} {'mem(MB)':>10}")
    for r in all_results:
        print(f"{r.graph:>6} {r.n:>6} {r.method:>18} "
              f"{r.sweeps:>10} {r.coordinate_updates:>10} "
              f"{r.wallclock_sec:>14.4f} {r.peak_memory_mb:>10.2f}")

    # Persist
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

"""
Coordinate-descent schedulers for EER.

- run_cyclic_gs_scheduler               : projected cyclic Gauss-Seidel
- run_random_priority_scheduler         : random priority (baseline)
- run_hybrid_priority_scheduler_optimized : hybrid priority (Algorithm 4)
"""

from __future__ import annotations

import heapq

import numpy as np
from numba import njit


# ----------------------------------------------------------------------
# Numba-JIT kernel: single coordinate update
# ----------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _update_coord(i, x, indptr, indices, data, b, lo, hi):
    s = 0.0
    diag = 0.0
    for k in range(indptr[i], indptr[i + 1]):
        j = indices[k]
        if j == i:
            diag = data[k]
        else:
            s += data[k] * x[j]
    if diag <= 0.0:
        return x[i]
    new_val = (b[i] - s) / diag
    if new_val < lo[i]:
        new_val = lo[i]
    elif new_val > hi[i]:
        new_val = hi[i]
    return new_val


@njit(cache=True, fastmath=True)
def _residual_i(i, x, indptr, indices, data, b):
    s = 0.0
    for k in range(indptr[i], indptr[i + 1]):
        s += data[k] * x[indices[k]]
    return s - b[i]


# ----------------------------------------------------------------------
# Cyclic Gauss-Seidel
# ----------------------------------------------------------------------

def run_cyclic_gs_scheduler(
    H,
    b,
    x0,
    lo=None,
    hi=None,
    tol: float = 1e-6,
    max_sweeps: int = 100_000,
):
    """
    Projected cyclic Gauss-Seidel.

    Returns (x, updates, residual_inf).
    """
    n = H.shape[0]
    H_csr = H.tocsr()
    indptr = H_csr.indptr.astype(np.int64)
    indices = H_csr.indices.astype(np.int64)
    data = H_csr.data.astype(np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)
    lo = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=np.float64)
    hi = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=np.float64)

    updates = 0
    for sweep in range(max_sweeps):
        for i in range(n):
            x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
        updates += n
        res = _residual_inf(x, indptr, indices, data, b)
        if res < tol:
            return x, updates, res
    return x, updates, res


@njit(cache=True, fastmath=True)
def _residual_inf(x, indptr, indices, data, b):
    n = x.shape[0]
    m = 0.0
    for i in range(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        r = abs(s - b[i])
        if r > m:
            m = r
    return m


# ----------------------------------------------------------------------
# Random priority (baseline)
# ----------------------------------------------------------------------

def run_random_priority_scheduler(
    H,
    b,
    x0,
    lo=None,
    hi=None,
    tol: float = 1e-6,
    max_updates: int = 1_000_000,
    seed: int = 0,
):
    """
    Random priority queue: pop a uniformly random node each iteration.
    """
    n = H.shape[0]
    H_csr = H.tocsr()
    indptr = H_csr.indptr.astype(np.int64)
    indices = H_csr.indices.astype(np.int64)
    data = H_csr.data.astype(np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)
    lo = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=np.float64)
    hi = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=np.float64)

    rng = np.random.default_rng(seed)
    updates = 0
    for _ in range(max_updates):
        i = int(rng.integers(0, n))
        x[i] = _update_coord(i, x, indptr, indices, data, b, lo, hi)
        updates += 1
        if updates % n == 0:
            res = _residual_inf(x, indptr, indices, data, b)
            if res < tol:
                return x, updates, res
    res = _residual_inf(x, indptr, indices, data, b)
    return x, updates, res


# ----------------------------------------------------------------------
# Hybrid priority (Algorithm 4)
# ----------------------------------------------------------------------

def run_hybrid_priority_scheduler_optimized(
    indptr_csr,
    indices_csr,
    data_csr,
    indptr_csc,
    indices_csc,
    data_csc,
    b,
    x0,
    M: int,
    epsilon: float = 1e-3,
    tol: float = 1e-6,
    max_sweeps: int = 100_000,
    lo=None,
    hi=None,
):
    """
    Hybrid Priority Scheduler (Algorithm 4).

    Alternates a mandatory cyclic backbone sweep every M iterations with
    priority-queue updates driven by aged residual pi_i = |r_i| + epsilon*tau_i.

    Parameters
    ----------
    indptr_csr, indices_csr, data_csr : CSR arrays of H_ext
    indptr_csc, indices_csc, data_csc : CSC arrays of H_ext
        (used for O(deg) residual recomputation on neighbors)
    b : np.ndarray (n,)
    x0 : np.ndarray (n,)
    M : int
        Cyclic backbone period (Theorem 9.1 uses M = n).
    epsilon : float
        Aging rate (eq. 9.1).
    tol : float
        Residual tolerance.
    max_sweeps : int
        Maximum number of iterations.

    Returns
    -------
    x : np.ndarray
    updates : int
    residual_inf : float
    """
    n = b.shape[0]
    x = np.asarray(x0, dtype=np.float64).copy()
    b = np.asarray(b, dtype=np.float64)

    lo_arr = np.full(n, -np.inf) if lo is None else np.asarray(lo, dtype=np.float64)
    hi_arr = np.full(n, np.inf) if hi is None else np.asarray(hi, dtype=np.float64)

    # Residuals
    r = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r[i] = _residual_i(i, x, indptr_csr, indices_csr, data_csr, b)

    tau = np.zeros(n, dtype=np.int64)

    # Initialize priority queue
    pq = [(-abs(r[i]), i) for i in range(n)]
    heapq.heapify(pq)

    updates = 0
    k = 0

    while updates < max_sweeps * n:
        if k % M == 0:
            # Cyclic backbone sweep
            for i in range(n):
                x[i] = _update_coord(
                    i, x, indptr_csr, indices_csr, data_csr, b, lo_arr, hi_arr
                )
                tau[i] = 0
            # Recompute residuals
            for i in range(n):
                r[i] = _residual_i(i, x, indptr_csr, indices_csr, data_csr, b)
            updates += n
        else:
            if not pq:
                pq = [(-(abs(r[i]) + epsilon * tau[i]), i) for i in range(n)]
                heapq.heapify(pq)
            _, i = heapq.heappop(pq)
            x[i] = _update_coord(
                i, x, indptr_csr, indices_csr, data_csr, b, lo_arr, hi_arr
            )
            tau[i] = 0
            updates += 1

            # Recompute residual at i
            r[i] = _residual_i(i, x, indptr_csr, indices_csr, data_csr, b)
            heapq.heappush(pq, (-(abs(r[i]) + epsilon * tau[i]), i))

            # Push CSC neighbors (O(deg) residual recomputation)
            for k_idx in range(indptr_csc[i], indptr_csc[i + 1]):
                j = indices_csc[k_idx]
                if j != i:
                    r[j] = _residual_i(j, x, indptr_csr, indices_csr, data_csr, b)
                    tau[j] += 1
                    heapq.heappush(
                        pq, (-(abs(r[j]) + epsilon * tau[j]), j)
                    )

        k += 1

        if updates % n == 0:
            res_inf = float(np.max(np.abs(r)))
            if res_inf < tol:
                return x, updates, res_inf

    res_inf = float(np.max(np.abs(r)))
    return x, updates, res_inf


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx
    from .core_graph import EpistemicGraph
    from .hessian_builder import assemble_extended_hessian
    from .cycle_basis import build_cycle_matrix_fundamental

    n = 200
    G_nx = nx.barabasi_albert_graph(n, 3, seed=0)
    g = EpistemicGraph(num_nodes=n)
    g.b = np.random.uniform(0, 1, size=n)
    for u, v in G_nx.edges():
        g.add_support_edge(u, v, 1.0)

    Q_cyc = build_cycle_matrix_fundamental(g)
    H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc)
    H_csr = H.tocsr()
    H_csc = H.tocsc()

    x0 = np.full(n, 0.5)

    print("=== Cyclic GS ===")
    x, updates, res = run_cyclic_gs_scheduler(H, g.b, x0, tol=1e-6)
    print(f"  updates={updates}, res={res:.2e}")

    print("=== Hybrid priority ===")
    x, updates, res = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        g.b, x0,
        M=n, epsilon=1e-3, tol=1e-6,
    )
    print(f"  updates={updates}, res={res:.2e}")

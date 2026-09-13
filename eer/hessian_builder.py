"""
Extended Hessian assembly for the Epistemic Ecology Runtime.

Assembles the extended precision matrix

    H_ext = H_0 + alpha * Q_cascade + gamma * Q_cycle

where

    H_0       = Lambda + L_S + L_D               (eq. 2.3, 9)
    Q_cascade = sum_p w(|p|) P_p^T P_p           (eq. 5.4, 15)
    Q_cycle   = sum_sigma |sigma|^{-1} B_sigma B_sigma^T   (eq. 5.6, 17)

All matrices are symmetric positive semi-definite; Lambda is strictly
positive definite; the sum is symmetric positive definite (Corollary 5.1).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, diags

from .core_graph import EpistemicGraph, EDGE_DERIVED_FROM
from .cycle_basis import build_cycle_matrix_fundamental


# ======================================================================
# Q_cascade — depth-bounded DFS over the derived-from subgraph
# ======================================================================

def build_cascade_matrix_bounded(
    G: EpistemicGraph,
    L_max: int = 4,
    max_paths_per_node: int = 500,
    decay: float = 1.0,
) -> csr_matrix:
    """
    Build the cascade precision matrix Q_cascade.

    For every simple directed path p = (v_0, ..., v_l) in the derived-from
    subgraph with l <= L_max, we add the rank-one term

        w(l) * P_p^T P_p

    where

        P_p x = sum_{k=0}^{l} (-1)^k x_{v_k}
        w(l)  = exp(-decay * l)

    The number of paths is capped at `max_paths_per_node` per source node
    to bound runtime on dense derivation subgraphs.

    Parameters
    ----------
    G : EpistemicGraph
        Graph containing derived-from edges (type EDGE_DERIVED_FROM).
    L_max : int
        Maximum path length. Runtime is exponential in L_max; use 3-5.
    max_paths_per_node : int
        Per-source cap on emitted paths.
    decay : float
        Path-length decay rate in w(l) = exp(-decay * l).

    Returns
    -------
    Q_cascade : csr_matrix, shape (n, n), symmetric PSD
    """
    n = G.num_nodes
    src, dst, w = G.edges_by_type(etype=EDGE_DERIVED_FROM)

    # Build adjacency (out-neighbours with weights)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for u, v, ww in zip(src, dst, w):
        adj[int(u)].append((int(v), float(ww)))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for start in range(n):
        # DFS stack: (current node, path so far, accumulated weight, depth)
        stack: list[tuple[int, list[int], float, int]] = [
            (start, [start], 1.0, 0)
        ]
        count = 0
        while stack and count < max_paths_per_node:
            node, path, weight, depth = stack.pop()

            if depth > 0:
                # Emit the rank-one term w(l) * P_p^T P_p for this path
                L = len(path)
                decay_w = float(np.exp(-decay * L))
                coeff = weight * decay_w
                signs = [1.0 if k % 2 == 0 else -1.0 for k in range(L)]
                for i in range(L):
                    ri = path[i]
                    si = signs[i]
                    for j in range(L):
                        rows.append(ri)
                        cols.append(path[j])
                        vals.append(coeff * si * signs[j])
                count += 1

            if depth < L_max:
                for nxt, ww in adj.get(node, []):
                    if nxt not in path:
                        stack.append(
                            (nxt, path + [nxt], weight * ww, depth + 1)
                        )

    if not vals:
        return csr_matrix((n, n), dtype=np.float64)

    Q = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    Q = 0.5 * (Q + Q.T)
    Q.eliminate_zeros()
    return Q.tocsr()


# ======================================================================
# H_0 = Lambda + L_S + L_D
# ======================================================================

def build_base_hessian(G: EpistemicGraph) -> csr_matrix:
    """
    Build H_0 = Lambda + L_S + L_D (the unregularized base Hessian).

    Returns
    -------
    H_0 : csr_matrix, shape (n, n), symmetric positive definite
    """
    Lambda = diags(G.lambda_vec, 0, format="csr")
    L_S = G.support_laplacian()
    L_D = G.derived_from_laplacian()
    H0 = (Lambda + L_S + L_D).tocsr()
    H0 = 0.5 * (H0 + H0.T)
    H0.eliminate_zeros()
    return H0


# ======================================================================
# H_ext = H_0 + alpha * Q_cascade + gamma * Q_cycle
# ======================================================================

def assemble_extended_hessian(
    G: EpistemicGraph,
    alpha: float = 0.0,
    gamma: float = 0.0,
    Q_cascade: Optional[csr_matrix] = None,
    Q_cycle: Optional[csr_matrix] = None,
    L_max: int = 4,
    max_paths_per_node: int = 500,
) -> csr_matrix:
    """
    Assemble the extended Hessian

        H_ext = Lambda + L_S + L_D + alpha * Q_cascade + gamma * Q_cycle

    Parameters
    ----------
    G : EpistemicGraph
    alpha : float >= 0
        Cascade precision (eq. 5.4).
    gamma : float >= 0
        Cycle precision (eq. 5.6).
    Q_cascade : csr_matrix, optional
        Pre-computed cascade matrix. If None and alpha > 0, built on demand.
    Q_cycle : csr_matrix, optional
        Pre-computed cycle matrix. If None and gamma > 0, built on demand.
    L_max : int
        Passed to build_cascade_matrix_bounded when Q_cascade is None.
    max_paths_per_node : int
        Passed to build_cascade_matrix_bounded when Q_cascade is None.

    Returns
    -------
    H_ext : csr_matrix, shape (n, n), symmetric positive definite

    Raises
    ------
    ValueError
        If alpha < 0 or gamma < 0.

    Notes
    -----
    Corollary 5.1 guarantees H_ext ≻ 0 for all alpha, gamma >= 0, because
    H_0 ≻ 0 (Lambda ≻ 0) and Q_cascade, Q_cycle ⪰ 0.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    n = G.num_nodes

    # Base Hessian (always required)
    H = build_base_hessian(G)

    # Cascade term
    if alpha > 0.0:
        if Q_cascade is None:
            Q_cascade = build_cascade_matrix_bounded(
                G, L_max=L_max, max_paths_per_node=max_paths_per_node
            )
        if Q_cascade.shape != (n, n):
            raise ValueError(
                f"Q_cascade has shape {Q_cascade.shape}, expected ({n}, {n})"
            )
        H = H + alpha * Q_cascade

    # Cycle term
    if gamma > 0.0:
        if Q_cycle is None:
            Q_cycle = build_cycle_matrix_fundamental(G)
        if Q_cycle.shape != (n, n):
            raise ValueError(
                f"Q_cycle has shape {Q_cycle.shape}, expected ({n}, {n})"
            )
        H = H + gamma * Q_cycle

    # Enforce symmetry and drop explicit zeros
    H = 0.5 * (H + H.T)
    H.eliminate_zeros()
    return H.tocsr()


# ======================================================================
# Diagnostics
# ======================================================================

def hessian_diagnostics(H: csr_matrix) -> dict:
    """
    Return summary statistics for a sparse Hessian.

    Useful for logging and for verifying SPD-ness in experiments.

    Returns
    -------
    dict with keys: shape, nnz, density, min_diag, max_diag, symmetric
    """
    n = H.shape[0]
    diag = H.diagonal()
    diff = H - H.T
    sym = abs(diff).nnz == 0 or float(np.abs(diff.data).max()) < 1e-12

    return {
        "shape": tuple(H.shape),
        "nnz": int(H.nnz),
        "density": float(H.nnz) / float(n * n) if n > 0 else 0.0,
        "min_diag": float(diag.min()) if n > 0 else 0.0,
        "max_diag": float(diag.max()) if n > 0 else 0.0,
        "symmetric": bool(sym),
    }


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    import networkx as nx

    n = 100
    G_nx = nx.barabasi_albert_graph(n, m=3, seed=0)
    g = EpistemicGraph(num_nodes=n)
    g.b = np.random.uniform(0.0, 1.0, size=n)
    g.lambda_vec = np.ones(n, dtype=np.float64)

    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    # Add ~10% of edges as derived-from
    rng = np.random.default_rng(0)
    edges = list(G_nx.edges())
    n_derived = max(1, len(edges) // 10)
    idx = rng.choice(len(edges), size=n_derived, replace=False)
    for k in idx:
        uu, vv = edges[int(k)]
        g.add_derived_from_edge(int(uu), int(vv), 0.5)

    # Base Hessian
    H0 = build_base_hessian(g)
    print(f"H_0: {hessian_diagnostics(H0)}")

    # Q_cascade
    Q_c = build_cascade_matrix_bounded(g, L_max=3)
    print(f"Q_cascade: shape={Q_c.shape}, nnz={Q_c.nnz}")

    # Full extended Hessian
    H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
    print(f"H_ext: {hessian_diagnostics(H)}")

    # SPD check
    from scipy.sparse.linalg import eigsh
    eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    print(f"lambda_min(H_ext) = {eig_min:.6e}")
    assert eig_min > 0, "H_ext is not positive definite!"
    print("OK: H_ext is SPD")

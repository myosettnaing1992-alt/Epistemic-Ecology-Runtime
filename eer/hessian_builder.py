"""
Extended Hessian assembly.

Assembles H_ext = H_0 + alpha * Q_cascade + gamma * Q_cycle
with H_0 = Lambda + L_S + L_D (eq. 9 of the paper).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix, diags, identity

from .core_graph import EpistemicGraph
from .cycle_basis import build_cycle_matrix_fundamental


# ----------------------------------------------------------------------
# Cascade matrix (bounded DFS)
# ----------------------------------------------------------------------

def build_cascade_matrix_bounded(
    G: EpistemicGraph,
    L_max: int = 4,
    max_paths_per_node: int = 500,
    decay: float = 1.0,
) -> csr_matrix:
    """
    Build Q_cascade from all simple directed paths in the derived-from
    subgraph, with |path| <= L_max.

    Each path p = (v_0, ..., v_l) contributes the rank-one term
    w(|p|) * P_p^T P_p where P_p x = sum_k (-1)^k x_{v_k} and
    w(l) = exp(-decay * l). Truncated at max_paths_per_node per source.

    Complexity: exponential in L_max; use L_max <= 5 in practice.
    """
    src, dst, w = G.edges_by_type(etype=2)  # EDGE_DERIVED_FROM
    n = G.num_nodes

    # Build adjacency
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for u, v, ww in zip(src, dst, w):
        adj[int(u)].append((int(v), float(ww)))

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for start in range(n):
        stack = [(start, [start], 1.0, 0)]
        count = 0
        while stack and count < max_paths_per_node:
            node, path, weight, depth = stack.pop()
            if depth > 0:
                # Emit alternating-projection term
                L = len(path)
                decay_w = np.exp(-decay * L)
                coeff = weight * decay_w
                signs = np.array([1.0 if k % 2 == 0 else -1.0 for k in range(L)])
                for i in range(L):
                    for j in range(L):
                        rows.append(path[i])
                        cols.append(path[j])
                        vals.append(coeff * signs[i] * signs[j])
                count += 1

            if depth < L_max:
                for nxt, ww in adj.get(node, []):
                    if nxt not in path:
                        stack.append((nxt, path + [nxt], weight * ww, depth + 1))

    if not vals:
        return csr_matrix((n, n), dtype=np.float64)

    Q = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    Q = 0.5 * (Q + Q.T)
    return Q.tocsr()


# ----------------------------------------------------------------------
# Extended Hessian assembly
# ----------------------------------------------------------------------

def assemble_extended_hessian(
    G: EpistemicGraph,
    alpha: float = 0.0,
    gamma: float = 0.0,
    Q_cascade: Optional[csr_matrix] = None,
    Q_cycle: Optional[csr_matrix] = None,
    L_max: int = 4,
) -> csr_matrix:
    """
    Assemble H_ext = Lambda + L_S + L_D + alpha * Q_cascade + gamma * Q_cycle.

    If Q_cascade / Q_cycle are None, they are built on demand.

    Returns
    -------
    H_ext : csr_matrix (n x n), symmetric positive definite
    """
    n = G.num_nodes

    Lambda = diags(G.lambda_vec, 0, format="csr")
    L_S = G.support_laplacian()
    L_D = G.derived_from_laplacian()

    H = (Lambda + L_S + L_D).tocsr()

    if alpha > 0.0:
        if Q_cascade is None:
            Q_cascade = build_cascade_matrix_bounded(G, L_max=L_max)
        H = H + alpha * Q_cascade

    if gamma > 0.0:
        if Q_cycle is None:
            Q_cycle = build_cycle_matrix_fundamental(G)
        H = H + gamma * Q_cycle

    H = 0.5 * (H + H.T)
    return H.tocsr()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    G_nx = nx.barabasi_albert_graph(100, 3, seed=0)
    g = EpistemicGraph(num_nodes=100)
    g.b = np.random.uniform(0, 1, size=100)
    for u, v in G_nx.edges():
        g.add_support_edge(u, v, 1.0)
    for u, v in list(G_nx.edges())[:30]:
        g.add_derived_from_edge(u, v, 0.5)

    H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5)
    print(f"H_ext shape: {H.shape}, nnz = {H.nnz}")
    print(f"Diagonal (first 5): {H.diagonal()[:5]}")

"""Matrix builders for Extended Hessian and structural regularizers."""

from typing import Optional
import numpy as np
import scipy.sparse as sp
import networkx as nx


# ---------------------------------------------------------------------------
# Base Hessian H_0 = Lambda + L_S + L_C  (eq. 2)
# ---------------------------------------------------------------------------

def build_base_hessian(graph) -> sp.csr_matrix:
    """Construct H_0 = Lambda + L_S + L_C (eq. 2)."""
    n = graph.n
    Lambda = sp.diags(graph.lambda_vec, format="csr")

    W_S = graph.get_W_S()
    deg_S = np.asarray(W_S.sum(axis=1)).flatten()
    L_S = sp.diags(deg_S, format="csr") - W_S

    W_C = graph.get_W_C()
    deg_C = np.asarray(W_C.sum(axis=1)).flatten()
    L_C = sp.diags(deg_C, format="csr") + W_C  # signless Laplacian

    return (Lambda + L_S + L_C).tocsr()


# ---------------------------------------------------------------------------
# SCC consensus Hessian (eq. 8)
# ---------------------------------------------------------------------------

def build_scc_hessian_vectorized(
    graph, gamma_D: float = 1.0, lambda_C: float = 1.0
) -> sp.csr_matrix:
    """Vectorized block-COO assembly of H_SCC (eq. 8)."""
    n = graph.n
    _, labels = graph.get_scc_decomposition()

    rows_list, cols_list, vals_list = [], [], []
    sort_idx = np.argsort(labels, kind="stable")
    sorted_labels = labels[sort_idx]
    boundaries = np.flatnonzero(np.diff(sorted_labels, prepend=-1, append=-1))

    for k in range(len(boundaries) - 1):
        start, end = boundaries[k], boundaries[k + 1]
        nodes = sort_idx[start:end]
        size_C = len(nodes)
        if size_C <= 1:
            continue

        factor = (gamma_D ** 2) / (gamma_D * size_C + lambda_C)
        ii, jj = np.meshgrid(nodes, nodes, indexing="ij")
        block = np.full((size_C, size_C), -factor, dtype=np.float64)
        np.fill_diagonal(block, gamma_D - factor)

        rows_list.append(ii.ravel())
        cols_list.append(jj.ravel())
        vals_list.append(block.ravel())

    if not rows_list:
        return sp.csr_matrix((n, n), dtype=np.float64)
    return sp.coo_matrix(
        (
            np.concatenate(vals_list),
            (np.concatenate(rows_list), np.concatenate(cols_list)),
        ),
        shape=(n, n),
    ).tocsr()


# ---------------------------------------------------------------------------
# Cascade regularizer (eq. 15) — bounded DFS with backtracking
# ---------------------------------------------------------------------------

def build_cascade_matrix_bounded(
    graph,
    L_max: int = 4,
    decay_lambda: float = 0.5,
    max_paths_per_node: int = 1000,
) -> sp.csr_matrix:
    """Q_cascade = sum_p w(|p|) P_p^T P_p over simple directed paths |p| <= L_max."""
    n = graph.n
    W_D = graph.get_W_D()
    adj = [W_D.indices[W_D.indptr[i]:W_D.indptr[i + 1]] for i in range(n)]
    rows, cols, vals = [], [], []

    for start in range(n):
        path = [start]
        on_path = {start}
        paths_emitted = 0

        def dfs(curr: int) -> None:
            nonlocal paths_emitted
            if paths_emitted >= max_paths_per_node:
                return
            if len(path) > 1:
                length = len(path) - 1
                w_p = np.exp(-decay_lambda * length)
                coeffs = np.array(
                    [(-1.0) ** k for k in range(len(path))], dtype=np.float64
                )
                nodes = np.asarray(path, dtype=np.int32)
                outer = np.outer(coeffs, coeffs) * w_p
                ii, jj = np.meshgrid(nodes, nodes, indexing="ij")
                rows.append(ii.ravel())
                cols.append(jj.ravel())
                vals.append(outer.ravel())
                paths_emitted += 1
            if len(path) > L_max:
                return
            for nxt in adj[curr]:
                if nxt not in on_path:
                    path.append(nxt)
                    on_path.add(nxt)
                    dfs(nxt)
                    path.pop()
                    on_path.discard(nxt)

        dfs(start)

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)
    return sp.coo_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n, n),
    ).tocsr()


# ---------------------------------------------------------------------------
# Cycle regularizer (eq. 17) — alternating signed incidence
# ---------------------------------------------------------------------------

def build_cycle_matrix_fundamental(graph) -> sp.csr_matrix:
    """Q_cycle = sum_sigma |sigma|^{-1} b_sigma b_sigma^T with alternating signs.

    Uses b_sigma(v_k) = (-1)^k to prevent cancellation on closed loops.
    """
    n = graph.n
    W_S = graph.get_W_S()
    if W_S.nnz == 0:
        return sp.csr_matrix((n, n), dtype=np.float64)

    G = nx.from_scipy_sparse_array(W_S)
    cycle_basis = nx.cycle_basis(G)

    rows, cols, vals = [], [], []
    for cycle in cycle_basis:
        k = len(cycle)
        if k == 0:
            continue
        cycle_nodes = np.asarray(cycle, dtype=np.int32)
        b_vec = np.zeros(n, dtype=np.float64)
        signs = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
        b_vec[cycle_nodes] = signs
        outer = np.outer(b_vec[cycle_nodes], b_vec[cycle_nodes]) / float(k)
        ii, jj = np.meshgrid(cycle_nodes, cycle_nodes, indexing="ij")
        rows.append(ii.ravel())
        cols.append(jj.ravel())
        vals.append(outer.ravel())

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)
    return sp.coo_matrix(
        (
            np.concatenate(vals),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n, n),
    ).tocsr()


# ---------------------------------------------------------------------------
# Full Extended Hessian assembly (eq. 9)
# ---------------------------------------------------------------------------

def assemble_extended_hessian(
    graph,
    alpha: float = 0.0,
    gamma: float = 0.0,
    gamma_D: float = 1.0,
    lambda_C: float = 1.0,
    Q_cascade: Optional[sp.csr_matrix] = None,
    Q_cycle: Optional[sp.csr_matrix] = None,
) -> sp.csr_matrix:
    """H_ext = H_0 + H_SCC + alpha * Q_cascade + gamma * Q_cycle (eq. 9)."""
    H_ext = build_base_hessian(graph) + build_scc_hessian_vectorized(
        graph, gamma_D=gamma_D, lambda_C=lambda_C
    )
    if alpha > 0 and Q_cascade is not None and Q_cascade.nnz > 0:
        H_ext = H_ext + alpha * Q_cascade
    if gamma > 0 and Q_cycle is not None and Q_cycle.nnz > 0:
        H_ext = H_ext + gamma * Q_cycle
    return H_ext.tocsr()

"""
Utility helpers.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def residual_inf(H: csr_matrix, x: np.ndarray, b: np.ndarray) -> float:
    """Return ||H x - b||_inf."""
    return float(np.max(np.abs(H @ x - b)))


def condition_number(H: csr_matrix) -> float:
    """Return the 2-norm condition number of a symmetric positive definite H."""
    from scipy.sparse.linalg import eigsh
    n = H.shape[0]
    if n <= 2:
        w = np.linalg.eigvalsh(H.toarray())
        return float(w[-1] / w[0])
    lam_max = eigsh(H, k=1, which="LA", return_eigenvectors=False)[0]
    lam_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    return float(lam_max / lam_min)


def rate_bound(H: csr_matrix) -> float:
    """
    Corollary 4.1: rho <= 1 - 1/(2 kappa(H)).
    """
    return float(1.0 - 1.0 / (2.0 * condition_number(H)))


def is_spd(H: csr_matrix, tol: float = 1e-10) -> bool:
    """Check that H is symmetric positive definite."""
    if abs(H - H.T).nnz > 0:
        return False
    from scipy.sparse.linalg import eigsh
    lam_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
    return bool(lam_min > tol)


def make_ba_graph(num_nodes: int, m: int = 3, seed: int = 0):
    """Convenience wrapper returning a networkx BA graph."""
    import networkx as nx
    return nx.barabasi_albert_graph(num_nodes, m, seed=seed)


def make_er_graph(num_nodes: int, avg_deg: float = 5.0, seed: int = 0):
    """Convenience wrapper returning a networkx Erdos-Renyi graph."""
    import networkx as nx
    p = avg_deg / max(num_nodes - 1, 1)
    return nx.erdos_renyi_graph(num_nodes, p, seed=seed)


def set_random_priors(
    graph,
    seed: int = 0,
    b_low: float = 0.0,
    b_high: float = 1.0,
    lambda_low: float = 0.5,
    lambda_high: float = 2.0,
):
    """Populate b and lambda_vec with random values."""
    rng = np.random.default_rng(seed)
    graph.b = rng.uniform(b_low, b_high, size=graph.num_nodes)
    graph.lambda_vec = rng.uniform(
        lambda_low, lambda_high, size=graph.num_nodes
    )
    return graph


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from .core_graph import EpistemicGraph
    from .hessian_builder import assemble_extended_hessian

    g = EpistemicGraph(num_nodes=100)
    set_random_priors(g, seed=0)
    G_nx = make_ba_graph(100, m=3, seed=0)
    for u, v in G_nx.edges():
        g.add_support_edge(u, v, 1.0)

    H = assemble_extended_hessian(g)
    print(f"H_ext is SPD: {is_spd(H)}")
    print(f"Condition number: {condition_number(H):.2f}")
    print(f"Rate bound: {rate_bound(H):.6f}")

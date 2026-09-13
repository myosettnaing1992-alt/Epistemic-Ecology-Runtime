"""
Fundamental Cycle Basis for EER.

Implements Algorithm 3 of the EER Monograph (Section 6.2):
    - Kruskal spanning forest construction
    - LCA-based path queries
    - Restriction to K_max shortest fundamental cycles

Reference:
    Epistemic Ecology Runtime (EER), Section 6.2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class CycleBasisResult:
    """Result container for fundamental cycle basis construction."""

    cycles: list[list[int]]           # each cycle: list of node indices
    cycle_lengths: np.ndarray         # |σ| for each cycle
    incidence: csr_matrix             # B_σ ∈ R^{n × |C|} (signed)
    n_cycles_total: int               # m_S - n + c (before K_max truncation)
    n_cycles_kept: int                # after K_max truncation
    spanning_forest_edges: list[tuple[int, int]]


def build_fundamental_cycle_basis(
    G_S: nx.Graph,
    K_max: int = 1000,
    weight: Optional[str] = None,
) -> CycleBasisResult:
    """
    Construct a fundamental cycle basis of the undirected support graph G_S.

    Parameters
    ----------
    G_S : nx.Graph
        Undirected support graph (may be disconnected).
    K_max : int
        Maximum number of fundamental cycles to retain. Cycles are ranked
        by length; only the K_max shortest are returned. Set to None for
        the full basis (infeasible for large graphs).
    weight : str, optional
        Edge attribute to use for weighted spanning tree. If None, unweighted.

    Returns
    -------
    CycleBasisResult
        Contains cycles, lengths, signed incidence matrix B_σ, and metadata.

    Notes
    -----
    Complexity:
        - Spanning forest: O(m_S α(n)) via Kruskal
        - Path queries:    O((m_S - n + c) log n) via binary-lifting LCA
        - Total:           O(m_S α(n) + (m_S - n + c) log n)

    Corresponds to Algorithm 3 in the EER Monograph.
    """
    if not isinstance(G_S, nx.Graph) or G_S.is_directed():
        raise ValueError("G_S must be an undirected networkx.Graph.")

    n = G_S.number_of_nodes()
    m_S = G_S.number_of_edges()
    c = nx.number_connected_components(G_S)

    # Relabel nodes to 0..n-1 for internal consistency with H_ext ordering
    node_list = list(G_S.nodes())
    node_to_idx = {v: i for i, v in enumerate(node_list)}
    G_relabeled = nx.relabel_nodes(G_S, node_to_idx)

    # ------------------------------------------------------------------
    # Step 1: Kruskal spanning forest
    # ------------------------------------------------------------------
    # Use minimum spanning forest; if unweighted, all edges weight 1
    if weight is not None:
        T = nx.minimum_spanning_tree(G_relabeled, weight=weight)
    else:
        T = nx.minimum_spanning_tree(G_relabeled)

    tree_edges = set(frozenset(e) for e in T.edges())
    non_tree_edges = [
        e for e in G_relabeled.edges() if frozenset(e) not in tree_edges
    ]

    # ------------------------------------------------------------------
    # Step 2: Binary-lifting LCA on the spanning forest
    # ------------------------------------------------------------------
    lca = _BinaryLiftingLCA(T, n)

    # ------------------------------------------------------------------
    # Step 3: For each non-tree edge, form the unique fundamental cycle
    # ------------------------------------------------------------------
    cycles: list[list[int]] = []
    for u, v in non_tree_edges:
        path_uv = lca.path(u, v)
        # Fundamental cycle = path(u, v) ∪ {edge (u, v)}
        cycle = path_uv  # path already includes u and v endpoints
        cycles.append(cycle)

    n_cycles_total = len(cycles)

    # ------------------------------------------------------------------
    # Step 4: Sort by length and truncate to K_max
    # ------------------------------------------------------------------
    cycles.sort(key=len)
    if K_max is not None and len(cycles) > K_max:
        cycles = cycles[:K_max]

    n_cycles_kept = len(cycles)
    cycle_lengths = np.array([len(c) for c in cycles], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 5: Build signed incidence matrix B_σ ∈ R^{n × |C|}
    # ------------------------------------------------------------------
    incidence = _build_signed_incidence(cycles, n)

    return CycleBasisResult(
        cycles=cycles,
        cycle_lengths=cycle_lengths,
        incidence=incidence,
        n_cycles_total=n_cycles_total,
        n_cycles_kept=n_cycles_kept,
        spanning_forest_edges=list(T.edges()),
    )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

class _BinaryLiftingLCA:
    """Binary-lifting LCA structure on a forest."""

    def __init__(self, T: nx.Graph, n: int):
        self.n = n
        self.LOG = max(1, (n).bit_length())
        self.depth = np.full(n, -1, dtype=np.int64)
        self.parent = np.full((self.LOG, n), -1, dtype=np.int64)

        # BFS from every component root
        for root in (v for v in T.nodes() if self.depth[v] < 0):
            self._bfs_root(T, root)

    def _bfs_root(self, T: nx.Graph, root: int) -> None:
        self.depth[root] = 0
        self.parent[0, root] = root
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in T.neighbors(u):
                if self.depth[v] < 0:
                    self.depth[v] = self.depth[u] + 1
                    self.parent[0, v] = u
                    queue.append(v)

        # Fill higher ancestors
        for k in range(1, self.LOG):
            for v in range(self.n):
                if self.parent[k - 1, v] >= 0:
                    self.parent[k, v] = self.parent[k - 1, self.parent[k - 1, v]]

    def lca(self, u: int, v: int) -> int:
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        # Lift u to depth of v
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if (diff >> k) & 1:
                u = int(self.parent[k, u])
        if u == v:
            return u
        # Lift both
        for k in reversed(range(self.LOG)):
            if self.parent[k, u] != self.parent[k, v]:
                u = int(self.parent[k, u])
                v = int(self.parent[k, v])
        return int(self.parent[0, u])

    def path(self, u: int, v: int) -> list[int]:
        w = self.lca(u, v)
        left = []
        x = u
        while x != w:
            left.append(x)
            x = int(self.parent[0, x])
        left.append(w)
        right = []
        x = v
        while x != w:
            right.append(x)
            x = int(self.parent[0, x])
        # path from u to v: left (u → w) then reverse(right without w)
        return left + right[::-1][1:]


def _build_signed_incidence(
    cycles: list[list[int]], n: int
) -> csr_matrix:
    """
    Build the signed incidence matrix B_σ ∈ R^{n × |C|}.

    Convention: For cycle σ = (v_0, v_1, ..., v_ℓ = v_0), we orient each
    edge (v_i, v_{i+1}) from v_i to v_{i+1}. Then B_σ[v_i] = +1 and
    B_σ[v_{i+1}] = -1 for each consecutive pair, with contributions summed
    for repeated nodes.
    """
    n_cycles = len(cycles)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j, cycle in enumerate(cycles):
        L = len(cycle)
        for i in range(L):
            u = cycle[i]
            v = cycle[(i + 1) % L]
            rows.append(u); cols.append(j); data.append(+1.0)
            rows.append(v); cols.append(j); data.append(-1.0)

    B = csr_matrix(
        (data, (rows, cols)), shape=(n, n_cycles), dtype=np.float64
    )
    return B


# ----------------------------------------------------------------------
# Convenience: build K_cycle = Σ_σ |σ|^{-1} B_σ B_σᵀ
# ----------------------------------------------------------------------

def build_cycle_precision_matrix(
    basis: CycleBasisResult, gamma: float = 1.0
) -> csr_matrix:
    """
    Build the cycle precision matrix K_cycle as in Eq. (5.6):

        K_cycle = Σ_σ |σ|^{-1} B_σ B_σᵀ

    Parameters
    ----------
    basis : CycleBasisResult
    gamma : float
        Cycle precision. Multiplies K_cycle (paper's γ).

    Returns
    -------
    K_cycle : csr_matrix, shape (n, n), symmetric PSD
    """
    B = basis.incidence                                # (n, |C|)
    L = basis.cycle_lengths.astype(np.float64)         # (|C|,)
    inv_L = 1.0 / L

    # Weighted product: B diag(1/|σ|) B^T
    Bw = B.multiply(inv_L[np.newaxis, :])              # column scaling
    K = (Bw @ B.T).tocsr()
    K = gamma * K
    # Symmetrize to kill floating-point asymmetry
    K = 0.5 * (K + K.T)
    return K.tocsr()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Building random BA graph (n=2000)...")
    G = nx.barabasi_albert_graph(2000, 3, seed=0)
    t0 = time.time()
    basis = build_fundamental_cycle_basis(G, K_max=1000)
    t1 = time.time()

    print(f"  Total fundamental cycles (before K_max): {basis.n_cycles_total}")
    print(f"  Cycles kept:                              {basis.n_cycles_kept}")
    print(f"  Mean cycle length:                        {basis.cycle_lengths.mean():.2f}")
    print(f"  Build time:                               {t1 - t0:.3f} s")

    K = build_cycle_precision_matrix(basis, gamma=0.5)
    print(f"  K_cycle shape: {K.shape}, nnz = {K.nnz}")
    print(f"  K_cycle is symmetric: {(abs(K - K.T)).nnz == 0}")

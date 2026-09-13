"""
Fundamental cycle basis for the support graph.

Implements Algorithm 3 (Section 6.2): Kruskal spanning forest + LCA path
queries, with signed incidence matrix B_sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .core_graph import EpistemicGraph


# ----------------------------------------------------------------------
# Binary-lifting LCA
# ----------------------------------------------------------------------

class _BinaryLiftingLCA:
    def __init__(self, n: int, adj: dict[int, list[int]]):
        self.n = n
        self.LOG = max(1, n.bit_length())
        self.depth = np.full(n, -1, dtype=np.int64)
        self.parent = np.full((self.LOG, n), -1, dtype=np.int64)

        visited = np.zeros(n, dtype=bool)
        for root in range(n):
            if not visited[root]:
                self._bfs(root, adj, visited)

    def _bfs(self, root, adj, visited):
        self.depth[root] = 0
        self.parent[0, root] = root
        visited[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    self.depth[v] = self.depth[u] + 1
                    self.parent[0, v] = u
                    queue.append(v)

        for k in range(1, self.LOG):
            for v in range(self.n):
                p = self.parent[k - 1, v]
                if p >= 0:
                    self.parent[k, v] = self.parent[k - 1, p]

    def lca(self, u: int, v: int) -> int:
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        diff = self.depth[u] - self.depth[v]
        for k in range(self.LOG):
            if (diff >> k) & 1:
                u = int(self.parent[k, u])
        if u == v:
            return u
        for k in reversed(range(self.LOG)):
            if self.parent[k, u] != self.parent[k, v]:
                u = int(self.parent[k, u])
                v = int(self.parent[k, v])
        return int(self.parent[0, u])

    def path(self, u: int, v: int) -> list[int]:
        """
        Return the tree path from u to v (inclusive of both endpoints).

        Uses left + reversed(right) directly, since `right` already
        excludes the LCA.
        """
        w = self.lca(u, v)
        left: list[int] = []
        x = u
        while x != w:
            left.append(x)
            x = int(self.parent[0, x])
        left.append(w)

        right: list[int] = []
        x = v
        while x != w:
            right.append(x)
            x = int(self.parent[0, x])

        return left + right[::-1]


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class CycleBasisResult:
    cycles: list[list[int]]
    cycle_lengths: np.ndarray
    incidence: csr_matrix
    n_cycles_total: int
    n_cycles_kept: int


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------

def build_fundamental_cycle_basis(
    G: EpistemicGraph,
    K_max: int = 1000,
) -> CycleBasisResult:
    """
    Build a fundamental cycle basis of the undirected support graph.

    Uses Kruskal spanning forest + binary-lifting LCA. Returns at most
    K_max shortest cycles.
    """
    import networkx as nx

    n = G.num_nodes
    A = G.support_adjacency()
    A_sym = 0.5 * (A + A.T)
    G_nx = nx.from_scipy_sparse_array(A_sym.tocoo())

    # Kruskal minimum spanning forest
    T = nx.minimum_spanning_tree(G_nx)

    # Collect tree edges as ordered tuples of Python ints (both directions)
    tree_edges: set[tuple[int, int]] = set()
    for u, v in T.edges():
        iu, iv = int(u), int(v)
        tree_edges.add((iu, iv))
        tree_edges.add((iv, iu))

    # Identify non-tree edges
    non_tree_edges: list[tuple[int, int]] = []
    for u, v in G_nx.edges():
        iu, iv = int(u), int(v)
        if (iu, iv) not in tree_edges:
            non_tree_edges.append((iu, iv))

    # Build adjacency for LCA from the spanning tree
    parent_edges: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in T.edges():
        iu, iv = int(u), int(v)
        parent_edges[iu].append(iv)
        parent_edges[iv].append(iu)

    lca = _BinaryLiftingLCA(n, parent_edges)

    # For each non-tree edge, form the unique fundamental cycle
    cycles: list[list[int]] = []
    for u, v in non_tree_edges:
        path_uv = lca.path(u, v)
        if len(path_uv) >= 3:      # fundamental cycles always have >= 3 nodes
            cycles.append(path_uv)

    n_cycles_total = len(cycles)
    cycles.sort(key=len)
    if K_max is not None and len(cycles) > K_max:
        cycles = cycles[:K_max]
    n_cycles_kept = len(cycles)
    cycle_lengths = np.array([len(c) for c in cycles], dtype=np.int64)

    # Signed incidence matrix B_sigma: b_sigma(v_k) = (-1)^k
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for j, cycle in enumerate(cycles):
        for k, v in enumerate(cycle):
            rows.append(v)
            cols.append(j)
            data.append(1.0 if k % 2 == 0 else -1.0)

    if data:
        B = csr_matrix(
            (data, (rows, cols)),
            shape=(n, n_cycles_kept),
            dtype=np.float64,
        )
    else:
        B = csr_matrix((n, 0), dtype=np.float64)

    return CycleBasisResult(
        cycles=cycles,
        cycle_lengths=cycle_lengths,
        incidence=B,
        n_cycles_total=n_cycles_total,
        n_cycles_kept=n_cycles_kept,
    )


# ----------------------------------------------------------------------
# Cycle precision matrix
# ----------------------------------------------------------------------

def build_cycle_matrix_fundamental(
    G: EpistemicGraph,
    K_max: int = 1000,
) -> csr_matrix:
    """
    Build Q_cycle = sum_sigma |sigma|^{-1} B_sigma B_sigma^T.
    """
    basis = build_fundamental_cycle_basis(G, K_max=K_max)
    n = G.num_nodes
    B = basis.incidence
    L = basis.cycle_lengths.astype(np.float64)

    if L.size == 0 or B.shape[1] == 0:
        return csr_matrix((n, n), dtype=np.float64)

    inv_L = 1.0 / L
    Bw = B.multiply(inv_L[np.newaxis, :])
    Q = (Bw @ B.T).tocsr()
    Q = 0.5 * (Q + Q.T)
    Q.eliminate_zeros()
    return Q.tocsr()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    G_nx = nx.barabasi_albert_graph(200, 3, seed=0)
    g = EpistemicGraph(num_nodes=200)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    basis = build_fundamental_cycle_basis(g, K_max=500)
    print(f"Total fundamental cycles: {basis.n_cycles_total}")
    print(f"Cycles kept: {basis.n_cycles_kept}")
    print(f"Mean length: {basis.cycle_lengths.mean():.2f}")

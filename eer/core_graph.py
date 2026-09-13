"""
Core epistemic graph structure.

Decoupled edge-list buffering: edges are stored in flat typed arrays,
supporting fast appends and vectorized conversion to scipy sparse matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


# ----------------------------------------------------------------------
# Edge type constants
# ----------------------------------------------------------------------

EDGE_SUPPORT = 0
EDGE_CONTRADICTION = 1
EDGE_DERIVED_FROM = 2


# ----------------------------------------------------------------------
# EpistemicGraph
# ----------------------------------------------------------------------

@dataclass
class EpistemicGraph:
    """
    Directed epistemic multigraph with three disjoint edge types.

    Parameters
    ----------
    num_nodes : int
        Number of belief-holding agents.
    """

    num_nodes: int

    # Typed edge buffers (flat lists; converted to arrays on demand)
    _src: list[int] = field(default_factory=list)
    _dst: list[int] = field(default_factory=list)
    _weight: list[float] = field(default_factory=list)
    _etype: list[int] = field(default_factory=list)

    # Node attributes
    b: np.ndarray | None = None
    lambda_vec: np.ndarray | None = None
    x_min: np.ndarray | None = None
    x_max: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.num_nodes
        if self.b is None:
            self.b = np.zeros(n, dtype=np.float64)
        if self.lambda_vec is None:
            self.lambda_vec = np.ones(n, dtype=np.float64)
        if self.x_min is None:
            self.x_min = np.zeros(n, dtype=np.float64)
        if self.x_max is None:
            self.x_max = np.ones(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Edge addition
    # ------------------------------------------------------------------

    def add_support_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check_edge(u, v, w)
        self._src.append(u)
        self._dst.append(v)
        self._weight.append(w)
        self._etype.append(EDGE_SUPPORT)

    def add_contradiction_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check_edge(u, v, w)
        self._src.append(u)
        self._dst.append(v)
        self._weight.append(w)
        self._etype.append(EDGE_CONTRADICTION)

    def add_derived_from_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check_edge(u, v, w)
        self._src.append(u)
        self._dst.append(v)
        self._weight.append(w)
        self._etype.append(EDGE_DERIVED_FROM)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_edge(self, u: int, v: int, w: float) -> None:
        if not (0 <= u < self.num_nodes) or not (0 <= v < self.num_nodes):
            raise IndexError(f"Node index out of range: ({u}, {v})")
        if w < 0:
            raise ValueError(f"Edge weight must be non-negative: {w}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_edges(self) -> int:
        return len(self._src)

    @property
    def edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (src, dst, weight, etype) as numpy arrays."""
        return (
            np.asarray(self._src, dtype=np.int64),
            np.asarray(self._dst, dtype=np.int64),
            np.asarray(self._weight, dtype=np.float64),
            np.asarray(self._etype, dtype=np.int64),
        )

    def edges_by_type(self, etype: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (src, dst, weight) for a single edge type."""
        src, dst, w, t = self.edges
        mask = t == etype
        return src[mask], dst[mask], w[mask]

    # ------------------------------------------------------------------
    # Sparse matrix builders
    # ------------------------------------------------------------------

    def support_adjacency(self) -> csr_matrix:
        src, dst, w = self.edges_by_type(EDGE_SUPPORT)
        n = self.num_nodes
        return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()

    def contradiction_adjacency(self) -> csr_matrix:
        src, dst, w = self.edges_by_type(EDGE_CONTRADICTION)
        n = self.num_nodes
        return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()

    def derived_from_adjacency(self) -> csr_matrix:
        src, dst, w = self.edges_by_type(EDGE_DERIVED_FROM)
        n = self.num_nodes
        return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()

    def support_laplacian(self) -> csr_matrix:
        """Symmetric Laplacian L_S = B_S W_S B_S^T (undirected)."""
        A = self.support_adjacency()
        A_sym = 0.5 * (A + A.T)
        deg = np.asarray(A_sym.sum(axis=1)).ravel()
        L = csr_matrix(
            (
                np.concatenate([deg, -A_sym.data]),
                (
                    np.concatenate([np.arange(self.num_nodes), A_sym.indices]),
                    np.concatenate([np.arange(self.num_nodes), A_sym.indices]),
                ),
            ),
            shape=(self.num_nodes, self.num_nodes),
        )
        return L.tocsr()

    def derived_from_laplacian(self) -> csr_matrix:
        """Symmetric Laplacian L_D = B_D W_D B_D^T."""
        A = self.derived_from_adjacency()
        A_sym = 0.5 * (A + A.T)
        deg = np.asarray(A_sym.sum(axis=1)).ravel()
        L = csr_matrix(
            (
                np.concatenate([deg, -A_sym.data]),
                (
                    np.concatenate([np.arange(self.num_nodes), A_sym.indices]),
                    np.concatenate([np.arange(self.num_nodes), A_sym.indices]),
                ),
            ),
            shape=(self.num_nodes, self.num_nodes),
        )
        return L.tocsr()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    g = EpistemicGraph(num_nodes=5)
    g.add_support_edge(0, 1, 1.0)
    g.add_support_edge(1, 2, 0.5)
    g.add_derived_from_edge(2, 3, 0.8)
    g.add_contradiction_edge(3, 4, 0.3)

    print(f"Nodes: {g.num_nodes}")
    print(f"Edges: {g.num_edges}")
    print(f"Support adjacency:\n{g.support_adjacency().toarray()}")
    print(f"Support Laplacian:\n{g.support_laplacian().toarray()}")

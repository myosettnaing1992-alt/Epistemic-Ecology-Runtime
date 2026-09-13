"""
Core epistemic graph structure.

Decoupled edge-list buffering: edges are stored in flat typed arrays,
supporting fast appends and vectorized conversion to scipy sparse matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags


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

    _src: list[int] = field(default_factory=list)
    _dst: list[int] = field(default_factory=list)
    _weight: list[float] = field(default_factory=list)
    _etype: list[int] = field(default_factory=list)

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
        return (
            np.asarray(self._src, dtype=np.int64),
            np.asarray(self._dst, dtype=np.int64),
            np.asarray(self._weight, dtype=np.float64),
            np.asarray(self._etype, dtype=np.int64),
        )

    def edges_by_type(
        self, etype: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src, dst, w, t = self.edges
        mask = t == etype
        return src[mask], dst[mask], w[mask]

    # ------------------------------------------------------------------
    # Sparse adjacency
    # ------------------------------------------------------------------

    def _adjacency_for(self, etype: int) -> csr_matrix:
        src, dst, w = self.edges_by_type(etype)
        n = self.num_nodes
        if src.size == 0:
            return csr_matrix((n, n), dtype=np.float64)
        return coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()

    def support_adjacency(self) -> csr_matrix:
        return self._adjacency_for(EDGE_SUPPORT)

    def contradiction_adjacency(self) -> csr_matrix:
        return self._adjacency_for(EDGE_CONTRADICTION)

    def derived_from_adjacency(self) -> csr_matrix:
        return self._adjacency_for(EDGE_DERIVED_FROM)

    # ------------------------------------------------------------------
    # Symmetric Laplacians
    # ------------------------------------------------------------------
    # FIX: previous implementation mis-constructed off-diagonal entries.
    # We now compute L = D - A_sym explicitly using scipy.sparse.diags,
    # which is guaranteed to be positive semi-definite.
    # ------------------------------------------------------------------

    def _symmetric_laplacian(self, A: csr_matrix) -> csr_matrix:
        n = self.num_nodes
        if A.nnz == 0:
            return csr_matrix((n, n), dtype=np.float64)
        A_sym = (0.5 * (A + A.T)).tocsr()
        A_sym.eliminate_zeros()
        deg = np.asarray(A_sym.sum(axis=1)).ravel()
        D = diags(deg, 0, format="csr")
        L = (D - A_sym).tocsr()
        L.eliminate_zeros()
        return L

    def support_laplacian(self) -> csr_matrix:
        """Symmetric support Laplacian L_S = D - A_sym."""
        return self._symmetric_laplacian(self.support_adjacency())

    def derived_from_laplacian(self) -> csr_matrix:
        """Symmetric derived-from Laplacian L_D = D - A_sym."""
        return self._symmetric_laplacian(self.derived_from_adjacency())


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
    print(f"Support Laplacian:\n{g.support_laplacian().toarray()}")

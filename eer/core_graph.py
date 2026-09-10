"""Graph representation with decoupled edge-list buffering."""

from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


class EpistemicGraph:
    """Memory-efficient epistemic graph G = (V, E_S, E_C, E_D)."""

    def __init__(self, num_nodes: int):
        if num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {num_nodes}")
        self.n = num_nodes
        self.b = np.full(num_nodes, 0.5, dtype=np.float64)
        self.lambda_vec = np.ones(num_nodes, dtype=np.float64)

        self._S_rows, self._S_cols, self._S_vals = [], [], []
        self._C_rows, self._C_cols, self._C_vals = [], [], []
        self._D_rows, self._D_cols, self._D_vals = [], [], []

        self._cache: dict = {}

    # -- validation ----------------------------------------------------------
    def _check(self, u: int, v: int, w: float) -> None:
        if not (0 <= u < self.n):
            raise IndexError(f"node index u={u} out of range [0, {self.n})")
        if not (0 <= v < self.n):
            raise IndexError(f"node index v={v} out of range [0, {self.n})")
        if w < 0:
            raise ValueError(f"edge weight must be non-negative, got {w}")

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    # -- edge adders ---------------------------------------------------------
    def add_support_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check(u, v, w)
        self._invalidate_cache()
        self._S_rows.extend([u, v])
        self._S_cols.extend([v, u])
        self._S_vals.extend([w, w])

    def add_contradiction_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check(u, v, w)
        self._invalidate_cache()
        self._C_rows.extend([u, v])
        self._C_cols.extend([v, u])
        self._C_vals.extend([w, w])

    def add_derivation_edge(self, u: int, v: int, w: float = 1.0) -> None:
        self._check(u, v, w)
        self._invalidate_cache()
        self._D_rows.append(u)
        self._D_cols.append(v)
        self._D_vals.append(w)

    # -- sparse builders -----------------------------------------------------
    def _build_csr(self, rows, cols, vals) -> sp.csr_matrix:
        if not rows:
            return sp.csr_matrix((self.n, self.n), dtype=np.float64)
        return sp.coo_matrix(
            (
                np.asarray(vals, dtype=np.float64),
                (
                    np.asarray(rows, dtype=np.int32),
                    np.asarray(cols, dtype=np.int32),
                ),
            ),
            shape=(self.n, self.n),
        ).tocsr()

    def get_W_S(self) -> sp.csr_matrix:
        if "W_S" not in self._cache:
            self._cache["W_S"] = self._build_csr(self._S_rows, self._S_cols, self._S_vals)
        return self._cache["W_S"]

    def get_W_C(self) -> sp.csr_matrix:
        if "W_C" not in self._cache:
            self._cache["W_C"] = self._build_csr(self._C_rows, self._C_cols, self._C_vals)
        return self._cache["W_C"]

    def get_W_D(self) -> sp.csr_matrix:
        if "W_D" not in self._cache:
            self._cache["W_D"] = self._build_csr(self._D_rows, self._D_cols, self._D_vals)
        return self._cache["W_D"]

    # -- SCC decomposition ---------------------------------------------------
    def get_scc_decomposition(self) -> Tuple[int, np.ndarray]:
        W_D = self.get_W_D()
        num_sccs, labels = connected_components(
            W_D, directed=True, connection="strong"
        )
        return num_sccs, labels.astype(np.int32)

    # -- diagnostics ---------------------------------------------------------
    def num_edges(self) -> dict:
        return {
            "support": len(self._S_rows) // 2,
            "contradiction": len(self._C_rows) // 2,
            "derivation": len(self._D_rows),
        }

    def __repr__(self) -> str:
        e = self.num_edges()
        return (
            f"EpistemicGraph(n={self.n}, "
            f"|E_S|={e['support']}, |E_C|={e['contradiction']}, "
            f"|E_D|={e['derivation']})"
          )

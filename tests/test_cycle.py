"""
Unit tests for eer.cycle_basis.

Covers:
    - Fundamental cycle basis cardinality (m_S - n + c)
    - Signed incidence matrix structure (alternating signs)
    - Cycle validity (no repeated nodes, length >= 3)
    - Q_cycle positive semi-definiteness
    - Q_cycle symmetry
    - K_max truncation
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from eer import EpistemicGraph
from eer.cycle_basis import (
    build_fundamental_cycle_basis,
    build_cycle_matrix_fundamental,
)
from eer.utils import make_ba_graph


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _make_support_graph(n: int, m: int = 3, seed: int = 0) -> EpistemicGraph:
    """Build a BA-based EpistemicGraph with support edges only."""
    G_nx = make_ba_graph(n, m=m, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


# ----------------------------------------------------------------------
# Fundamental cycle basis construction
# ----------------------------------------------------------------------

class TestFundamentalCycleBasis:

    def test_tree_has_no_cycles(self):
        """A tree has m_S - n + 1 = 0 fundamental cycles."""
        import networkx as nx

        G_nx = nx.path_graph(10)
        g = EpistemicGraph(num_nodes=10)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 0
        assert basis.n_cycles_kept == 0
        assert basis.incidence.shape == (10, 0)

    def test_cycle_count_matches_formula(self):
        """
        For connected G, |C_T| = m_S - n + 1.
        """
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        m_S = g.num_edges
        n = g.num_nodes
        c = 1  # connected

        basis = build_fundamental_cycle_basis(g)
        expected = m_S - n + c
        assert basis.n_cycles_total == expected, (
            f"Expected {expected}, got {basis.n_cycles_total}"
        )

    def test_k_max_truncation(self):
        """K_max limits the number of returned cycles."""
        g = _make_support_graph(100, m=3, seed=0)

        basis_full = build_fundamental_cycle_basis(g, K_max=None)
        basis_trunc = build_fundamental_cycle_basis(g, K_max=10)

        assert basis_trunc.n_cycles_kept == 10
        assert basis_trunc.n_cycles_total == basis_full.n_cycles_total
        assert basis_trunc.incidence.shape[1] == 10

    def test_cycles_are_valid_paths(self):
        """
        Each fundamental cycle must:
          - contain no repeated nodes (it is a simple path in the tree)
          - have length at least 3 (since it includes both endpoints and
            at least one intermediate node for a non-tree edge)
        """
        g = _make_support_graph(30, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g)

        assert basis.n_cycles_total > 0, "Expected at least one cycle"

        for idx, cycle in enumerate(basis.cycles[:10]):
            # No repeated nodes
            assert len(set(cycle)) == len(cycle), (
                f"Cycle {idx} has repeated nodes: {cycle}"
            )
            # Minimum length 3
            assert len(cycle) >= 3, (
                f"Cycle {idx} has length {len(cycle)} < 3: {cycle}"
            )

    def test_all_cycles_are_simple_paths(self):
        """Every returned cycle must be a simple path (no repeats)."""
        g = _make_support_graph(50, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=100)

        for idx, cycle in enumerate(basis.cycles):
            assert len(set(cycle)) == len(cycle), (
                f"Cycle {idx} has repeats: {cycle}"
            )

    def test_cycles_sorted_by_length(self):
        """K_max truncation should keep the shortest cycles."""
        g = _make_support_graph(100, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=20)

        lengths = basis.cycle_lengths
        assert np.all(np.diff(lengths) >= 0), (
            "Cycles are not sorted by increasing length"
        )

    def test_cycle_lengths_match_cycles(self):
        """cycle_lengths[k] must equal len(cycles[k])."""
        g = _make_support_graph(50, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=50)

        for k, cycle in enumerate(basis.cycles):
            assert basis.cycle_lengths[k] == len(cycle)


# ----------------------------------------------------------------------
# Signed incidence matrix
# ----------------------------------------------------------------------

class TestSignedIncidence:

    def test_incidence_shape(self):
        """B has shape (n, |C_kept|)."""
        g = _make_support_graph(50, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=20)
        assert basis.incidence.shape == (50, 20)

    def test_incidence_shape_empty(self):
        """Empty cycle basis yields B of shape (n, 0)."""
        import networkx as nx

        G_nx = nx.path_graph(10)
        g = EpistemicGraph(num_nodes=10)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g)
        assert basis.incidence.shape == (10, 0)
        assert basis.incidence.nnz == 0

    def test_alternating_signs(self):
        """b_sigma(v_k) = (-1)^k."""
        g = _make_support_graph(20, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=5)
        B = basis.incidence.toarray()

        for j, cycle in enumerate(basis.cycles):
            for k, v in enumerate(cycle):
                expected = 1.0 if k % 2 == 0 else -1.0
                assert B[v, j] == pytest.approx(expected), (
                    f"Cycle {j}, position {k}: expected {expected}, "
                    f"got {B[v, j]}"
                )

    def test_incidence_is_sparse(self):
        """B should be sparse (no dense allocation for large n)."""
        g = _make_support_graph(200, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=100)
        B = basis.incidence

        assert isinstance(B, csr_matrix)
        # nnz <= 2 * sum of cycle lengths (each entry has at most one value)
        total_len = int(basis.cycle_lengths.sum())
        assert B.nnz <= total_len


# ----------------------------------------------------------------------
# Q_cycle properties
# ----------------------------------------------------------------------

class TestQCycle:

    def test_q_cycle_shape(self):
        g = _make_support_graph(50, m=3, seed=0)
        Q = build_cycle_matrix_fundamental(g)
        assert Q.shape == (50, 50)

    def test_q_cycle_is_symmetric(self):
        g = _make_support_graph(50, m=3, seed=0)
        Q = build_cycle_matrix_fundamental(g)
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_q_cycle_is_psd(self):
        """Q_cycle = sum |sigma|^{-1} B_sigma B_sigma^T >= 0."""
        from scipy.sparse.linalg import eigsh

        g = _make_support_graph(50, m=3, seed=0)
        Q = build_cycle_matrix_fundamental(g)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cycle")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10, f"Q_cycle has negative eigenvalue {eig_min}"

    def test_q_cycle_rank_leq_basis_size(self):
        """rank(Q_cycle) <= number of cycles kept."""
        g = _make_support_graph(50, m=3, seed=0)
        basis = build_fundamental_cycle_basis(g, K_max=20)
        Q = build_cycle_matrix_fundamental(g, K_max=20)

        dense = Q.toarray()
        sv = np.linalg.svd(dense, compute_uv=False)
        rank = int(np.sum(sv > 1e-8))
        assert rank <= basis.n_cycles_kept

    def test_q_cycle_empty_for_tree(self):
        """A tree has no cycles → Q_cycle should be zero."""
        import networkx as nx

        G_nx = nx.path_graph(20)
        g = EpistemicGraph(num_nodes=20)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        Q = build_cycle_matrix_fundamental(g)
        assert Q.nnz == 0
        assert Q.shape == (20, 20)

    def test_q_cycle_is_symmetric_for_larger(self):
        g = _make_support_graph(100, m=3, seed=0)
        Q = build_cycle_matrix_fundamental(g, K_max=200)
        diff = (Q - Q.T)
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_q_cycle_psd_for_larger(self):
        from scipy.sparse.linalg import eigsh

        g = _make_support_graph(100, m=3, seed=0)
        Q = build_cycle_matrix_fundamental(g, K_max=200)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cycle")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-8


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------

class TestCycleIntegration:

    def test_cycle_basis_for_disconnected_graph(self):
        """Two disjoint triangles → 2 independent cycles."""
        g = EpistemicGraph(num_nodes=6)
        # Triangle 1: 0-1-2-0
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        # Triangle 2: 3-4-5-3
        g.add_support_edge(3, 4, 1.0)
        g.add_support_edge(4, 5, 1.0)
        g.add_support_edge(5, 3, 1.0)

        basis = build_fundamental_cycle_basis(g)
        # m_S - n + c = 6 - 6 + 2 = 2
        assert basis.n_cycles_total == 2

    def test_q_cycle_block_diagonal_for_disconnected(self):
        """Disconnected components → Q_cycle block-diagonal."""
        g = EpistemicGraph(num_nodes=6)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        g.add_support_edge(3, 4, 1.0)
        g.add_support_edge(4, 5, 1.0)
        g.add_support_edge(5, 3, 1.0)

        Q = build_cycle_matrix_fundamental(g)
        # Cross-blocks (between components) should be zero
        assert Q[0, 3] == 0.0
        assert Q[0, 4] == 0.0
        assert Q[1, 5] == 0.0
        # Diagonal blocks should have non-zero entries
        assert Q[0, 1] != 0.0
        assert Q[3, 4] != 0.0

    def test_single_triangle(self):
        """Minimal test: a single triangle has exactly 1 cycle."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)

        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 1
        assert basis.n_cycles_kept == 1
        assert basis.cycle_lengths[0] == 3
        # The cycle must contain all three nodes
        assert set(basis.cycles[0]) == {0, 1, 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

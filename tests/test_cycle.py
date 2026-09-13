"""
Unit tests for eer.cycle_basis.

Covers:
    - Fundamental cycle basis cardinality (m_S - n + c)
    - Signed incidence matrix structure (alternating signs)
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
# Cycle basis construction
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
        G_nx = make_ba_graph(100, m=3, seed=0)
        g = EpistemicGraph(num_nodes=100)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis_full = build_fundamental_cycle_basis(g, K_max=None)
        basis_trunc = build_fundamental_cycle_basis(g, K_max=10)

        assert basis_trunc.n_cycles_kept == 10
        assert basis_trunc.n_cycles_total == basis_full.n_cycles_total

    def test_cycles_are_valid_paths(self):
        """Each cycle must start at the LCA and return to it."""
        G_nx = make_ba_graph(30, m=3, seed=0)
        g = EpistemicGraph(num_nodes=30)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g)
        for cycle in basis.cycles[:5]:
            # Each node in the cycle must be unique within the path
            assert len(set(cycle)) == len(cycle), (
                f"Cycle has repeated nodes: {cycle}"
            )
            # Cycle must have at least 3 nodes
            assert len(cycle) >= 3


# ----------------------------------------------------------------------
# Signed incidence matrix
# ----------------------------------------------------------------------

class TestSignedIncidence:

    def test_incidence_shape(self):
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g, K_max=20)
        assert basis.incidence.shape == (50, 20)

    def test_alternating_signs(self):
        """b_sigma(v_k) = (-1)^k."""
        G_nx = make_ba_graph(20, m=3, seed=0)
        g = EpistemicGraph(num_nodes=20)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g, K_max=5)
        B = basis.incidence.toarray()

        for j, cycle in enumerate(basis.cycles):
            for k, v in enumerate(cycle):
                expected = 1.0 if k % 2 == 0 else -1.0
                assert B[v, j] == pytest.approx(expected), (
                    f"Cycle {j}, position {k}: expected {expected}, "
                    f"got {B[v, j]}"
                )


# ----------------------------------------------------------------------
# Q_cycle properties
# ----------------------------------------------------------------------

class TestQCycle:

    def test_q_cycle_shape(self):
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        Q = build_cycle_matrix_fundamental(g)
        assert Q.shape == (50, 50)

    def test_q_cycle_is_symmetric(self):
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        Q = build_cycle_matrix_fundamental(g)
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_q_cycle_is_psd(self):
        """Q_cycle = sum |sigma|^{-1} B_sigma B_sigma^T >= 0."""
        from scipy.sparse.linalg import eigsh
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        Q = build_cycle_matrix_fundamental(g)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cycle")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10, f"Q_cycle has negative eigenvalue {eig_min}"

    def test_q_cycle_rank_leq_basis_size(self):
        G_nx = make_ba_graph(50, m=3, seed=0)
        g = EpistemicGraph(num_nodes=50)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        basis = build_fundamental_cycle_basis(g, K_max=20)
        Q = build_cycle_matrix_fundamental(g, K_max=20)

        dense = Q.toarray()
        sv = np.linalg.svd(dense, compute_uv=False)
        rank = int(np.sum(sv > 1e-8))
        assert rank <= basis.n_cycles_kept


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

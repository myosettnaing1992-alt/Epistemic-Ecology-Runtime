"""
Unit tests for eer.core_graph.EpistemicGraph.

Covers:
    - Construction and default attributes
    - Edge addition and type separation
    - Error handling (invalid nodes, negative weights)
    - Sparse adjacency matrices
    - Symmetric Laplacians (paper Eq. 2.3: L = B W B^T, no 0.5 factor)
    - Empty graph behavior
    - Disconnected graphs
    - Integration with networkx
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from eer import EpistemicGraph
from eer.core_graph import (
    EDGE_SUPPORT,
    EDGE_CONTRADICTION,
    EDGE_DERIVED_FROM,
)
from eer.utils import make_ba_graph


# ======================================================================
# 1. Construction
# ======================================================================

class TestConstruction:

    def test_default_attributes(self):
        g = EpistemicGraph(num_nodes=10)
        assert g.num_nodes == 10
        assert g.num_edges == 0
        assert g.b is not None and g.b.shape == (10,)
        assert g.lambda_vec is not None and g.lambda_vec.shape == (10,)
        assert g.x_min is not None and g.x_min.shape == (10,)
        assert g.x_max is not None and g.x_max.shape == (10,)

    def test_default_values(self):
        """Defaults: b=0, lambda=1, x_min=0, x_max=1."""
        g = EpistemicGraph(num_nodes=5)
        np.testing.assert_array_equal(g.b, np.zeros(5))
        np.testing.assert_array_equal(g.lambda_vec, np.ones(5))
        np.testing.assert_array_equal(g.x_min, np.zeros(5))
        np.testing.assert_array_equal(g.x_max, np.ones(5))

    def test_custom_priors(self):
        b = np.arange(10, dtype=np.float64)
        lam = np.full(10, 2.0)
        x_lo = np.full(10, -1.0)
        x_hi = np.full(10, 3.0)
        g = EpistemicGraph(
            num_nodes=10, b=b, lambda_vec=lam, x_min=x_lo, x_max=x_hi,
        )
        np.testing.assert_array_equal(g.b, b)
        np.testing.assert_array_equal(g.lambda_vec, lam)
        np.testing.assert_array_equal(g.x_min, x_lo)
        np.testing.assert_array_equal(g.x_max, x_hi)

    def test_zero_node_graph(self):
        g = EpistemicGraph(num_nodes=0)
        assert g.num_nodes == 0
        assert g.num_edges == 0


# ======================================================================
# 2. Edge addition
# ======================================================================

class TestEdgeAddition:

    def test_add_single_support_edge(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.5)
        assert g.num_edges == 1
        src, dst, w, t = g.edges
        assert src[0] == 0 and dst[0] == 1
        assert w[0] == 1.5
        assert t[0] == EDGE_SUPPORT

    def test_add_contradiction_edge(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_contradiction_edge(1, 2, 2.0)
        assert g.num_edges == 1
        src, dst, w, t = g.edges
        assert src[0] == 1 and dst[0] == 2
        assert w[0] == 2.0
        assert t[0] == EDGE_CONTRADICTION

    def test_add_derived_from_edge(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_derived_from_edge(2, 3, 0.5)
        assert g.num_edges == 1
        src, dst, w, t = g.edges
        assert src[0] == 2 and dst[0] == 3
        assert w[0] == 0.5
        assert t[0] == EDGE_DERIVED_FROM

    def test_edge_types_are_separated(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.0)
        g.add_contradiction_edge(1, 2, 2.0)
        g.add_derived_from_edge(2, 3, 3.0)
        assert g.num_edges == 3

        src_s, dst_s, w_s = g.edges_by_type(EDGE_SUPPORT)
        assert len(src_s) == 1 and src_s[0] == 0

        src_c, dst_c, w_c = g.edges_by_type(EDGE_CONTRADICTION)
        assert len(src_c) == 1 and src_c[0] == 1

        src_d, dst_d, w_d = g.edges_by_type(EDGE_DERIVED_FROM)
        assert len(src_d) == 1 and src_d[0] == 2

    def test_default_weight_is_one(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1)
        _, _, w, _ = g.edges
        assert w[0] == 1.0

    def test_parallel_edges(self):
        """Multigraph: same (u, v) can appear multiple times."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(0, 1, 2.0)
        assert g.num_edges == 2

    def test_invalid_node_index_raises(self):
        g = EpistemicGraph(num_nodes=5)
        with pytest.raises(IndexError):
            g.add_support_edge(0, 5, 1.0)
        with pytest.raises(IndexError):
            g.add_support_edge(-1, 0, 1.0)
        with pytest.raises(IndexError):
            g.add_support_edge(5, 5, 1.0)

    def test_negative_weight_raises(self):
        g = EpistemicGraph(num_nodes=5)
        with pytest.raises(ValueError):
            g.add_support_edge(0, 1, -0.5)
        with pytest.raises(ValueError):
            g.add_contradiction_edge(0, 1, -1.0)
        with pytest.raises(ValueError):
            g.add_derived_from_edge(0, 1, -0.1)

    def test_zero_weight_allowed(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 0.0)
        assert g.num_edges == 1


# ======================================================================
# 3. Edges property
# ======================================================================

class TestEdgesProperty:

    def test_edges_returns_arrays(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.0)
        src, dst, w, t = g.edges
        assert isinstance(src, np.ndarray)
        assert isinstance(dst, np.ndarray)
        assert isinstance(w, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert src.dtype == np.int64
        assert dst.dtype == np.int64
        assert w.dtype == np.float64
        assert t.dtype == np.int64

    def test_edges_empty(self):
        g = EpistemicGraph(num_nodes=5)
        src, dst, w, t = g.edges
        assert len(src) == 0
        assert len(dst) == 0
        assert len(w) == 0
        assert len(t) == 0

    def test_edges_by_type_empty(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.0)
        src_d, _, _ = g.edges_by_type(EDGE_DERIVED_FROM)
        assert len(src_d) == 0


# ======================================================================
# 4. Sparse adjacency matrices
# ======================================================================

class TestSparseAdjacency:

    def test_support_adjacency_shape_and_values(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 2.0)
        g.add_support_edge(1, 2, 3.0)
        A = g.support_adjacency()
        assert A.shape == (4, 4)
        assert A[0, 1] == 2.0
        assert A[1, 2] == 3.0
        assert A[2, 1] == 0.0  # directed: no reverse edge

    def test_support_adjacency_empty(self):
        g = EpistemicGraph(num_nodes=3)
        A = g.support_adjacency()
        assert A.shape == (3, 3)
        assert A.nnz == 0

    def test_contradiction_adjacency_isolates_type(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_contradiction_edge(1, 2, 2.0)
        A_C = g.contradiction_adjacency()
        assert A_C[0, 1] == 0.0
        assert A_C[1, 2] == 2.0

    def test_derived_from_adjacency_isolates_type(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 0.5)
        A_D = g.derived_from_adjacency()
        assert A_D[0, 1] == 0.0
        assert A_D[1, 2] == 0.5


# ======================================================================
# 5. Symmetric Laplacians (paper Eq. 2.3)
# ======================================================================

class TestSymmetricLaplacian:

    def test_single_edge_laplacian(self):
        """Edge 0-1 with weight 1: L = [[1,-1],[-1,1]]."""
        g = EpistemicGraph(num_nodes=2)
        g.add_support_edge(0, 1, 1.0)
        L = g.support_laplacian().toarray()
        expected = np.array([[ 1.0, -1.0], [-1.0,  1.0]])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_triangle_laplacian(self):
        """Triangle with unit weights: L[0,0] = 2 (no 0.5 factor)."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)

        L = g.support_laplacian().toarray()
        expected = np.array([
            [ 2.0, -1.0, -1.0],
            [-1.0,  2.0, -1.0],
            [-1.0, -1.0,  2.0],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_weighted_triangle_laplacian(self):
        """Weights 2, 3, 4: deg[0] = 2 + 4 = 6."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 2.0)
        g.add_support_edge(1, 2, 3.0)
        g.add_support_edge(2, 0, 4.0)

        L = g.support_laplacian().toarray()
        expected = np.array([
            [ 6.0, -2.0, -4.0],
            [-2.0,  5.0, -3.0],
            [-4.0, -3.0,  7.0],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_support_laplacian_is_symmetric(self):
        g = make_ba_graph(50, m=3, seed=0)
        gE = EpistemicGraph(num_nodes=g.number_of_nodes())
        for u, v in g.edges():
            gE.add_support_edge(int(u), int(v), 1.0)
        L = gE.support_laplacian()
        diff = L - L.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_support_laplacian_row_sums_zero(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 3, 1.0)
        L = g.support_laplacian()
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_support_laplacian_empty_graph(self):
        """Empty graph: L = 0 (with eliminate_zeros)."""
        g = EpistemicGraph(num_nodes=3)
        L = g.support_laplacian()
        assert L.nnz == 0
        assert L.shape == (3, 3)
        np.testing.assert_allclose(L.toarray(), 0.0)

    def test_derived_from_laplacian(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_derived_from_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 2.0)
        L_D = g.derived_from_laplacian().toarray()
        # deg[0] = 1, deg[1] = 1 + 2 = 3, deg[2] = 2
        expected = np.array([
            [ 1.0, -1.0,  0.0],
            [-1.0,  3.0, -2.0],
            [ 0.0, -2.0,  2.0],
        ])
        np.testing.assert_allclose(L_D, expected, atol=1e-12)

    def test_laplacian_is_psd(self):
        """L = D - A_undir is always PSD."""
        from scipy.sparse.linalg import eigsh

        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 3, 1.0)
        g.add_support_edge(3, 0, 1.0)
        L = g.support_laplacian()
        eig_min = eigsh(L, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10

    def test_parallel_edges_sum_weights(self):
        """Parallel edges: Laplacian off-diagonal accumulates weights."""
        g = EpistemicGraph(num_nodes=2)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(0, 1, 2.0)
        L = g.support_laplacian().toarray()
        # A[0,1] = 1 + 2 = 3, A_undir[0,1] = 3, deg[0] = 3
        expected = np.array([[ 3.0, -3.0], [-3.0,  3.0]])
        np.testing.assert_allclose(L, expected, atol=1e-12)


# ======================================================================
# 6. Disconnected graphs
# ======================================================================

class TestDisconnectedGraphs:

    def test_two_components(self):
        """Two triangles: Laplacian is block-diagonal."""
        g = EpistemicGraph(num_nodes=6)
        # Component A: 0, 1, 2
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        # Component B: 3, 4, 5
        g.add_support_edge(3, 4, 1.0)
        g.add_support_edge(4, 5, 1.0)
        g.add_support_edge(5, 3, 1.0)

        L = g.support_laplacian().toarray()
        # Cross-blocks should be zero
        assert np.all(L[:3, 3:] == 0.0)
        assert np.all(L[3:, :3] == 0.0)
        # Diagonal blocks should be triangles
        np.testing.assert_allclose(L[:3, :3], np.array([
            [ 2.0, -1.0, -1.0],
            [-1.0,  2.0, -1.0],
            [-1.0, -1.0,  2.0],
        ]), atol=1e-12)

    def test_isolated_nodes(self):
        """Isolated nodes have zero row in Laplacian."""
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        # Nodes 2, 3 are isolated
        L = g.support_laplacian().toarray()
        np.testing.assert_allclose(L[2, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(L[3, :], 0.0, atol=1e-12)


# ======================================================================
# 7. Integration with networkx
# ======================================================================

class TestNetworkXIntegration:

    def test_ba_graph_construction(self):
        import networkx as nx

        G_nx = nx.barabasi_albert_graph(30, 3, seed=0)
        g = EpistemicGraph(num_nodes=30)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)
        assert g.num_edges == G_nx.number_of_edges()

    def test_er_graph_construction(self):
        import networkx as nx

        G_nx = nx.erdos_renyi_graph(30, 0.2, seed=0)
        g = EpistemicGraph(num_nodes=30)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)
        assert g.num_edges == G_nx.number_of_edges()

    def test_path_graph(self):
        import networkx as nx

        G_nx = nx.path_graph(10)
        g = EpistemicGraph(num_nodes=10)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)
        assert g.num_edges == 9

    def test_mixed_edge_types(self):
        import networkx as nx

        G_nx = nx.barabasi_albert_graph(20, 3, seed=0)
        g = EpistemicGraph(num_nodes=20)
        edges = list(G_nx.edges())
        for i, (u, v) in enumerate(edges):
            if i % 3 == 0:
                g.add_support_edge(int(u), int(v), 1.0)
            elif i % 3 == 1:
                g.add_contradiction_edge(int(u), int(v), 0.5)
            else:
                g.add_derived_from_edge(int(u), int(v), 0.7)

        n_s = len(g.edges_by_type(EDGE_SUPPORT)[0])
        n_c = len(g.edges_by_type(EDGE_CONTRADICTION)[0])
        n_d = len(g.edges_by_type(EDGE_DERIVED_FROM)[0])
        assert n_s + n_c + n_d == g.num_edges


# ======================================================================
# 8. Integration with EER pipeline
# ======================================================================

class TestEERPipelineIntegration:

    def test_laplacians_have_consistent_shapes(self):
        g = EpistemicGraph(num_nodes=10)
        g.add_support_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 0.5)

        L_S = g.support_laplacian()
        L_D = g.derived_from_laplacian()

        assert L_S.shape == (10, 10)
        assert L_D.shape == (10, 10)
        assert L_S.nnz > 0
        assert L_D.nnz > 0

    def test_graph_builds_correctly_for_hessian(self):
        """The graph should feed cleanly into Hessian assembly."""
        from eer import assemble_extended_hessian

        G_nx = make_ba_graph(30, m=3, seed=0)
        g = EpistemicGraph(num_nodes=30)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        H = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        assert H.shape == (30, 30)
        # Diagonal should be positive
        assert np.all(H.diagonal() > 0)


# ======================================================================
# 9. Edge cases
# ======================================================================

class TestEdgeCases:

    def test_single_node_with_self_loop_intent(self):
        """Single node, no edges: graph is well-defined."""
        g = EpistemicGraph(num_nodes=1)
        assert g.num_edges == 0
        assert g.support_laplacian().nnz == 0

    def test_very_large_graph_construction(self):
        """Construction of n=1000 should complete quickly."""
        g = EpistemicGraph(num_nodes=1000)
        for i in range(0, 999, 2):
            g.add_support_edge(i, i + 1, 1.0)
        assert g.num_edges == 500

    def test_isolated_node_has_zero_degree(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        L = g.support_laplacian()
        assert L[2, 2] == 0.0
        assert L[2, 0] == 0.0
        assert L[0, 2] == 0.0


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

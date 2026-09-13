"""
Unit tests for eer.core_graph.EpistemicGraph.

Covers:
    - Edge addition and type separation
    - Sparse matrix construction (adjacency, Laplacian)
    - Prior attributes and bounds
    - Error handling
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


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

class TestConstruction:

    def test_default_attributes(self):
        g = EpistemicGraph(num_nodes=10)
        assert g.num_nodes == 10
        assert g.num_edges == 0
        assert g.b is not None and g.b.shape == (10,)
        assert g.lambda_vec is not None and g.lambda_vec.shape == (10,)
        assert g.x_min is not None and g.x_min.shape == (10,)
        assert g.x_max is not None and g.x_max.shape == (10,)

    def test_custom_priors(self):
        b = np.arange(10, dtype=np.float64)
        lam = np.full(10, 2.0)
        g = EpistemicGraph(num_nodes=10, b=b, lambda_vec=lam)
        np.testing.assert_array_equal(g.b, b)
        np.testing.assert_array_equal(g.lambda_vec, lam)


# ----------------------------------------------------------------------
# Edge addition
# ----------------------------------------------------------------------

class TestEdgeAddition:

    def test_add_single_support_edge(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.5)
        assert g.num_edges == 1
        src, dst, w, t = g.edges
        assert src[0] == 0 and dst[0] == 1
        assert w[0] == 1.5
        assert t[0] == EDGE_SUPPORT

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

    def test_invalid_node_index_raises(self):
        g = EpistemicGraph(num_nodes=5)
        with pytest.raises(IndexError):
            g.add_support_edge(0, 5, 1.0)
        with pytest.raises(IndexError):
            g.add_support_edge(-1, 0, 1.0)

    def test_negative_weight_raises(self):
        g = EpistemicGraph(num_nodes=5)
        with pytest.raises(ValueError):
            g.add_support_edge(0, 1, -0.5)


# ----------------------------------------------------------------------
# Sparse matrix builders
# ----------------------------------------------------------------------

class TestSparseMatrices:

    def test_support_adjacency_shape_and_values(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 2.0)
        g.add_support_edge(1, 2, 3.0)
        A = g.support_adjacency()
        assert A.shape == (4, 4)
        assert A[0, 1] == 2.0
        assert A[1, 2] == 3.0
        assert A[2, 1] == 0.0  # directed

    def test_support_laplacian_is_symmetric(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 3, 1.0)
        L = g.support_laplacian()
        diff = L - L.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_support_laplacian_row_sums_zero(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        L = g.support_laplacian()
        row_sums = np.asarray(L.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_derived_from_laplacian_shape(self):
        g = EpistemicGraph(num_nodes=5)
        g.add_derived_from_edge(0, 1, 0.5)
        L_D = g.derived_from_laplacian()
        assert L_D.shape == (5, 5)

    def test_empty_graph_laplacian_is_zero(self):
        g = EpistemicGraph(num_nodes=3)
        L = g.support_laplacian()
        assert L.nnz == 0


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------

class TestGraphIntegration:

    def test_ba_graph_construction(self):
        import networkx as nx
        G_nx = nx.barabasi_albert_graph(30, 3, seed=0)
        g = EpistemicGraph(num_nodes=30)
        for u, v in G_nx.edges():
            g.add_support_edge(u, v, 1.0)
        assert g.num_edges == G_nx.number_of_edges()

    def test_graph_with_mixed_edge_types(self):
        g = EpistemicGraph(num_nodes=10)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_contradiction_edge(2, 3, 0.5)
        g.add_derived_from_edge(3, 4, 0.7)

        L_S = g.support_laplacian()
        L_D = g.derived_from_laplacian()

        assert L_S.shape == (10, 10)
        assert L_D.shape == (10, 10)
        assert L_S.nnz > 0
        assert L_D.nnz > 0


# ----------------------------------------------------------------------
# Edge property
# ----------------------------------------------------------------------

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
        assert w.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for EpistemicGraph data structure."""

import numpy as np
import pytest

from eer.core_graph import EpistemicGraph


def test_invalid_num_nodes():
    with pytest.raises(ValueError):
        EpistemicGraph(0)
    with pytest.raises(ValueError):
        EpistemicGraph(-1)


def test_support_edge_symmetry():
    g = EpistemicGraph(3)
    g.add_support_edge(0, 1, w=2.0)
    W = g.get_W_S()
    assert W[0, 1] == 2.0
    assert W[1, 0] == 2.0


def test_support_edge_accumulation():
    g = EpistemicGraph(3)
    g.add_support_edge(0, 1, w=0.5)
    g.add_support_edge(0, 1, w=0.5)
    W = g.get_W_S()
    assert W[0, 1] == 1.0


def test_derivation_edge_directed():
    g = EpistemicGraph(3)
    g.add_derivation_edge(0, 1, w=1.5)
    W_D = g.get_W_D()
    assert W_D[0, 1] == 1.5
    assert W_D[1, 0] == 0.0


def test_contradiction_edge_symmetry():
    g = EpistemicGraph(3)
    g.add_contradiction_edge(0, 2, w=0.7)
    W_C = g.get_W_C()
    assert W_C[0, 2] == 0.7
    assert W_C[2, 0] == 0.7


def test_out_of_range_index():
    g = EpistemicGraph(3)
    with pytest.raises(IndexError):
        g.add_support_edge(0, 5)
    with pytest.raises(IndexError):
        g.add_derivation_edge(-1, 0)


def test_negative_weight_rejected():
    g = EpistemicGraph(3)
    with pytest.raises(ValueError):
        g.add_support_edge(0, 1, w=-1.0)


def test_default_priors_and_lambda():
    g = EpistemicGraph(5)
    assert np.all(g.b == 0.5)
    assert np.all(g.lambda_vec == 1.0)


def test_scc_decomposition_simple():
    g = EpistemicGraph(3)
    g.add_derivation_edge(0, 1)
    g.add_derivation_edge(1, 0)
    g.add_derivation_edge(1, 2)
    num_sccs, labels = g.get_scc_decomposition()
    # {0,1} is one SCC, {2} is another
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_repr_contains_edge_counts():
    g = EpistemicGraph(4)
    g.add_support_edge(0, 1)
    g.add_derivation_edge(1, 2)
    s = repr(g)
    assert "n=4" in s
    assert "|E_S|=1" in s
    assert "|E_D|=1" in s

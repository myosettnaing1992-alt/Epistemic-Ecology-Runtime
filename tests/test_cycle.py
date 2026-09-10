"""Regression tests for fundamental cycle matrix construction.

Guards against the sign-cancellation bug where symmetric incidence
(+1/-1 per edge) yields b_sigma = 0 and silently zeroes Q_cycle.
"""

import numpy as np
import pytest

from eer.core_graph import EpistemicGraph
from eer.hessian_builder import build_cycle_matrix_fundamental


def test_cycle_matrix_nonzero_incidence(four_cycle):
    Q = build_cycle_matrix_fundamental(four_cycle)
    assert Q.nnz > 0, (
        "Q_cycle is empty for a closed cycle. "
        "Likely cause: symmetric incidence cancels b_sigma. "
        "Fix: use alternating signs b_sigma(v_k) = (-1)^k."
    )


def test_uniform_beliefs_zero_energy(four_cycle):
    Q = build_cycle_matrix_fundamental(four_cycle)
    x = np.full(4, 0.5)
    energy = float(x @ Q @ x)
    assert abs(energy) < 1e-12


def test_alternating_beliefs_positive_energy(four_cycle):
    Q = build_cycle_matrix_fundamental(four_cycle)
    x = np.array([1.0, 0.0, 1.0, 0.0])
    energy = float(x @ Q @ x)
    assert energy > 0.0


def test_cycle_matrix_symmetric(four_cycle):
    Q = build_cycle_matrix_fundamental(four_cycle)
    assert (Q - Q.T).nnz == 0


def test_cycle_matrix_psd(four_cycle):
    Q = build_cycle_matrix_fundamental(four_cycle).toarray()
    eigs = np.linalg.eigvalsh(Q)
    assert np.all(eigs >= -1e-10)


def test_tree_graph_returns_empty(connected_tree):
    Q = build_cycle_matrix_fundamental(connected_tree)
    assert Q.nnz == 0


def test_disconnected_acyclic_forest():
    eg = EpistemicGraph(6)
    for u, v in [(0, 1), (1, 2)]:
        eg.add_support_edge(u, v)
    for u, v in [(3, 4), (4, 5)]:
        eg.add_support_edge(u, v)
    Q = build_cycle_matrix_fundamental(eg)
    assert Q.nnz == 0


def test_triangle_with_tail(triangle_with_tail):
    Q = build_cycle_matrix_fundamental(triangle_with_tail)
    assert Q.nnz > 0

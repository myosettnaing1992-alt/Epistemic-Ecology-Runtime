"""Tests for Hessian assembly: SPD, symmetry, correctness."""

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from eer.core_graph import EpistemicGraph
from eer.hessian_builder import (
    assemble_extended_hessian,
    build_base_hessian,
    build_scc_hessian_vectorized,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
)


def test_base_hessian_spd(random_graph):
    H = build_base_hessian(random_graph)
    eigs = eigsh(H, k=1, which="SA", return_eigenvectors=False)
    assert eigs[0] > 0, f"min eig = {eigs[0]}"


def test_base_hessian_symmetric(random_graph):
    H = build_base_hessian(random_graph)
    assert (H - H.T).nnz == 0


def test_scc_hessian_matches_formula():
    g = EpistemicGraph(2)
    g.add_derivation_edge(0, 1)
    g.add_derivation_edge(1, 0)
    H = build_scc_hessian_vectorized(g, gamma_D=1.0, lambda_C=1.0)
    expected = np.array([[2 / 3, -1 / 3], [-1 / 3, 2 / 3]])
    np.testing.assert_allclose(H.toarray(), expected, atol=1e-12)


def test_scc_hessian_psd(random_graph):
    H = build_scc_hessian_vectorized(random_graph)
    eigs = np.linalg.eigvalsh(H.toarray())
    assert np.all(eigs >= -1e-10)


def test_cascade_matrix_psd(random_graph):
    Q = build_cascade_matrix_bounded(random_graph, L_max=3)
    if Q.nnz > 0:
        eigs = np.linalg.eigvalsh(Q.toarray())
        assert np.all(eigs >= -1e-10)


def test_extended_hessian_spd(random_graph):
    Q_cas = build_cascade_matrix_bounded(random_graph, L_max=3)
    Q_cyc = build_cycle_matrix_fundamental(random_graph)
    H = assemble_extended_hessian(
        random_graph, alpha=0.1, gamma=0.1, Q_cascade=Q_cas, Q_cycle=Q_cyc
    )
    eigs = eigsh(H, k=1, which="SA", return_eigenvectors=False)
    assert eigs[0] > 0


def test_extended_hessian_symmetric(random_graph):
    H = assemble_extended_hessian(random_graph, alpha=0.1, gamma=0.1)
    assert (H - H.T).nnz == 0


@pytest.mark.parametrize(
    "alpha,gamma", [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0), (0.0, 1.0)]
)
def test_spd_across_hyperparameters(random_graph, alpha, gamma):
    H = assemble_extended_hessian(random_graph, alpha=alpha, gamma=gamma)
    eigs = eigsh(H, k=1, which="SA", return_eigenvectors=False)
    assert eigs[0] > 0

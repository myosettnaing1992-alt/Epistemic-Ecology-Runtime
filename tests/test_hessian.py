"""
Unit tests for eer.hessian_builder.

Covers:
    - Corollary 5.1: H_ext is positive definite for all alpha, gamma >= 0
    - Q_cascade symmetry and PSD
    - Extended system assembly (eq. 5.10)
    - Regularizers improve conditioning
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, spsolve

from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
)
from eer.cycle_basis import build_cycle_matrix_fundamental
from eer.utils import make_ba_graph, set_random_priors


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_test_graph(n: int = 50, seed: int = 0) -> EpistemicGraph:
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    # Add some derived-from edges
    rng = np.random.default_rng(seed)
    edges = list(G_nx.edges())
    for u, v in rng.choice(len(edges), size=len(edges) // 10, replace=False):
        uu, vv = edges[u]
        g.add_derived_from_edge(int(uu), int(vv), 0.5)
    return g


# ----------------------------------------------------------------------
# Corollary 5.1
# ----------------------------------------------------------------------

class TestCorollary51WellPosedness:
    """H_ext = H_0 + alpha Q_cascade + gamma Q_cycle must be SPD."""

    @pytest.mark.parametrize("alpha,gamma", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 0.0),
        (0.0, 5.0),
        (5.0, 5.0),
    ])
    def test_hext_is_spd(self, alpha, gamma):
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
        H = assemble_extended_hessian(
            g, alpha=alpha, gamma=gamma, Q_cycle=Q_cyc, L_max=3,
        )

        # Symmetry
        diff = H - H.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-10

        # Positive definiteness
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, (
            f"alpha={alpha}, gamma={gamma}: lambda_min={eig_min:.6e}"
        )

    def test_random_alpha_gamma_sweep(self):
        """Random sweep over alpha, gamma in [0, 10]."""
        rng = np.random.default_rng(42)
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)

        for _ in range(10):
            alpha = float(rng.uniform(0, 10))
            gamma = float(rng.uniform(0, 10))
            H = assemble_extended_hessian(
                g, alpha=alpha, gamma=gamma, Q_cycle=Q_cyc, L_max=3,
            )
            eig_min = eigsh(H, k=1, which="SA",
                            return_eigenvectors=False)[0]
            assert eig_min > 0, (
                f"alpha={alpha:.3f}, gamma={gamma:.3f}: "
                f"lambda_min={eig_min:.6e}"
            )

    def test_regularizers_increase_lambda_min(self):
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)

        H_base = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        H_ext = assemble_extended_hessian(
            g, alpha=1.0, gamma=1.0, Q_cycle=Q_cyc, L_max=3,
        )

        lam_base = eigsh(H_base, k=1, which="SA",
                         return_eigenvectors=False)[0]
        lam_ext = eigsh(H_ext, k=1, which="SA",
                        return_eigenvectors=False)[0]

        assert lam_ext >= lam_base - 1e-10


# ----------------------------------------------------------------------
# Q_cascade
# ----------------------------------------------------------------------

class TestQCascade:

    def test_q_cascade_symmetric(self):
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_q_cascade_psd(self):
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cascade")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10

    def test_q_cascade_shape(self):
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        assert Q.shape == (30, 30)


# ----------------------------------------------------------------------
# Extended system (eq. 5.10)
# ----------------------------------------------------------------------

class TestExtendedSystem:

    @pytest.mark.parametrize("alpha,gamma", [(0.5, 0.5), (1.0, 2.0)])
    def test_equilibrium_satisfies_linear_system(self, alpha, gamma):
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
        H = assemble_extended_hessian(
            g, alpha=alpha, gamma=gamma, Q_cycle=Q_cyc, L_max=3,
        )

        # RHS
        rng = np.random.default_rng(1)
        b = rng.standard_normal(g.num_nodes)

        # Solve
        x_star = spsolve(H.tocsc(), b)

        # Check residual
        res = H @ x_star - b
        assert np.max(np.abs(res)) < 1e-8, (
            f"alpha={alpha}, gamma={gamma}: ||Hx - b||_inf = "
            f"{np.max(np.abs(res)):.2e}"
        )

    def test_equilibrium_minimizes_energy(self):
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
        H = assemble_extended_hessian(
            g, alpha=0.5, gamma=0.5, Q_cycle=Q_cyc, L_max=3,
        )
        rng = np.random.default_rng(2)
        b = rng.standard_normal(g.num_nodes)

        x_star = spsolve(H.tocsc(), b)
        E_star = 0.5 * float(x_star @ (H @ x_star)) - float(b @ x_star)

        for _ in range(5):
            delta = 0.01 * rng.standard_normal(g.num_nodes)
            x_pert = x_star + delta
            E_pert = 0.5 * float(x_pert @ (H @ x_pert)) - float(b @ x_pert)
            assert E_pert > E_star


# ----------------------------------------------------------------------
# Alpha = gamma = 0 recovers base
# ----------------------------------------------------------------------

class TestBaseCase:

    def test_zero_alpha_gamma_recovers_base(self):
        g = _make_test_graph(30, seed=0)
        H_base = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        H_ext = assemble_extended_hessian(
            g, alpha=0.0, gamma=0.0, L_max=3,
        )
        diff = H_ext - H_base
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

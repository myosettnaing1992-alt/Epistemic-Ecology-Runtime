"""
Unit tests for eer.hessian_builder.

Covers:
    - Corollary 5.1: H_ext is positive definite for all alpha, gamma >= 0
    - Q_cascade symmetry and PSD
    - Q_cascade shape
    - Extended system assembly (eq. 5.10)
    - Regularizers improve conditioning
    - Base case (alpha = gamma = 0) recovers H_0
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
# Helper — build a synthetic test graph
# ----------------------------------------------------------------------

def _make_test_graph(n: int = 50, seed: int = 0) -> EpistemicGraph:
    """
    Build a BA-based EpistemicGraph with support and derived-from edges.

    The derived-from subgraph is constructed by selecting ~10% of the
    support edges at random and adding them as directed derived-from edges.
    The selection uses integer indices into the edge list; the previous
    implementation mistakenly unpacked the raw integers themselves.
    """
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)

    # Support edges (undirected in spirit; stored as directed pairs)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    # Derived-from edges: ~10% of the support edges
    rng = np.random.default_rng(seed)
    edges = list(G_nx.edges())
    n_derived = max(1, len(edges) // 10)
    idx = rng.choice(len(edges), size=n_derived, replace=False)
    for k in idx:
        uu, vv = edges[int(k)]
        g.add_derived_from_edge(int(uu), int(vv), 0.5)

    return g


# ----------------------------------------------------------------------
# Corollary 5.1 — Well-posedness
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
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-10, (
            "H_ext is not symmetric"
        )

        # Positive definiteness
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, (
            f"alpha={alpha}, gamma={gamma}: lambda_min={eig_min:.6e} <= 0"
        )

    def test_random_alpha_gamma_sweep(self):
        """Random sweep over alpha, gamma in [0, 10] to confirm robustness."""
        rng = np.random.default_rng(42)
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)

        for trial in range(10):
            alpha = float(rng.uniform(0, 10))
            gamma = float(rng.uniform(0, 10))
            H = assemble_extended_hessian(
                g, alpha=alpha, gamma=gamma, Q_cycle=Q_cyc, L_max=3,
            )
            eig_min = eigsh(H, k=1, which="SA",
                            return_eigenvectors=False)[0]
            assert eig_min > 0, (
                f"trial {trial}: alpha={alpha:.3f}, gamma={gamma:.3f}, "
                f"lambda_min={eig_min:.6e}"
            )

    def test_regularizers_increase_lambda_min(self):
        """Adding regularizers cannot decrease lambda_min."""
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

        assert lam_ext >= lam_base - 1e-10, (
            f"lambda_min decreased: base={lam_base:.6e}, "
            f"ext={lam_ext:.6e}"
        )

    def test_negative_alpha_raises(self):
        """alpha < 0 must raise ValueError."""
        g = _make_test_graph(30, seed=0)
        with pytest.raises(ValueError):
            assemble_extended_hessian(g, alpha=-0.5)

    def test_negative_gamma_raises(self):
        """gamma < 0 must raise ValueError."""
        g = _make_test_graph(30, seed=0)
        with pytest.raises(ValueError):
            assemble_extended_hessian(g, gamma=-0.5)


# ----------------------------------------------------------------------
# Q_cascade
# ----------------------------------------------------------------------

class TestQCascade:

    def test_q_cascade_shape(self):
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        assert Q.shape == (30, 30)

    def test_q_cascade_symmetric(self):
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12, (
            "Q_cascade is not symmetric"
        )

    def test_q_cascade_psd(self):
        """Q_cascade = sum w(l) P_p^T P_p >= 0."""
        g = _make_test_graph(30, seed=0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cascade")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10, (
            f"Q_cascade has negative eigenvalue {eig_min:.6e}"
        )

    def test_q_cascade_empty_when_no_derived_edges(self):
        """No derived-from edges -> Q_cascade = 0."""
        G_nx = make_ba_graph(20, m=3, seed=0)
        g = EpistemicGraph(num_nodes=20)
        for u, v in G_nx.edges():
            g.add_support_edge(int(u), int(v), 1.0)

        Q = build_cascade_matrix_bounded(g, L_max=3)
        assert Q.nnz == 0
        assert Q.shape == (20, 20)


# ----------------------------------------------------------------------
# Extended system — eq. (5.10)
# ----------------------------------------------------------------------

class TestExtendedSystem:

    @pytest.mark.parametrize("alpha,gamma", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 2.0),
    ])
    def test_equilibrium_satisfies_linear_system(self, alpha, gamma):
        """Solving H_ext x = b gives the correct equilibrium."""
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
        H = assemble_extended_hessian(
            g, alpha=alpha, gamma=gamma, Q_cycle=Q_cyc, L_max=3,
        )

        rng = np.random.default_rng(1)
        b = rng.standard_normal(g.num_nodes)

        x_star = spsolve(H.tocsc(), b)
        res = H @ x_star - b
        assert np.max(np.abs(res)) < 1e-8, (
            f"alpha={alpha}, gamma={gamma}: ||Hx - b||_inf = "
            f"{np.max(np.abs(res)):.2e}"
        )

    def test_equilibrium_minimizes_energy(self):
        """Perturbing x_star increases E_tot (strict convexity)."""
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
            assert E_pert > E_star, (
                f"Perturbation reduced energy: E*={E_star:.6f}, "
                f"E_pert={E_pert:.6f}"
            )

    def test_equilibrium_inside_box(self):
        """Unconstrained solve should be finite."""
        g = _make_test_graph(50, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
        H = assemble_extended_hessian(
            g, alpha=0.5, gamma=0.5, Q_cycle=Q_cyc, L_max=3,
        )
        rng = np.random.default_rng(3)
        b = rng.standard_normal(g.num_nodes)

        x_star = spsolve(H.tocsc(), b)
        assert np.all(np.isfinite(x_star))


# ----------------------------------------------------------------------
# Base case — alpha = gamma = 0
# ----------------------------------------------------------------------

class TestBaseCase:

    def test_zero_alpha_gamma_recovers_base(self):
        """When alpha = gamma = 0, H_ext must equal H_0."""
        g = _make_test_graph(30, seed=0)
        H_base = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        H_ext = assemble_extended_hessian(g, alpha=0.0, gamma=0.0, L_max=3)
        diff = H_ext - H_base
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_base_hessian_is_spd(self):
        """H_0 = Lambda + L_S + L_D is SPD because Lambda > 0."""
        g = _make_test_graph(30, seed=0)
        H0 = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        eig_min = eigsh(H0, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0

    def test_alpha_only_increases_diagonal(self):
        """Adding alpha * Q_cascade increases (or keeps) diagonal entries."""
        g = _make_test_graph(30, seed=0)
        H0 = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        H1 = assemble_extended_hessian(g, alpha=1.0, gamma=0.0, L_max=3)

        d0 = H0.diagonal()
        d1 = H1.diagonal()
        assert np.all(d1 >= d0 - 1e-10)


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

class TestDiagnostics:

    def test_hessian_is_symmetric(self):
        g = _make_test_graph(30, seed=0)
        Q_cyc = build_cycle_matrix_fundamental(g, K_max=50)
        H = assemble_extended_hessian(
            g, alpha=0.5, gamma=0.5, Q_cycle=Q_cyc, L_max=3,
        )
        diff = H - H.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_hessian_sparse_format(self):
        g = _make_test_graph(30, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        assert isinstance(H, csr_matrix)

    def test_hessian_has_positive_diagonal(self):
        g = _make_test_graph(30, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        d = H.diagonal()
        assert np.all(d > 0), f"Non-positive diagonal: {d[d <= 0]}"

    def test_hessian_no_explicit_zeros(self):
        g = _make_test_graph(30, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        if H.nnz > 0:
            assert np.abs(H.data).min() > 0


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

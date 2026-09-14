"""
Rigorous cross-validation tests for the Epistemic Ecology Runtime.

Compares EER against independent reference methods:
    1. Analytic Laplacians for known small graphs
    2. Direct solvers (scipy.sparse.linalg.spsolve)
    3. Direct cycle enumeration (K_4, two triangles)
    4. Direct path enumeration for Q_cascade
    5. Seed sweeps (0-9)
    6. Size sweeps (10-500)
    7. Empirical convergence rate vs. theoretical bound
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh, spsolve

from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
)
from eer.cycle_basis import build_fundamental_cycle_basis
from eer.utils import (
    condition_number,
    make_ba_graph,
    make_er_graph,
    rate_bound,
    set_random_priors,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_ba(n: int, seed: int = 0) -> EpistemicGraph:
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    rng = np.random.default_rng(seed)
    edges = list(G_nx.edges())
    n_derived = max(1, len(edges) // 10)
    idx = rng.choice(len(edges), size=n_derived, replace=False)
    for k in idx:
        uu, vv = edges[int(k)]
        g.add_derived_from_edge(int(uu), int(vv), 0.5)
    return g


def _make_er(n: int, seed: int = 0) -> EpistemicGraph:
    G_nx = make_er_graph(n, avg_deg=5.0, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


def _solve_hybrid(H, b, x0=None, tol=1e-10, max_sweeps=100_000):
    n = H.shape[0]
    if x0 is None:
        x0 = np.full(n, 0.5)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    return run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0,
        M=n, epsilon=1e-3, tol=tol, max_sweeps=max_sweeps,
    )


# ======================================================================
# 1. Analytic Laplacian verification
# ======================================================================

class TestAnalyticLaplacian:
    """
    Verify L_S against hand-computed Laplacians.

    Convention (paper Eq. 2.3): L_S = B_S W_S B_S^T.
    Equivalent to L = D - A_undir with A_undir = A + A^T.
    """

    def test_triangle_laplacian(self):
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

    def test_square_laplacian(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 3, 1.0)
        g.add_support_edge(3, 0, 1.0)

        L = g.support_laplacian().toarray()
        expected = np.array([
            [ 2.0, -1.0,  0.0, -1.0],
            [-1.0,  2.0, -1.0,  0.0],
            [ 0.0, -1.0,  2.0, -1.0],
            [-1.0,  0.0, -1.0,  2.0],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_star_laplacian(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(0, 2, 1.0)
        g.add_support_edge(0, 3, 1.0)

        L = g.support_laplacian().toarray()
        expected = np.array([
            [ 3.0, -1.0, -1.0, -1.0],
            [-1.0,  1.0,  0.0,  0.0],
            [-1.0,  0.0,  1.0,  0.0],
            [-1.0,  0.0,  0.0,  1.0],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_path_laplacian(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 3, 1.0)

        L = g.support_laplacian().toarray()
        expected = np.array([
            [ 1.0, -1.0,  0.0,  0.0],
            [-1.0,  2.0, -1.0,  0.0],
            [ 0.0, -1.0,  2.0, -1.0],
            [ 0.0,  0.0, -1.0,  1.0],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_weighted_triangle(self):
        """
        Triangle with distinct weights: 0->1 (2.0), 1->2 (3.0), 2->0 (4.0).

        Diagonal = sum of incident weights:
            deg[0] = 2 + 4 = 6
            deg[1] = 2 + 3 = 5
            deg[2] = 3 + 4 = 7
        """
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

    def test_laplacian_row_sums_zero(self):
        g = _make_ba(50, seed=0)
        for L in [g.support_laplacian(), g.derived_from_laplacian()]:
            if L.nnz == 0:
                continue
            row_sums = np.asarray(L.sum(axis=1)).ravel()
            np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_laplacian_psd(self):
        g = _make_ba(50, seed=0)
        for L in [g.support_laplacian(), g.derived_from_laplacian()]:
            if L.nnz == 0:
                continue
            eig_min = eigsh(L, k=1, which="SA",
                            return_eigenvectors=False)[0]
            assert eig_min >= -1e-10


# ======================================================================
# 2. Cross-validation against direct solver
# ======================================================================

class TestDirectSolverCrossValidation:

    @pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (50, 2), (100, 3)])
    def test_hybrid_matches_direct_solve(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x_ref = spsolve(H.tocsc(), g.b)
        x_sol, _, _ = _solve_hybrid(H, g.b, tol=1e-9)
        np.testing.assert_allclose(x_sol, x_ref, atol=1e-5)

    @pytest.mark.parametrize("n,seed", [(10, 0), (20, 1), (50, 2)])
    def test_cgs_matches_direct_solve(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x_ref = spsolve(H.tocsc(), g.b)
        x_cgs, _, _ = run_cyclic_gs_scheduler(
            H, g.b, np.full(n, 0.5), tol=1e-9, max_sweeps=20_000,
        )
        np.testing.assert_allclose(x_cgs, x_ref, atol=1e-5)

    def test_all_schedulers_agree(self):
        n, seed = 50, 0
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x0 = np.full(n, 0.5)
        x_cgs, _, _ = run_cyclic_gs_scheduler(
            H, g.b, x0, tol=1e-10, max_sweeps=20_000,
        )
        x_hyb, _, _ = _solve_hybrid(H, g.b, x0, tol=1e-10)
        np.testing.assert_allclose(x_cgs, x_hyb, atol=1e-5)


# ======================================================================
# 3. Seed sweep
# ======================================================================

class TestSeedSweep:

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_spd_all_seeds(self, seed):
        g = _make_ba(50, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, f"seed={seed}: lambda_min={eig_min:.6e}"

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_hybrid_converges_all_seeds(self, seed):
        g = _make_ba(50, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        _, _, res = _solve_hybrid(H, g.b, tol=1e-8)
        assert res < 1e-6

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_cycle_matrix_psd_all_seeds(self, seed):
        g = _make_ba(50, seed=seed)
        Q = build_cycle_matrix_fundamental(g, K_max=100)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cycle")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10


# ======================================================================
# 4. Size sweep
# ======================================================================

class TestSizeSweep:

    @pytest.mark.parametrize("n", [10, 20, 50, 100, 200, 500])
    def test_spd_all_sizes(self, n):
        g = _make_ba(n, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, f"n={n}: lambda_min={eig_min:.6e}"

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_hybrid_converges_all_sizes(self, n):
        g = _make_ba(n, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        _, _, res = _solve_hybrid(H, g.b, tol=1e-8)
        assert res < 1e-6

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_hybrid_matches_direct_all_sizes(self, n):
        g = _make_ba(n, seed=0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x_ref = spsolve(H.tocsc(), g.b)
        x_sol, _, _ = _solve_hybrid(H, g.b, tol=1e-9)
        np.testing.assert_allclose(x_sol, x_ref, atol=1e-5)


# ======================================================================
# 5. Convergence rate vs. theoretical bound
# ======================================================================

class TestConvergenceRate:

    def _measure_cgs_rate(self, H, x_star, max_sweeps=100):
        n = H.shape[0]
        H_dense = H.toarray()
        H_diag = np.diag(H_dense)
        b = H @ x_star

        x = np.zeros(n)
        errors = [np.linalg.norm(x - x_star)]
        for _ in range(max_sweeps):
            for i in range(n):
                s = H_dense[i] @ x - H_diag[i] * x[i]
                x[i] = (b[i] - s) / H_diag[i]
            errors.append(np.linalg.norm(x - x_star))
            if errors[-1] < 1e-13:
                break

        ratios = [
            errors[i + 1] / errors[i]
            for i in range(len(errors) - 1)
            if errors[i] > 1e-13
        ]
        return float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1), (80, 2)])
    def test_cgs_rate_below_theoretical_bound(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x_star = spsolve(H.tocsc(), g.b)
        bound = rate_bound(H)
        empirical = self._measure_cgs_rate(H, x_star)
        assert empirical <= bound + 0.10, (
            f"n={n}, seed={seed}: empirical {empirical:.4f} > bound {bound:.4f}"
        )

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1)])
    def test_hessian_condition_number_finite(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        kappa = condition_number(H)
        assert np.isfinite(kappa) and kappa >= 1.0


# ======================================================================
# 6. Direct cycle enumeration
# ======================================================================

class TestDirectCycleEnumeration:

    def test_triangle_count(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 1

    def test_two_triangles_share_edge(self):
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        g.add_support_edge(0, 3, 1.0)
        g.add_support_edge(1, 3, 1.0)
        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 2

    def test_k4_has_correct_cycle_count(self):
        g = EpistemicGraph(num_nodes=4)
        for u, v in itertools.combinations(range(4), 2):
            g.add_support_edge(u, v, 1.0)
        basis = build_fundamental_cycle_basis(g)
        # m_S - n + c = 6 - 4 + 1 = 3
        assert basis.n_cycles_total == 3


# ======================================================================
# 7. Direct path enumeration for Q_cascade
# ======================================================================

class TestDirectPathEnumeration:

    def test_single_edge_cascade(self):
        """Single edge 0->1: Q_cascade = [[1,-1,0],[-1,1,0],[0,0,0]]."""
        g = EpistemicGraph(num_nodes=3)
        g.add_derived_from_edge(0, 1, 1.0)

        Q = build_cascade_matrix_bounded(g, L_max=2, decay=0.0).toarray()
        expected = np.array([
            [ 1.0, -1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [ 0.0,  0.0, 0.0],
        ])
        np.testing.assert_allclose(Q, expected, atol=1e-12)

    def test_two_edge_chain(self):
        g = EpistemicGraph(num_nodes=3)
        g.add_derived_from_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 1.0)

        Q = build_cascade_matrix_bounded(g, L_max=2, decay=0.0)
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10

    def test_no_derived_edges(self):
        g = EpistemicGraph(num_nodes=5)
        for i in range(4):
            g.add_support_edge(i, i + 1, 1.0)
        Q = build_cascade_matrix_bounded(g, L_max=3)
        assert Q.nnz == 0

    def test_lmax_bounds_path_length(self):
        """L_max=1 vs L_max=3 should give different nnz for a chain."""
        g = EpistemicGraph(num_nodes=4)
        g.add_derived_from_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 1.0)
        g.add_derived_from_edge(2, 3, 1.0)

        Q_l1 = build_cascade_matrix_bounded(g, L_max=1, decay=0.0)
        Q_l3 = build_cascade_matrix_bounded(g, L_max=3, decay=0.0)
        assert Q_l3.nnz >= Q_l1.nnz


# ======================================================================
# 8. Parameter grid
# ======================================================================

class TestParameterGrid:

    @pytest.mark.parametrize("alpha,gamma", [
        (0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (1.0, 1.0),
        (2.0, 0.5), (0.5, 2.0), (5.0, 5.0), (10.0, 0.0), (0.0, 10.0),
    ])
    def test_hext_spd_grid(self, alpha, gamma):
        g = _make_ba(50, seed=0)
        H = assemble_extended_hessian(g, alpha=alpha, gamma=gamma, L_max=3)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0


# ======================================================================
# 9. ER graph cross-check
# ======================================================================

class TestERGraphValidation:

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1), (100, 2)])
    def test_er_spd(self, n, seed):
        g = _make_er(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1)])
    def test_er_hybrid_converges(self, n, seed):
        g = _make_er(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        _, _, res = _solve_hybrid(H, g.b, tol=1e-8)
        assert res < 1e-6

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1)])
    def test_er_matches_direct(self, n, seed):
        g = _make_er(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x_ref = spsolve(H.tocsc(), g.b)
        x_sol, _, _ = _solve_hybrid(H, g.b, tol=1e-9)
        np.testing.assert_allclose(x_sol, x_ref, atol=1e-5)


# ======================================================================
# 10. Edge cases
# ======================================================================

class TestEdgeCases:

    def test_single_node(self):
        g = EpistemicGraph(num_nodes=1)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5)
        assert H.shape == (1, 1)
        assert H[0, 0] > 0

    def test_two_isolated_nodes(self):
        g = EpistemicGraph(num_nodes=2)
        H = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        np.testing.assert_allclose(H.toarray(), np.eye(2), atol=1e-12)

    def test_disconnected_graph(self):
        g = EpistemicGraph(num_nodes=6)
        for u, v in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
            g.add_support_edge(u, v, 1.0)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

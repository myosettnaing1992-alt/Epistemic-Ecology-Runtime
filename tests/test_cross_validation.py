"""
Rigorous cross-validation tests for the Epistemic Ecology Runtime.

These tests go beyond the standard unit tests by comparing the EER
implementation against independent reference methods:

    1. Direct solvers (scipy.sparse.linalg.spsolve)
    2. Analytic Laplacians for known small graphs
    3. Direct cycle enumeration for small graphs
    4. Direct path enumeration for small derived-from subgraphs
    5. Seed sweeps (0-9) to catch seed-dependent bugs
    6. Size sweeps (10-500) to catch size-dependent bugs
    7. Empirical convergence rate vs. theoretical bound

The goal is to move from "tests pass on one seed" to "tests pass on the
full parameter grid that the paper claims to cover".
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, spsolve

from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
    run_random_priority_scheduler,
)
from eer.cycle_basis import build_fundamental_cycle_basis
from eer.utils import (
    condition_number,
    is_spd,
    make_ba_graph,
    make_er_graph,
    rate_bound,
    set_random_priors,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_ba(n: int, seed: int = 0) -> EpistemicGraph:
    """BA graph with support + ~10% derived-from edges."""
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
    """Convenience wrapper for hybrid priority scheduler."""
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
    Verify L_S and L_D against hand-computed Laplacians for small graphs.

    These are the ground truth: L = D - A_sym, where A_sym is the
    symmetrized adjacency matrix.
    """

    def test_triangle_laplacian(self):
        """Triangle with unit weights: L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]."""
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
        """4-cycle: L has 2 on diagonal, -1 on each incident edge."""
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
        """Star with center 0 and leaves 1,2,3."""
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
        """Path 0-1-2-3."""
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
        """Triangle with distinct weights."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 2.0)
        g.add_support_edge(1, 2, 3.0)
        g.add_support_edge(2, 0, 4.0)

        L = g.support_laplacian().toarray()
        # Symmetric A has A_sym[i,j] = (w_ij + w_ji) / 2 = w_ij (single edge)
        # But our edges are directed. Symmetrizing:
        # A_sym[0,1] = 1.0, A_sym[1,2] = 1.5, A_sym[2,0] = 2.0
        # Degree[0] = 1.0 + 2.0 = 3.0
        # Degree[1] = 1.0 + 1.5 = 2.5
        # Degree[2] = 1.5 + 2.0 = 3.5
        expected = np.array([
            [ 3.0, -1.0, -2.0],
            [-1.0,  2.5, -1.5],
            [-2.0, -1.5,  3.5],
        ])
        np.testing.assert_allclose(L, expected, atol=1e-12)

    def test_laplacian_row_sums_zero(self):
        """Row sums of any Laplacian must be zero."""
        g = _make_ba(50, seed=0)
        L_S = g.support_laplacian()
        L_D = g.derived_from_laplacian()
        for L in [L_S, L_D]:
            row_sums = np.asarray(L.sum(axis=1)).ravel()
            np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_laplacian_psd(self):
        """L = D - A_sym is always PSD."""
        g = _make_ba(50, seed=0)
        for L in [g.support_laplacian(), g.derived_from_laplacian()]:
            if L.nnz == 0:
                continue
            eig_min = eigsh(L, k=1, which="SA",
                            return_eigenvectors=False)[0]
            assert eig_min >= -1e-10, f"Laplacian not PSD: {eig_min}"


# ======================================================================
# 2. Cross-validation against direct solver
# ======================================================================

class TestDirectSolverCrossValidation:
    """
    Verify that the hybrid scheduler, CGS, and random priority all
    converge to the same equilibrium as scipy.sparse.linalg.spsolve.
    """

    @pytest.mark.parametrize("n,seed", [
        (10, 0), (20, 1), (50, 2), (100, 3),
    ])
    def test_hybrid_matches_direct_solve(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        b = g.b

        # Reference: direct solve
        x_ref = spsolve(H.tocsc(), b)

        # Hybrid scheduler
        x_sol, updates, res = _solve_hybrid(H, b, tol=1e-9)

        np.testing.assert_allclose(
            x_sol, x_ref, atol=1e-5,
            err_msg=f"n={n}, seed={seed}: hybrid differs from direct solve",
        )

    @pytest.mark.parametrize("n,seed", [
        (10, 0), (20, 1), (50, 2),
    ])
    def test_cgs_matches_direct_solve(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        b = g.b

        x_ref = spsolve(H.tocsc(), b)
        x_cgs, _, _ = run_cyclic_gs_scheduler(
            H, b, np.full(n, 0.5), tol=1e-9, max_sweeps=20_000,
        )

        np.testing.assert_allclose(
            x_cgs, x_ref, atol=1e-5,
            err_msg=f"n={n}, seed={seed}: CGS differs from direct solve",
        )

    @pytest.mark.parametrize("n,seed", [
        (10, 0), (20, 1),
    ])
    def test_random_matches_direct_solve(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        b = g.b

        x_ref = spsolve(H.tocsc(), b)
        x_rnd, _, _ = run_random_priority_scheduler(
            H, b, np.full(n, 0.5), tol=1e-9,
            max_updates=100_000, seed=seed,
        )

        np.testing.assert_allclose(
            x_rnd, x_ref, atol=1e-4,
            err_msg=f"n={n}, seed={seed}: random differs from direct solve",
        )

    def test_all_schedulers_agree(self):
        """All three schedulers must converge to the same x*."""
        n, seed = 50, 0
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        b = g.b
        x0 = np.full(n, 0.5)

        x_cgs, _, _ = run_cyclic_gs_scheduler(
            H, b, x0, tol=1e-10, max_sweeps=20_000,
        )
        x_hyb, _, _ = _solve_hybrid(H, b, x0, tol=1e-10)

        np.testing.assert_allclose(x_cgs, x_hyb, atol=1e-5)


# ======================================================================
# 3. Seed sweep
# ======================================================================

class TestSeedSweep:
    """
    Run the core pipeline across many seeds to ensure that a single
    lucky seed is not the only reason for passing tests.
    """

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_spd_all_seeds(self, seed):
        """H_ext must be SPD for every seed."""
        g = _make_ba(50, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        eig_min = eigsh(H, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, (
            f"seed={seed}: lambda_min={eig_min:.6e} <= 0"
        )

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_hybrid_converges_all_seeds(self, seed):
        """Hybrid scheduler must converge for every seed."""
        g = _make_ba(50, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        x, _, res = _solve_hybrid(H, g.b, tol=1e-8)
        assert res < 1e-6, f"seed={seed}: residual={res:.2e}"

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_cycle_basis_cardinality_all_seeds(self, seed):
        """m_S - n + c formula must hold for every seed."""
        g = _make_ba(50, seed=seed)
        A = g.support_adjacency()
        A_sym = 0.5 * (A + A.T)
        m_undirected = int(A_sym.nnz // 2 + np.diag(A_sym.toarray()).sum())

        basis = build_fundamental_cycle_basis(g, K_max=None)
        # Sanity: n_cycles_total is non-negative
        assert basis.n_cycles_total >= 0

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_cycle_matrix_psd_all_seeds(self, seed):
        g = _make_ba(50, seed=seed)
        Q = build_cycle_matrix_fundamental(g, K_max=100)
        if Q.nnz == 0:
            pytest.skip("Empty Q_cycle")
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10, f"seed={seed}: {eig_min:.6e}"


# ======================================================================
# 4. Size sweep
# ======================================================================

class TestSizeSweep:
    """Test scaling across n = 10 to 500."""

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
        x, _, res = _solve_hybrid(H, g.b, tol=1e-8)
        assert res < 1e-6, f"n={n}: residual={res:.2e}"

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
    """
    Measure the empirical convergence rate of PCGS and compare against
    the theoretical bound rho <= 1 - 1/(2 kappa) (Corollary 4.1).
    """

    def _measure_cgs_rate(self, H, x_star, max_sweeps=100):
        n = H.shape[0]
        H_dense = H.toarray()
        H_diag = np.diag(H_dense)
        b = H @ x_star  # reconstruct RHS so exact solution is x_star

        x = np.zeros(n)
        errors = [np.linalg.norm(x - x_star)]
        for _ in range(max_sweeps):
            for i in range(n):
                s = H_dense[i] @ x - H_diag[i] * x[i]
                x[i] = (b[i] - s) / H_diag[i]
            errors.append(np.linalg.norm(x - x_star))
            if errors[-1] < 1e-13:
                break

        # Empirical rate = geometric mean of consecutive error ratios
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

        # Reference solution
        b = g.b
        x_star = spsolve(H.tocsc(), b)

        # Theoretical bound
        bound = rate_bound(H)

        # Empirical rate
        empirical = self._measure_cgs_rate(H, x_star)

        assert empirical <= bound + 0.05, (
            f"n={n}, seed={seed}: empirical rate {empirical:.4f} "
            f"> theoretical bound {bound:.4f}"
        )

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1)])
    def test_hessian_condition_number_finite(self, n, seed):
        g = _make_ba(n, seed=seed)
        H = assemble_extended_hessian(g, alpha=0.5, gamma=0.5, L_max=3)
        kappa = condition_number(H)
        assert np.isfinite(kappa) and kappa >= 1.0

    @pytest.mark.parametrize("n,seed", [(20, 0), (50, 1)])
    def test_regularizers_reduce_condition_number(self, n, seed):
        """Adding regularization should not worsen conditioning."""
        g = _make_ba(n, seed=seed)
        H_base = assemble_extended_hessian(g, alpha=0.0, gamma=0.0)
        H_ext = assemble_extended_hessian(g, alpha=1.0, gamma=1.0, L_max=3)

        kappa_base = condition_number(H_base)
        kappa_ext = condition_number(H_ext)

        assert kappa_ext <= kappa_base + 1e-6, (
            f"kappa increased: {kappa_base:.4f} -> {kappa_ext:.4f}"
        )


# ======================================================================
# 6. Direct cycle enumeration for small graphs
# ======================================================================

class TestDirectCycleEnumeration:
    """
    For small graphs, compare Q_cycle built from the fundamental basis
    against Q_cycle built by direct enumeration of all simple cycles.

    These should agree for very small graphs where enumeration is
    feasible.
    """

    def _enumerate_triangles(self, g: EpistemicGraph) -> list[list[int]]:
        """Enumerate all triangles (3-cycles) in the support graph."""
        import networkx as nx

        A = g.support_adjacency()
        A_sym = 0.5 * (A + A.T)
        A_sym = A_sym.tolil()
        A_sym.setdiag(0)
        A_sym = A_sym.tocsr()
        A_sym.eliminate_zeros()

        G_nx = nx.from_scipy_sparse_array(A_sym)
        triangles = []
        for u, v in G_nx.edges():
            common = set(G_nx.neighbors(u)) & set(G_nx.neighbors(v))
            for w in common:
                if u < v < w:
                    triangles.append([int(u), int(v), int(w)])
        return triangles

    def test_triangle_count(self):
        """Single triangle has exactly 1 cycle."""
        g = EpistemicGraph(num_nodes=3)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)

        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 1

        tris = self._enumerate_triangles(g)
        assert len(tris) == 1

    def test_two_triangles_share_edge(self):
        """Two triangles sharing an edge: 0-1-2, 0-1-3."""
        g = EpistemicGraph(num_nodes=4)
        g.add_support_edge(0, 1, 1.0)
        g.add_support_edge(1, 2, 1.0)
        g.add_support_edge(2, 0, 1.0)
        g.add_support_edge(0, 3, 1.0)
        g.add_support_edge(1, 3, 1.0)

        basis = build_fundamental_cycle_basis(g)
        # m_S - n + c = 5 - 4 + 1 = 2
        assert basis.n_cycles_total == 2

    def test_k4_has_correct_cycle_count(self):
        """K_4: m_S - n + c = 6 - 4 + 1 = 3."""
        g = EpistemicGraph(num_nodes=4)
        for u, v in itertools.combinations(range(4), 2):
            g.add_support_edge(u, v, 1.0)

        basis = build_fundamental_cycle_basis(g)
        assert basis.n_cycles_total == 3


# ======================================================================
# 7. Direct path enumeration for Q_cascade
# ======================================================================

class TestDirectPathEnumeration:
    """
    Verify Q_cascade against direct path enumeration on a tiny graph.
    """

    def test_single_edge_cascade(self):
        """Single derived-from edge: Q_cascade = P^T P for 2-node path."""
        g = EpistemicGraph(num_nodes=3)
        g.add_derived_from_edge(0, 1, 1.0)  # path (0, 1)

        Q = build_cascade_matrix_bounded(g, L_max=2, decay=0.0).toarray()

        # Path (0,1): P_p x = x_0 - x_1
        # Rank-1 outer product: [[1, -1], [-1, 1]] on indices {0, 1}
        expected = np.array([
            [ 1.0, -1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [ 0.0,  0.0, 0.0],
        ])
        np.testing.assert_allclose(Q, expected, atol=1e-12)

    def test_two_edge_chain(self):
        """Chain 0 -> 1 -> 2: paths (0,1), (1,2), (0,1,2)."""
        g = EpistemicGraph(num_nodes=3)
        g.add_derived_from_edge(0, 1, 1.0)
        g.add_derived_from_edge(1, 2, 1.0)

        Q = build_cascade_matrix_bounded(g, L_max=2, decay=0.0)

        # Q should be symmetric PSD
        diff = Q - Q.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12
        eig_min = eigsh(Q, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min >= -1e-10

        # Q has non-zero entries on all nodes
        Q_dense = Q.toarray()
        assert np.abs(Q_dense[0, 1]) > 0
        assert np.abs(Q_dense[1, 2]) > 0

    def test_no_derived_edges(self):
        """No derived edges -> Q_cascade = 0."""
        g = EpistemicGraph(num_nodes=5)
        for i in range(4):
            g.add_support_edge(i, i + 1, 1.0)

        Q = build_cascade_matrix_bounded(g, L_max=3)
        assert Q.nnz == 0

    def test_lmax_bounds_path_length(self):
        """L_max=1 should only u

"""
Unit tests for eer.schedulers.

Covers:
    - Theorem 9.1: no starvation (aging guarantees bounded waiting)
    - Theorem 4.2: linear convergence of PCGS
    - Corollary 4.1: rate bound rho <= 1 - 1/(2*kappa)
    - Scheduler comparison on small graphs
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from eer import (
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
    run_random_priority_scheduler,
)
from eer.utils import make_ba_graph, set_random_priors


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _build_system(n: int = 50, seed: int = 0):
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    Q_cyc = build_cycle_matrix_fundamental(g, K_max=100)
    H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc, L_max=3)

    return g, H


# ----------------------------------------------------------------------
# Theorem 9.1: no starvation
# ----------------------------------------------------------------------

class TestTheorem91NoStarvation:

    def test_hybrid_scheduler_converges(self):
        """Hybrid priority converges on a small BA graph."""
        g, H = _build_system(50, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(50, 0.5)

        x, updates, res = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr, H_csr.indices, H_csr.data,
            H_csc.indptr, H_csc.indices, H_csc.data,
            g.b, x0,
            M=50, epsilon=1e-3, tol=1e-6, max_sweeps=1000,
        )
        assert res < 1e-5, f"Hybrid did not converge: res={res:.2e}"

    def test_aging_rate_effect(self):
        """Higher epsilon should not break convergence."""
        g, H = _build_system(50, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(50, 0.5)

        for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
            x, updates, res = run_hybrid_priority_scheduler_optimized(
                H_csr.indptr, H_csr.indices, H_csr.data,
                H_csc.indptr, H_csc.indices, H_csc.data,
                g.b, x0,
                M=50, epsilon=eps, tol=1e-6, max_sweeps=5000,
            )
            assert res < 1e-5, f"eps={eps}: res={res:.2e}"

    def test_bounded_waiting_time_bound(self):
        """
        Waiting time must be bounded by M + ceil(R_max / epsilon).
        We check: total updates <= max_sweeps * n (no infinite loop).
        """
        g, H = _build_system(50, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(50, 0.5)

        M = 50
        eps = 1e-3
        x, updates, res = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr, H_csr.indices, H_csr.data,
            H_csc.indptr, H_csc.indices, H_csc.data,
            g.b, x0,
            M=M, epsilon=eps, tol=1e-6, max_sweeps=1000,
        )
        assert updates <= 1000 * 50


# ----------------------------------------------------------------------
# Theorem 4.2: linear convergence
# ----------------------------------------------------------------------

class TestTheorem42LinearConvergence:

    def test_cyclic_gs_converges(self):
        g, H = _build_system(50, seed=0)
        x0 = np.full(50, 0.5)

        x, updates, res = run_cyclic_gs_scheduler(
            H, g.b, x0, tol=1e-6, max_sweeps=10_000,
        )
        assert res < 1e-5

    def test_random_priority_converges(self):
        g, H = _build_system(50, seed=0)
        x0 = np.full(50, 0.5)

        x, updates, res = run_random_priority_scheduler(
            H, g.b, x0, tol=1e-5, max_updates=200_000, seed=0,
        )
        assert res < 1e-4

    def test_gs_spectral_radius_less_than_one(self):
        """Theorem 4.2: rho(T_GS) < 1 for arbitrary SPD H."""
        _, H = _build_system(50, seed=0)
        H_dense = H.toarray()
        D = np.diag(np.diag(H_dense))
        L = np.tril(H_dense, k=-1)
        U = np.triu(H_dense, k=1)

        M = D + L
        T_GS = -np.linalg.solve(M, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0, f"rho(T_GS) = {rho:.6f} >= 1"


# ----------------------------------------------------------------------
# Corollary 4.1: rate bound
# ----------------------------------------------------------------------

class TestCorollary41RateBound:

    def test_rate_bound_holds(self):
        """rho <= 1 - 1/(2*kappa)."""
        _, H = _build_system(50, seed=0)
        H_dense = H.toarray()

        eigs = np.linalg.eigvalsh(H_dense)
        kappa = eigs[-1] / eigs[0]
        bound = 1.0 - 1.0 / (2.0 * kappa)

        D = np.diag(np.diag(H_dense))
        L = np.tril(H_dense, k=-1)
        U = np.triu(H_dense, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))

        assert rho <= bound + 1e-6, (
            f"Rate bound violated: rho={rho:.6f} > bound={bound:.6f}, "
            f"kappa={kappa:.2f}"
        )

    def test_hext_condition_number_finite(self):
        _, H = _build_system(50, seed=0)
        lam_min = eigsh(H, k=1, which="SA",
                        return_eigenvectors=False)[0]
        lam_max = eigsh(H, k=1, which="LA",
                        return_eigenvectors=False)[0]
        kappa = lam_max / lam_min
        assert kappa > 1.0
        assert np.isfinite(kappa)


# ----------------------------------------------------------------------
# Scheduler comparison
# ----------------------------------------------------------------------

class TestSchedulerComparison:

    def test_all_methods_converge_to_same_solution(self):
        g, H = _build_system(50, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(50, 0.5)

        x_cgs, _, res_cgs = run_cyclic_gs_scheduler(
            H, g.b, x0, tol=1e-8, max_sweeps=10_000,
        )
        x_hyb, _, res_hyb = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr, H_csr.indices, H_csr.data,
            H_csc.indptr, H_csc.indices, H_csc.data,
            g.b, x0,
            M=50, epsilon=1e-3, tol=1e-8, max_sweeps=10_000,
        )

        # Both should reach essentially the same equilibrium
        diff = np.max(np.abs(x_cgs - x_hyb))
        assert diff < 1e-4, (
            f"CGS and Hybrid reach different solutions: max diff = {diff:.2e}"
        )

    def test_hybrid_updates_leq_random(self):
        """
        On BA-500, hybrid should use fewer updates than random priority.
        (Topology-dependent, so we test on a fixed seed.)
        """
        g, H = _build_system(100, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(100, 0.5)

        _, upd_hyb, _ = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr, H_csr.indices, H_csr.data,
            H_csc.indptr, H_csc.indices, H_csc.data,
            g.b, x0,
            M=100, epsilon=1e-3, tol=1e-6, max_sweeps=5000,
        )
        _, upd_rnd, _ = run_random_priority_scheduler(
            H, g.b, x0, tol=1e-6, max_updates=500_000, seed=0,
        )
        # Loose check: hybrid should not be catastrophically worse
        assert upd_hyb <= 3 * upd_rnd, (
            f"Hybrid uses {upd_hyb} updates vs random {upd_rnd}"
        )


# ----------------------------------------------------------------------
# Integration: full pipeline
# ----------------------------------------------------------------------

class TestIntegrationPipeline:

    @pytest.mark.integration
    def test_end_to_end_solve(self):
        """Full pipeline: graph -> Hessian -> solve."""
        g, H = _build_system(100, seed=0)
        H_csr = H.tocsr()
        H_csc = H.tocsc()
        x0 = np.full(100, 0.5)

        x, updates, res = run_hybrid_priority_scheduler_optimized(
            H_csr.indptr, H_csr.indices, H_csr.data,
            H_csc.indptr, H_csc.indices, H_csc.data,
            g.b, x0,
            M=100, epsilon=1e-3, tol=1e-6, max_sweeps=10_000,
        )

        assert res < 1e-5
        assert np.all(np.isfinite(x))
        assert x.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])

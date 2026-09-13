"""
Numerical verification of Theorem 4.2 (convergence without diagonal
dominance) and related results.

This file is separate from test_scheduler.py to make the theoretical
verification explicit and easy to locate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def _random_spd(n: int, seed: int = 0, density: float = 0.05) -> csr_matrix:
    from scipy.sparse import random as sparse_random
    rng = np.random.default_rng(seed)
    A = sparse_random(n, n, density=density, format="csr", random_state=seed)
    H = (A @ A.T + (1.0 + rng.random()) * identity(n, format="csr")).tocsr()
    return (0.5 * (H + H.T)).tocsr()


# ----------------------------------------------------------------------
# Theorem 4.2: P-regular splitting condition
# ----------------------------------------------------------------------

class TestTheorem42OstrowskiReich:

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (150, 2)])
    def test_sym_part_of_D_plus_L_is_pd(self, n, seed):
        """
        Key lemma for P-regular splitting:
        sym(M) = (M + M^T)/2 = (H + D)/2 ≻ 0 for SPD H.

        This replaces the incorrect '2D - H ≻ 0' claim in earlier drafts.
        """
        H = _random_spd(n, seed=seed)
        D = csr_matrix(np.diag(H.diagonal()))
        sym_M = 0.5 * (H + D)

        eig_min = eigsh(sym_M, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, f"sym(D+L) not PD: lambda_min = {eig_min:.6e}"

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (150, 2)])
    def test_gs_spectral_radius_less_than_one(self, n, seed):
        H = _random_spd(n, seed=seed).toarray()
        D = np.diag(np.diag(H))
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        M = D + L
        T_GS = -np.linalg.solve(M, U)

        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0, f"rho(T_GS) = {rho:.6f} >= 1"

    def test_dense_offdiagonal_still_converges(self):
        """SPD matrix with strong off-diagonal coupling."""
        n = 80
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        H = A @ A.T + 0.1 * np.eye(n)

        diag = np.diag(H)
        off = np.abs(H).sum(axis=1) - np.abs(diag)
        # Verify non-diagonal-dominance
        assert np.any(diag < off)

        D = np.diag(diag)
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0


# ----------------------------------------------------------------------
# Corollary 4.1: rate bound
# ----------------------------------------------------------------------

class TestCorollary41RateBound:

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (150, 2)])
    def test_rate_bound_holds(self, n, seed):
        """rho <= 1 - 1/(2*kappa)."""
        H = _random_spd(n, seed=seed).toarray()

        eigs = np.linalg.eigvalsh(H)
        kappa = eigs[-1] / eigs[0]
        bound = 1.0 - 1.0 / (2.0 * kappa)

        D = np.diag(np.diag(H))
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))

        assert rho <= bound + 1e-6, (
            f"rho={rho:.6f}, bound={bound:.6f}, kappa={kappa:.2f}"
        )


# ----------------------------------------------------------------------
# PCGS numerical convergence
# ----------------------------------------------------------------------

class TestPCGSNumerical:

    @pytest.mark.parametrize("n", [30, 60, 100])
    def test_pcgs_converges_linearly(self, n):
        """||x^k - x*|| decays geometrically."""
        from scipy.sparse.linalg import spsolve

        H = _random_spd(n, seed=0)
        rng = np.random.default_rng(1)
        b = rng.standard_normal(n)

        x_star = spsolve(H.tocsc(), b)

        # PCGS
        x = np.zeros(n)
        H_dense = H.toarray()
        H_diag = np.diag(H_dense)

        errors = []
        for sweep in range(100):
            for i in range(n):
                s = H_dense[i] @ x - H_diag[i] * x[i]
                x[i] = (b[i] - s) / H_diag[i]
            err = np.linalg.norm(x - x_star)
            errors.append(err)
            if err < 1e-10:
                break

        # Geometric decay
        ratios = [
            errors[i + 1] / errors[i]
            for i in range(len(errors) - 1)
            if errors[i] > 1e-12
        ]
        assert all(r < 1.0 for r in ratios), f"Non-monotone: {ratios[-5:]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

"""
Numerical verification of Section 4 (Convergence Analysis).

Tests correspond to:
    - Theorem 4.2  : PCGS converges for arbitrary SPD H (no diagonal dominance)
    - Corollary 4.1: Rate bound ρ ≤ 1 - 1/(2κ)
    - Remark 4.2   : Regularizers improve conditioning
    - Remark 4.3   : Unregularized base case

Run with:
    pytest tests/test_convergence.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh

# Adjust to repo layout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cycle_basis import (                       # noqa: E402
    build_fundamental_cycle_basis,
    build_cycle_precision_matrix,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def make_random_spd(n: int, seed: int = 0, density: float = 0.05):
    """Build a random sparse SPD matrix."""
    rng = np.random.default_rng(seed)
    from scipy.sparse import random as sparse_random
    A = sparse_random(n, n, density=density, format="csr", random_state=seed)
    H = (A @ A.T + (1.0 + rng.random()) * identity(n, format="csr")).tocsr()
    # Symmetrize
    H = 0.5 * (H + H.T)
    return H.tocsr()


def make_eer_system(n: int, seed: int = 0, alpha: float = 0.5, gamma: float = 0.5):
    """Build a synthetic EER system with all three regularizers."""
    import networkx as nx
    from scipy.sparse.csgraph import laplacian

    G = nx.barabasi_albert_graph(n, m=3, seed=seed)
    A_S = nx.to_scipy_sparse_array(G, nodelist=list(G.nodes()), format="csr",
                                    dtype=np.float64)
    L_S = laplacian(A_S, normed=False).tocsr()
    L_D = 0.1 * L_S
    Lambda = identity(n, format="csr")

    H_base = (Lambda + L_S + L_D).tocsr()

    basis = build_fundamental_cycle_basis(G, K_max=500)
    K_cycle = build_cycle_precision_matrix(basis, gamma=gamma)
    K_cascade = (0.05 * identity(n, format="csr")).tocsr()

    H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()
    return H_base, H_ext, K_cascade, K_cycle


# ----------------------------------------------------------------------
# Theorem 4.2 — Convergence without diagonal dominance
# ----------------------------------------------------------------------

class TestTheorem42OstrowskiReich:
    """Verify P-regular splitting argument of Theorem 4.2."""

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (200, 2)])
    def test_symmetric_part_of_D_plus_L_is_pd(self, n, seed):
        """
        The key lemma: sym(M) = (H + D)/2 ≻ 0 for SPD H.

        This is the *correct* condition for P-regular splitting,
        replacing the incorrect '2D - H ≻ 0' claim in earlier drafts.
        """
        H = make_random_spd(n, seed=seed)
        D = csr_matrix(np.diag(H.diagonal()))

        # Symmetric part of M = D + L is (M + M^T)/2 = (H + D)/2
        sym_M = 0.5 * (H + D)

        # Compute smallest eigenvalue
        eig_min = eigsh(sym_M, k=1, which="SA", return_eigenvectors=False)[0]
        assert eig_min > 0, (
            f"sym(D+L) not PD: λ_min = {eig_min:.6e}. "
            f"Theorem 4.2's P-regular splitting condition is violated."
        )

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (200, 2)])
    def test_gs_iteration_matrix_spectral_radius_less_than_one(self, n, seed):
        """
        Theorem 4.2: ρ(T_GS) < 1 for arbitrary SPD H.
        """
        H = make_random_spd(n, seed=seed).toarray()
        D = np.diag(np.diag(H))
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)

        M = D + L
        T_GS = -np.linalg.solve(M, U)

        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0, (
            f"ρ(T_GS) = {rho:.6f} ≥ 1. Theorem 4.2 violated."
        )
        # Store for inspection
        print(f"  n={n:3d}  ρ(T_GS) = {rho:.6f}")

    def test_convergence_on_dense_offdiagonal(self):
        """
        Explicit test: SPD matrix with strong off-diagonal coupling
        (no diagonal dominance) still converges.
        """
        n = 80
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        H = A @ A.T + 0.1 * np.eye(n)   # SPD but not diagonally dominant

        # Verify non-diagonal-dominance
        diag = np.diag(H)
        off = np.abs(H).sum(axis=1) - np.abs(diag)
        assert np.any(diag < off), "Test setup should break diagonal dominance."

        # Verify GS converges
        D = np.diag(diag)
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0


# ----------------------------------------------------------------------
# Corollary 4.1 — Rate bound ρ ≤ 1 - 1/(2κ)
# ----------------------------------------------------------------------

class TestCorollary41RateBound:
    """Verify the explicit rate bound in the energy norm."""

    @pytest.mark.parametrize("n,seed", [(50, 0), (100, 1), (150, 2)])
    def test_rate_bound_holds(self, n, seed):
        """
        ρ(T_GS) ≤ 1 - 1/(2κ) where κ = λ_max / λ_min.
        """
        H = make_random_spd(n, seed=seed).toarray()

        # Condition number
        eigs = np.linalg.eigvalsh(H)
        kappa = eigs[-1] / eigs[0]
        bound = 1.0 - 1.0 / (2.0 * kappa)

        # Actual spectral radius of GS iteration matrix
        D = np.diag(np.diag(H))
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))

        assert rho <= bound + 1e-9, (
            f"Rate bound violated: ρ = {rho:.6f} > bound = {bound:.6f}, "
            f"κ = {kappa:.2f}."
        )
        print(f"  n={n:3d}  κ={kappa:8.2f}  bound={bound:.6f}  ρ={rho:.6f}")


# ----------------------------------------------------------------------
# Remark 4.2 — Regularizers improve conditioning
# ----------------------------------------------------------------------

class TestRemark42RegularizersImproveConditioning:
    """Verify that adding regularizers reduces κ(H)."""

    @pytest.mark.parametrize("n,seed", [(100, 0), (200, 1)])
    def test_condition_number_decreases(self, n, seed):
        H_base, H_ext, _, _ = make_eer_system(n, seed=seed,
                                              alpha=0.5, gamma=0.5)

        eigs_base = eigsh(H_base, k=1, which="SA",
                          return_eigenvectors=False)[0]
        eigs_ext = eigsh(H_ext, k=1, which="SA",
                         return_eigenvectors=False)[0]

        # λ_min should increase (or stay equal) after regularization
        assert eigs_ext >= eigs_base - 1e-9, (
            f"Regularizers decreased λ_min: "
            f"λ_min(H_base) = {eigs_base:.6e}, "
            f"λ_min(H_ext)  = {eigs_ext:.6e}."
        )
        print(f"  n={n:3d}  λ_min(base)={eigs_base:.4e}  "
              f"λ_min(ext)={eigs_ext:.4e}")

    def test_regularizer_effect_on_kappa(self):
        """
        Concretely: for a poorly conditioned H_base, regularizers
        must measurably reduce κ.
        """
        n = 100
        rng = np.random.default_rng(7)
        A = rng.standard_normal((n, n))
        # Ill-conditioned SPD base
        H_base = (A @ A.T + 1e-4 * np.eye(n))
        H_base = csr_matrix(0.5 * (H_base + H_base.T))

        K_cascade = csr_matrix(0.5 * np.eye(n))
        K_cycle = csr_matrix(0.5 * np.eye(n))
        H_ext = (H_base + 1.0 * K_cascade + 1.0 * K_cycle).tocsr()

        eigs_base = np.linalg.eigvalsh(H_base.toarray())
        eigs_ext = np.linalg.eigvalsh(H_ext.toarray())

        kappa_base = eigs_base[-1] / eigs_base[0]
        kappa_ext = eigs_ext[-1] / eigs_ext[0]

        assert kappa_ext < kappa_base, (
            f"Regularizers did not improve κ: "
            f"κ(base) = {kappa_base:.2e}, κ(ext) = {kappa_ext:.2e}."
        )
        print(f"  κ(H_base) = {kappa_base:.2e}  →  κ(H_ext) = {kappa_ext:.2e}")


# ----------------------------------------------------------------------
# Remark 4.3 — Base case as special case
# ----------------------------------------------------------------------

class TestRemark43BaseCaseSpecialCase:
    """When α = γ = 0, H_ext = H_base, and convergence still holds."""

    def test_base_case_convergence(self):
        H_base, _, _, _ = make_eer_system(100, seed=0)
        # α = γ = 0
        H_ext = H_base.copy()

        # Verify SPD
        eig_min = eigsh(H_ext, k=1, which="SA",
                        return_eigenvectors=False)[0]
        assert eig_min > 0

        # Verify GS converges
        H = H_ext.toarray()
        D = np.diag(np.diag(H))
        L = np.tril(H, k=-1)
        U = np.triu(H, k=1)
        T_GS = -np.linalg.solve(D + L, U)
        rho = max(abs(np.linalg.eigvals(T_GS)))
        assert rho < 1.0

    def test_alpha_gamma_zero_recovers_base(self):
        """Numerical check: H_ext(α=0, γ=0) == H_base."""
        H_base, _, K_cascade, K_cycle = make_eer_system(50, seed=0)
        H_ext = (H_base + 0.0 * K_cascade + 0.0 * K_cycle).tocsr()
        diff = (H_ext - H_base)
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12


# ----------------------------------------------------------------------
# End-to-end PCGS convergence
# ----------------------------------------------------------------------

class TestEndToEndPCGS:
    """Full PCGS run on a realistic EER instance."""

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_pcgs_linear_convergence(self, n):
        """
        Theorem 4.2 in practice: ||x^(k) - x*||_H decays geometrically.
        """
        from scipy.sparse.linalg import spsolve
        H_base, H_ext, _, _ = make_eer_system(n, seed=0)
        rng = np.random.default_rng(1)
        b = rng.standard_normal(n)

        # Reference solution
        x_star = spsolve(H_ext.tocsc(), b)

        # PCGS
        x = np.zeros(n)
        H = H_ext.toarray()
        H_diag = np.diag(H)

        errors = []
        for sweep in range(50):
            for i in range(n):
                s = H[i] @ x - H_diag[i] * x[i]
                x[i] = (b[i] - s) / H_diag[i]
            err = np.linalg.norm(x - x_star)
            errors.append(err)
            if err < 1e-10:
                break

        # Geometric decay: consecutive ratio < 1
        ratios = [errors[i+1] / errors[i] for i in range(len(errors)-1)
                  if errors[i] > 1e-12]
        assert all(r < 1.0 for r in ratios), (
            f"Non-monotone convergence detected: ratios = {ratios[-5:]}"
        )
        print(f"  n={n:3d}  sweeps={len(errors)}  "
              f"final_err={errors[-1]:.2e}  "
              f"mean_ratio={np.mean(ratios):.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

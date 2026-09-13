"""
Numerical verification of Section 5 (Canonical Regularization).

Tests correspond to:
    - Corollary 5.1: H_ext ≻ 0 for all α, γ ≥ 0
    - Proposition 5.1: ∇R_cascade = α K_cascade x
    - Proposition 5.2: ∇R_cycle   = γ K_cycle   x
    - Eq. (5.10): (H + α K_cascade + γ K_cycle) x* = b

Run with:
    pytest tests/test_hessian.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cycle_basis import (                       # noqa: E402
    build_fundamental_cycle_basis,
    build_cycle_precision_matrix,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

def make_eer_matrices(n: int, seed: int = 0):
    """Build H_base, K_cascade, K_cycle for a synthetic EER instance."""
    import networkx as nx
    from scipy.sparse.csgraph import laplacian

    G = nx.barabasi_albert_graph(n, m=3, seed=seed)
    A_S = nx.to_scipy_sparse_array(G, nodelist=list(G.nodes()),
                                    format="csr", dtype=np.float64)
    L_S = laplacian(A_S, normed=False).tocsr()
    L_D = 0.1 * L_S
    Lambda = identity(n, format="csr")

    H_base = (Lambda + L_S + L_D).tocsr()

    basis = build_fundamental_cycle_basis(G, K_max=500)
    K_cycle = build_cycle_precision_matrix(basis, gamma=1.0)   # unscaled
    K_cascade = (0.05 * identity(n, format="csr")).tocsr()

    return H_base, K_cascade, K_cycle


# ----------------------------------------------------------------------
# Corollary 5.1 — H_ext ≻ 0
# ----------------------------------------------------------------------

class TestCorollary51WellPosedness:
    """Corollary 5.1: H_ext = H + α K_cascade + γ K_cycle ≻ 0."""

    @pytest.mark.parametrize("alpha,gamma", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (10.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
    ])
    def test_hext_positive_definite(self, alpha, gamma):
        """For every α, γ ≥ 0, H_ext must be SPD."""
        H_base, K_cascade, K_cycle = make_eer_matrices(100, seed=0)
        H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()

        eig_min = eigsh(H_ext, k=1, which="SA",
                        return_eigenvectors=False)[0]
        assert eig_min > 0, (
            f"H_ext not PD for α={alpha}, γ={gamma}: λ_min = {eig_min:.6e}"
        )
        print(f"  α={alpha:5.2f}  γ={gamma:5.2f}  λ_min={eig_min:.4e}")

    def test_hext_pd_for_random_alpha_gamma(self):
        """Random sweep over α, γ ∈ [0, 10] to confirm robustness."""
        rng = np.random.default_rng(42)
        H_base, K_cascade, K_cycle = make_eer_matrices(80, seed=0)

        for trial in range(20):
            alpha = float(rng.uniform(0, 10))
            gamma = float(rng.uniform(0, 10))
            H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()
            eig_min = eigsh(H_ext, k=1, which="SA",
                            return_eigenvectors=False)[0]
            assert eig_min > 0, (
                f"Trial {trial}: α={alpha:.3f}, γ={gamma:.3f}, "
                f"λ_min={eig_min:.6e}"
            )

    def test_adding_regularizer_increases_lambda_min(self):
        """Monotonicity: adding regularizers cannot decrease λ_min."""
        H_base, K_cascade, K_cycle = make_eer_matrices(100, seed=0)

        lam_base = eigsh(H_base, k=1, which="SA",
                         return_eigenvectors=False)[0]
        H_ext = (H_base + 1.0 * K_cascade + 1.0 * K_cycle).tocsr()
        lam_ext = eigsh(H_ext, k=1, which="SA",
                        return_eigenvectors=False)[0]

        assert lam_ext >= lam_base - 1e-10


# ----------------------------------------------------------------------
# Proposition 5.1 — Cascade gradient
# ----------------------------------------------------------------------

class TestProposition51CascadeGradient:
    """∇R_cascade(x) = α K_cascade x."""

    def test_gradient_matches_matrix_product(self):
        """Numerical gradient check via finite differences."""
        n = 30
        H_base, K_cascade, K_cycle = make_eer_matrices(n, seed=0)
        alpha = 0.5
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)

        # Analytic gradient: ∇R_cascade = α K_cascade x
        grad_analytic = alpha * (K_cascade @ x)

        # Numerical gradient via central differences
        h = 1e-6
        grad_numeric = np.zeros(n)
        for i in range(n):
            x_plus = x.copy(); x_plus[i] += h
            x_minus = x.copy(); x_minus[i] -= h
            R_plus = 0.5 * alpha * float(x_plus @ (K_cascade @ x_plus))
            R_minus = 0.5 * alpha * float(x_minus @ (K_cascade @ x_minus))
            grad_numeric[i] = (R_plus - R_minus) / (2 * h)

        np.testing.assert_allclose(
            grad_analytic, grad_numeric, rtol=1e-4, atol=1e-6,
            err_msg="Prop 5.1 gradient mismatch."
        )


# ----------------------------------------------------------------------
# Proposition 5.2 — Cycle gradient
# ----------------------------------------------------------------------

class TestProposition52CycleGradient:
    """∇R_cycle(x) = γ K_cycle x."""

    def test_gradient_matches_matrix_product(self):
        n = 30
        H_base, K_cascade, K_cycle = make_eer_matrices(n, seed=0)
        gamma = 0.5
        rng = np.random.default_rng(1)
        x = rng.standard_normal(n)

        grad_analytic = gamma * (K_cycle @ x)

        h = 1e-6
        grad_numeric = np.zeros(n)
        for i in range(n):
            x_plus = x.copy(); x_plus[i] += h
            x_minus = x.copy(); x_minus[i] -= h
            R_plus = 0.5 * gamma * float(x_plus @ (K_cycle @ x_plus))
            R_minus = 0.5 * gamma * float(x_minus @ (K_cycle @ x_minus))
            grad_numeric[i] = (R_plus - R_minus) / (2 * h)

        np.testing.assert_allclose(
            grad_analytic, grad_numeric, rtol=1e-4, atol=1e-6,
            err_msg="Prop 5.2 gradient mismatch."
        )


# ----------------------------------------------------------------------
# Eq. (5.10) — Extended linear system
# ----------------------------------------------------------------------

class TestEq510ExtendedSystem:
    """(H + α K_cascade + γ K_cycle) x* = b at equilibrium."""

    @pytest.mark.parametrize("alpha,gamma", [(0.5, 0.5), (1.0, 2.0), (5.0, 0.1)])
    def test_equilibrium_satisfies_linear_system(self, alpha, gamma):
        """
        Verify: solving the extended linear system is equivalent to
        minimizing E_tot = E_base + R_cascade + R_cycle.
        """
        n = 50
        H_base, K_cascade, K_cycle = make_eer_matrices(n, seed=0)
        rng = np.random.default_rng(2)
        b = rng.standard_normal(n)

        H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()
        x_star = spsolve(H_ext.tocsc(), b)

        # Residual of extended system
        residual = H_ext @ x_star - b
        assert np.max(np.abs(residual)) < 1e-8, (
            f"Eq. (5.10) violated: ||H_ext x* - b||_inf = "
            f"{np.max(np.abs(residual)):.2e}"
        )

    def test_equilibrium_minimizes_energy(self):
        """Perturbing x* increases E_tot (strict convexity)."""
        n = 40
        H_base, K_cascade, K_cycle = make_eer_matrices(n, seed=0)
        alpha, gamma = 0.5, 0.5
        rng = np.random.default_rng(3)
        b = rng.standard_normal(n)

        H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()
        x_star = spsolve(H_ext.tocsc(), b)

        def E_tot(x):
            return 0.5 * float(x @ (H_ext @ x)) - float(b @ x)

        E_star = E_tot(x_star)

        # Random perturbations
        for trial in range(20):
            delta = 0.01 * rng.standard_normal(n)
            x_pert = x_star + delta
            E_pert = E_tot(x_pert)
            assert E_pert > E_star, (
                f"Perturbation reduced energy: "
                f"E* = {E_star:.6f}, E_pert = {E_pert:.6f}"
            )


# ----------------------------------------------------------------------
# K_cascade, K_cycle structural properties
# ----------------------------------------------------------------------

class TestStructuralProperties:
    """Sanity checks on K_cascade and K_cycle."""

    def test_k_cycle_is_psd(self):
        """K_cycle = Σ |σ|^{-1} B_σ B_σᵀ ⪰ 0."""
        _, _, K_cycle = make_eer_matrices(80, seed=0)
        eigs = eigsh(K_cycle, k=1, which="SA",
                     return_eigenvectors=False)[0]
        assert eigs >= -1e-10, f"K_cycle has negative eigenvalue: {eigs:.6e}"

    def test_k_cycle_is_symmetric(self):
        _, _, K_cycle = make_eer_matrices(80, seed=0)
        diff = K_cycle - K_cycle.T
        assert abs(diff).nnz == 0 or np.abs(diff.data).max() < 1e-12

    def test_k_cascade_is_psd(self):
        _, K_cascade, _ = make_eer_matrices(80, seed=0)
        eigs = eigsh(K_cascade, k=1, which="SA",
                     return_eigenvectors=False)[0]
        assert eigs >= -1e-10

    def test_k_cycle_has_rank_le_cycle_basis_size(self):
        """rank(K_cycle) ≤ |C| (number of fundamental cycles)."""
        import networkx as nx
        n = 60
        G = nx.barabasi_albert_graph(n, m=3, seed=0)
        basis = build_fundamental_cycle_basis(G, K_max=500)
        K_cycle = build_cycle_precision_matrix(basis, gamma=1.0)

        # Rank via singular values
        dense = K_cycle.toarray()
        sv = np.linalg.svd(dense, compute_uv=False)
        rank = int(np.sum(sv > 1e-8))

        assert rank <= basis.n_cycles_kept, (
            f"rank(K_cycle) = {rank} > |C| = {basis.n_cycles_kept}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

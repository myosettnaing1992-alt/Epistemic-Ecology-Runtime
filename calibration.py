"""
Hyperparameter calibration for EER.

Implements Algorithm 2 (validation-based surrogate) and Eq. (5.7)
(Type-II maximum likelihood) of the EER Monograph.

IMPORTANT: This module provides TWO estimators:
    (a) `fit_type_ii_ml`   — closed-form fixed-point for Type-II ML
    (b) `fit_validation_grid` — grid-search surrogate (Section 5.4)

Paper explicitly distinguishes (a) from (b); see Section 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import splu


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class CalibrationResult:
    alpha: float
    gamma: float
    loss_history: list[float] = field(default_factory=list)
    method: str = "validation_grid"
    n_evaluations: int = 0


# ----------------------------------------------------------------------
# (a) Type-II Maximum Likelihood (Eq. 5.7 and Appendix B)
# ----------------------------------------------------------------------

def fit_type_ii_ml(
    x_val: np.ndarray,          # (m, n) validation snapshots
    H_base: csr_matrix,         # H from Assumption 2.1
    K_cascade: csr_matrix,      # K_cascade from Eq. (5.4)
    K_cycle: csr_matrix,        # K_cycle from Eq. (5.6)
    b: np.ndarray,              # observation vector
    max_iter: int = 50,
    tol: float = 1e-6,
) -> CalibrationResult:
    """
    Type-II maximum likelihood estimator for (α, γ).

    Solves the fixed-point iteration (Appendix B, Eq. B.2):
        α̂ = tr(K_cascade · H_ext^{-1}) / (xᵀ K_cascade x)
        γ̂ = tr(K_cycle  · H_ext^{-1}) / (xᵀ K_cycle  x)

    where H_ext = H_base + α K_cascade + γ K_cycle.

    This is the *canonical* estimator of Section 5.4. It is only
    appropriate when m (validation set size) is large and H_ext is
    moderate in size. For large n, use `fit_validation_grid` instead.
    """
    if x_val.ndim != 2:
        raise ValueError("x_val must be 2-D of shape (m, n).")
    m, n = x_val.shape
    if H_base.shape != (n, n):
        raise ValueError(f"H_base must be ({n}, {n}).")

    x_mean = x_val.mean(axis=0)

    alpha = 1.0
    gamma = 1.0
    history: list[float] = []

    for it in range(max_iter):
        H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsc()

        # Solve H_ext Y = I  →  we only need the diagonal of the inverse
        try:
            lu = splu(H_ext)
        except RuntimeError as e:
            raise RuntimeError(f"Factorization failed at iter {it}: {e}")

        # Diagonal of H_ext^{-1}: solve one column at a time (n solves)
        # For n up to ~5000 this is fine; larger n should use Hutchinson.
        Hinv_diag = np.empty(n, dtype=np.float64)
        for i in range(n):
            e_i = np.zeros(n, dtype=np.float64)
            e_i[i] = 1.0
            Hinv_diag[i] = lu.solve(e_i)[i]

        # Numerator: tr(K · H_ext^{-1})
        num_alpha = float((K_cascade.diagonal() * Hinv_diag).sum())
        num_gamma = float((K_cycle.diagonal() * Hinv_diag).sum())

        # Denominator: xᵀ K x  (evaluated at the mean snapshot)
        den_alpha = float(x_mean @ (K_cascade @ x_mean))
        den_gamma = float(x_mean @ (K_cycle @ x_mean))

        new_alpha = num_alpha / max(den_alpha, 1e-12)
        new_gamma = num_gamma / max(den_gamma, 1e-12)

        # Clamp to non-negative
        new_alpha = max(new_alpha, 0.0)
        new_gamma = max(new_gamma, 0.0)

        # Marginal log-likelihood at current iterate (Eq. 5.7)
        sign, logdet = np.linalg.slogdet(
            H_ext.toarray() if n <= 512 else _logdet_sparse(H_ext)
        )
        quad = float(x_mean @ (H_ext @ x_mean))
        log_ml = 0.5 * (sign * logdet - quad)   # up to constants

        history.append(log_ml)

        if abs(new_alpha - alpha) < tol and abs(new_gamma - gamma) < tol:
            alpha, gamma = new_alpha, new_gamma
            break

        alpha, gamma = new_alpha, new_gamma

    return CalibrationResult(
        alpha=alpha,
        gamma=gamma,
        loss_history=history,
        method="type_ii_ml",
        n_evaluations=(it + 1) * n,
    )


def _logdet_sparse(H_ext) -> tuple[float, float]:
    """Return (sign, log|det|) for a sparse SPD matrix via Cholesky."""
    from scipy.sparse.linalg import splu
    n = H_ext.shape[0]
    lu = splu(H_ext.tocsc())
    # det = product of diagonal of U
    U_diag = lu.U.diagonal()
    sign = float(np.sign(np.prod(np.sign(U_diag))))
    logabs = float(np.log(np.abs(U_diag)).sum())
    return sign, logabs


# ----------------------------------------------------------------------
# (b) Validation-based grid-search surrogate (Algorithm 2)
# ----------------------------------------------------------------------

def fit_validation_grid(
    x_val: np.ndarray,
    b: np.ndarray,
    H_base: csr_matrix,
    K_cascade: csr_matrix,
    K_cycle: csr_matrix,
    solver_fn: Callable[[csr_matrix, np.ndarray], np.ndarray],
    alpha_grid: np.ndarray = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0]),
    gamma_grid: np.ndarray = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0]),
    verbose: bool = True,
) -> CalibrationResult:
    """
    Validation-based grid search (Algorithm 2, Section 5.4).

    Minimizes the mean squared residual on a held-out set:
        L(α, γ) = (1/m) Σ_s ||x^(s) - x*(α, γ)||²

    where x*(α, γ) solves H_ext(α, γ) x = b.

    NOTE: This is a *surrogate* for Type-II ML. It coincides with the
    fixed-point solution of `fit_type_ii_ml` only in the limit of large m.
    Paper explicitly acknowledges this distinction (Section 5.4).
    """
    if x_val.ndim != 2:
        raise ValueError("x_val must be 2-D of shape (m, n).")
    m, n = x_val.shape

    best_loss = np.inf
    best_alpha = 0.0
    best_gamma = 0.0
    history: list[float] = []
    n_eval = 0

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            H_ext = (H_base + alpha * K_cascade + gamma * K_cycle).tocsr()
            try:
                x_star = solver_fn(H_ext, b)
            except Exception as e:
                if verbose:
                    print(f"    [skip] α={alpha}, γ={gamma}: {e}")
                continue

            # Validation loss: mean squared residual over snapshots
            resid = x_val - x_star[np.newaxis, :]
            loss = float(np.mean(np.sum(resid ** 2, axis=1)))

            history.append(loss)
            n_eval += 1

            if loss < best_loss:
                best_loss = loss
                best_alpha = float(alpha)
                best_gamma = float(gamma)

            if verbose:
                print(f"    α={alpha:.3f}  γ={gamma:.3f}  loss={loss:.6e}")

    return CalibrationResult(
        alpha=best_alpha,
        gamma=best_gamma,
        loss_history=history,
        method="validation_grid",
        n_evaluations=n_eval,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.sparse import random as sparse_random
    from scipy.sparse.linalg import spsolve

    np.random.seed(0)
    n = 100
    m = 20

    # Build a synthetic SPD H_base
    A = sparse_random(n, n, density=0.05, format="csr", random_state=0)
    H_base = (A @ A.T + 5.0 * identity(n)).tocsr()
    K_cascade = (identity(n) * 0.1).tocsr()
    K_cycle = (identity(n) * 0.1).tocsr()
    b = np.random.randn(n)

    x_val = np.random.randn(m, n) * 0.1

    print("Running validation grid search...")
    result = fit_validation_grid(
        x_val, b, H_base, K_cascade, K_cycle,
        solver_fn=lambda H, bb: spsolve(H.tocsc(), bb),
        verbose=False,
    )
    print(f"  Best α = {result.alpha:.4f}")
    print(f"  Best γ = {result.gamma:.4f}")
    print(f"  Evaluations = {result.n_evaluations}")

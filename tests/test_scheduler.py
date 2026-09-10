"""Tests for the Numba hybrid priority scheduler."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from eer.hessian_builder import assemble_extended_hessian
from eer.schedulers import run_hybrid_priority_scheduler_optimized


def _run(H, b, x0, max_sweeps=3000, tol=1e-8):
    Hc = H.tocsc()
    return run_hybrid_priority_scheduler_optimized(
        H.indptr, H.indices, H.data,
        Hc.indptr, Hc.indices, Hc.data,
        b, x0, M=H.shape[0], epsilon=1e-3,
        max_sweeps=max_sweeps, tol=tol,
    )


def test_box_constraint(spd_problem):
    H = assemble_extended_hessian(spd_problem, alpha=0.0, gamma=0.0)
    x0 = np.full(spd_problem.n, 0.5)
    x, _, _ = _run(H, spd_problem.b, x0)
    assert np.all(x >= -1e-12)
    assert np.all(x <= 1.0 + 1e-12)


def test_residual_decreases(spd_problem):
    H = assemble_extended_hessian(spd_problem, alpha=0.0, gamma=0.0)
    x0 = np.full(spd_problem.n, 0.5)
    r0 = np.linalg.norm(H @ x0 - spd_problem.b, np.inf)
    _, _, final_res = _run(H, spd_problem.b, x0)
    assert final_res < r0


def test_converges_to_direct_solve(spd_problem):
    H = assemble_extended_hessian(spd_problem, alpha=0.0, gamma=0.0)
    x0 = np.full(spd_problem.n, 0.5)
    x_hat, _, _ = _run(H, spd_problem.b, x0)
    x_star = spsolve(H.tocsc(), spd_problem.b)
    interior = (x_star > 1e-6) & (x_star < 1 - 1e-6)
    if interior.sum() > 0:
        np.testing.assert_allclose(
            x_hat[interior], x_star[interior], atol=1e-4
        )


def test_determinism(spd_problem):
    H = assemble_extended_hessian(spd_problem, alpha=0.0, gamma=0.0)
    x0 = np.full(spd_problem.n, 0.5)
    x1, _, _ = _run(H, spd_problem.b, x0)
    x2, _, _ = _run(H, spd_problem.b, x0)
    np.testing.assert_array_equal(x1, x2)


def test_returns_finite_residual(spd_problem):
    H = assemble_extended_hessian(spd_problem, alpha=0.0, gamma=0.0)
    x0 = np.full(spd_problem.n, 0.5)
    _, updates, final_res = _run(H, spd_problem.b, x0)
    assert np.isfinite(final_res)
    assert updates > 0

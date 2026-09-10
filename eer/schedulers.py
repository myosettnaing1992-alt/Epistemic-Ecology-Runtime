"""Numba JIT coordinate descent solvers with incremental O(deg) updates."""

from typing import Tuple
import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def numba_coordinate_step(
    x: np.ndarray,
    b: np.ndarray,
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    csr_data: np.ndarray,
    i: int,
) -> Tuple[float, float]:
    """Single projected coordinate update along coordinate i.

    Returns (x_new, delta) where delta = x_new - x_old.
    """
    start, end = csr_indptr[i], csr_indptr[i + 1]
    diag_val = 0.0
    off_diag_sum = 0.0

    for idx in range(start, end):
        j = csr_indices[idx]
        val = csr_data[idx]
        if j == i:
            diag_val = val
        else:
            off_diag_sum += val * x[j]

    if diag_val == 0.0:
        return x[i], 0.0

    x_old = x[i]
    x_new = min(1.0, max(0.0, (b[i] - off_diag_sum) / diag_val))
    return x_new, x_new - x_old


@nb.njit(fastmath=True)
def update_residual_incremental(
    r: np.ndarray,
    delta: float,
    i: int,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
) -> None:
    """Update residual vector r += delta * H[:, i] in O(deg(i)) time using CSC."""
    if delta == 0.0:
        return
    for idx in range(csc_indptr[i], csc_indptr[i + 1]):
        row_j = csc_indices[idx]
        val = csc_data[idx]
        r[row_j] += delta * val


@nb.njit(fastmath=True)
def run_hybrid_priority_scheduler_optimized(
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    csr_data: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    csc_data: np.ndarray,
    b: np.ndarray,
    x_init: np.ndarray,
    M: int,
    epsilon: float,
    max_sweeps: int,
    tol: float,
) -> Tuple[np.ndarray, int, float]:
    """Algorithm 3: Hybrid Priority Scheduler with incremental residuals.

    Combines a mandatory cyclic backbone (every M iterations) with
    aged residual priority updates (eq. 34).
    """
    n = len(b)
    x = x_init.copy()
    ages = np.zeros(n, dtype=np.int64)
    r = np.zeros(n, dtype=np.float64)

    # Initial residual r = Hx - b
    for i in range(n):
        s = 0.0
        for idx in range(csr_indptr[i], csr_indptr[i + 1]):
            s += csr_data[idx] * x[csr_indices[idx]]
        r[i] = s - b[i]

    k = 0
    total_updates = 0
    max_res = 0.0

    while k < max_sweeps:
        if k % M == 0:
            # Mandatory cyclic backbone sweep
            for i in range(n):
                x_new, delta = numba_coordinate_step(
                    x, b, csr_indptr, csr_indices, csr_data, i
                )
                x[i] = x_new
                update_residual_incremental(
                    r, delta, i, csc_indptr, csc_indices, csc_data
                )
                ages[i] = 0
                total_updates += 1
        else:
            # Aged residual priority selection
            max_prio = -1.0
            target_node = 0
            for i in range(n):
                prio = abs(r[i]) + epsilon * float(ages[i])
                if prio > max_prio:
                    max_prio = prio
                    target_node = i

            i = target_node
            x_new, delta = numba_coordinate_step(
                x, b, csr_indptr, csr_indices, csr_data, i
            )
            x[i] = x_new
            update_residual_incremental(
                r, delta, i, csc_indptr, csc_indices, csc_data
            )
            ages[i] = 0
            total_updates += 1

        # Age increment
        for i in range(n):
            ages[i] += 1

        # Stopping criterion: infinity norm of residual (allocation-free)
        max_res = 0.0
        for i in range(n):
            abs_r = abs(r[i])
            if abs_r > max_res:
                max_res = abs_r

        if max_res < tol:
            break
        k += 1

    return x, total_updates, max_res

"""Utility and helper functions for matrix operations and reproducibility."""

import numpy as np
import scipy.sparse as sp


def set_global_seed(seed: int = 42) -> np.random.Generator:
    """Return a seeded NumPy Generator.

    Notes
    -----
    Does not touch the legacy global state (``np.random.seed``).
    """
    return np.random.default_rng(seed)


def make_csc_copy(H_csr: sp.csr_matrix) -> sp.csc_matrix:
    """Convert CSR Hessian to CSC for incremental column updates."""
    return H_csr.tocsc()


def symmetrize(A: sp.spmatrix) -> sp.csr_matrix:
    """Return (A + A^T) / 2 as CSR."""
    return ((A + A.T) * 0.5).tocsr()


def infinity_norm_bound(A: sp.csr_matrix) -> float:
    """Return ||A||_inf (max absolute row sum), an upper bound on rho(A).

    Equivalent to the Gerschgorin disk union bound when all diagonal entries
    share the same sign.
    """
    abs_row_sums = np.asarray(np.abs(A).sum(axis=1)).flatten()
    return float(np.max(abs_row_sums))

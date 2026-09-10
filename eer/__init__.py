"""Epistemic Ecology Runtime (EER).

A high-performance Python/Numba framework for strictly convex variational
belief aggregation on directed epistemic graphs.
"""

from eer.core_graph import EpistemicGraph
from eer.hessian_builder import (
    assemble_extended_hessian,
    build_base_hessian,
    build_scc_hessian_vectorized,
    build_cascade_matrix_bounded,
    build_cycle_matrix_fundamental,
)
from eer.schedulers import (
    run_hybrid_priority_scheduler_optimized,
    numba_coordinate_step,
    update_residual_incremental,
)
from eer.calibration import FastGridSearchCalibrator
from eer.utils import set_global_seed, make_csc_copy, symmetrize, infinity_norm_bound

__version__ = "0.1.0"
__author__ = "Myo Sett Naing"
__orcid__ = "0009-0002-9133-0058"

__all__ = [
    "EpistemicGraph",
    "assemble_extended_hessian",
    "build_base_hessian",
    "build_scc_hessian_vectorized",
    "build_cascade_matrix_bounded",
    "build_cycle_matrix_fundamental",
    "run_hybrid_priority_scheduler_optimized",
    "numba_coordinate_step",
    "update_residual_incremental",
    "FastGridSearchCalibrator",
    "set_global_seed",
    "make_csc_copy",
    "symmetrize",
    "infinity_norm_bound",
    "__version__",
]

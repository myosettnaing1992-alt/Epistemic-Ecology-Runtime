"""
Epistemic Ecology Runtime (EER).

A strictly convex variational solver for belief aggregation on directed
epistemic graphs.
"""

from .core_graph import EpistemicGraph
from .hessian_builder import (
    assemble_extended_hessian,
    build_cascade_matrix_bounded,
)
from .cycle_basis import build_cycle_matrix_fundamental
from .schedulers import (
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
    run_random_priority_scheduler,
)
from .calibration import FastGridSearchCalibrator

__version__ = "0.1.0"

__all__ = [
    "EpistemicGraph",
    "assemble_extended_hessian",
    "build_cascade_matrix_bounded",
    "build_cycle_matrix_fundamental",
    "run_cyclic_gs_scheduler",
    "run_hybrid_priority_scheduler_optimized",
    "run_random_priority_scheduler",
    "FastGridSearchCalibrator",
]

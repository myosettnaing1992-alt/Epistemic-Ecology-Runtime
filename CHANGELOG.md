# Changelog

All notable changes to the **Epistemic Ecology Runtime (EER)** framework
are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-09

### Added
- Core `EpistemicGraph` with support, contradiction, and derivation
  edge-list buffering, plus cached CSR builders and SCC decomposition.
- Vectorized block-COO Hessian assembly: `build_base_hessian` (eq. 2),
  `build_scc_hessian_vectorized` (eq. 8), `assemble_extended_hessian` (eq. 9).
- Structural regularizers:
  - `build_cascade_matrix_bounded` (eq. 15) with DFS backtracking and
    bounded path enumeration.
  - `build_cycle_matrix_fundamental` (eq. 17) with **alternating signed
    incidence** to prevent cancellation on closed loops.
- Numba-JIT hybrid priority coordinate-descent solver with incremental
  O(deg) residual updates (`run_hybrid_priority_scheduler_optimized`).
- `FastGridSearchCalibrator` — parallel grid search over `(alpha, gamma)`
  with pre-cached structural regularizers.
- Test suite: `test_core_graph.py`, `test_cycle.py`, `test_hessian.py`,
  `test_scheduler.py`.
- Benchmark suite with projected CGS baseline, JIT warm-up, non-trivial
  priors, and multi-seed averaging.
- Benchmark auto-formatter: `benchmarks/format_benchmark.py`.
- GitHub Actions CI workflow (Python 3.11 + 3.12).
- Dockerfile and Conda `environment.yml`.

### Fixed
- Cycle matrix sign-cancellation bug: symmetric incidence produced
  `b_sigma = 0` and silently zeroed the cycle regularizer.
- CI workflow input parameter mismatch (`python-python-version` →
  `python-version`).
- `FastGridSearchCalibrator` now uses the `alpha` parameter correctly.

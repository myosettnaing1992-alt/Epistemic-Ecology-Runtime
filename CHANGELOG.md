```markdown
# Changelog

All notable changes to the Epistemic Ecology Runtime will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial public release.
- `eer.core_graph.EpistemicGraph` with decoupled edge-list buffering.
- `eer.hessian_builder.assemble_extended_hessian` with vectorized block-COO assembly.
- `eer.hessian_builder.build_cascade_matrix_bounded` with L_max-bounded DFS.
- `eer.cycle_basis.build_fundamental_cycle_basis` (Kruskal + LCA).
- `eer.cycle_basis.build_cycle_matrix_fundamental` with alternating signed incidence.
- `eer.schedulers.run_cyclic_gs_scheduler`, `run_random_priority_scheduler`, `run_hybrid_priority_scheduler_optimized`.
- `eer.calibration.FastGridSearchCalibrator` with joblib parallelism.
- Comprehensive test suite (52 tests) covering Theorem 4.2, Corollary 4.1, Corollary 5.1, Theorem 9.1.
- `benchmarks/run_benchmarks.py` and `benchmarks/format_benchmark.py`.
- CI workflow (`.github/workflows/test.yml`) with Python 3.11 and 3.12.
- Docker support (`Dockerfile`, `.dockerignore`).
- Conda environment (`environment.yml`).

### Changed
- None.

### Fixed
- Theorem 4.2 proof: corrected the P-regular splitting condition from
  `2D - H ≻ 0` to `sym(D + L) = (H + D)/2 ≻ 0`.
- Corollary 4.1 rate bound: replaced the SOR bound
  `1 - 2/(κ + 1)` with the Gauss-Seidel bound `1 - 1/(2κ)`.

### Deprecated
- None.

### Removed
- None.

### Security
- None.

---

## [0.1.0] — 2026-08-31

Initial release accompanying the research monograph
*Epistemic Ecology Runtime: A Variational Framework for Continuous Belief
Relaxation on Dynamic Epistemic Graphs*.

### Benchmark summary

| Graph         | n    | Method         | Updates | Speedup |
|---------------|------|----------------|---------|---------|
| BA scale-free | 500  | Cyclic GS      | 15,500  | 1.0x    |
| BA scale-free | 500  | Hybrid         | 1,403   | 11.0x   |
| BA scale-free | 10^4 | Cyclic GS      | 8,420   | 1.0x    |
| BA scale-free | 10^4 | Hybrid         | 5,230   | 1.6x    |
| ER random     | 10^4 | Cyclic GS      | 7,890   | 1.0x    |
| ER random     | 10^4 | Hybrid         | 7,950   | 1.0x    |

> The 11x figure applies specifically to BA scale-free graphs at n = 500
> and tolerance 1e-6. Larger graphs and ER graphs show smaller or no
> speedup; see Section 7.3 of the monograph.

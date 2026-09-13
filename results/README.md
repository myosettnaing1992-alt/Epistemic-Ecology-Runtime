# Results

This directory holds JSON output from experiments and benchmarks.

**Generated files are not committed** (see `.gitignore`). Only this
`README.md` and `.gitkeep` are tracked.

## Contents

| File | Committed? | Purpose |
|---|---|---|
| `.gitkeep` | yes | Keep directory in Git |
| `README.md` | yes | This file |
| `benchmark.json` | no | Full benchmark run (main table) |
| `benchmark.md` | no | Formatted markdown summary |
| `smoke.json` | no | Smoke test output |
| `scalability.json` | no | n sweep (Section 7.4) |
| `topology_ablation.json` | no | BA vs ER vs MVE (Section 7.3) |
| `sensitivity.json` | no | (alpha, gamma) grid (Section 7.5) |
| `residual_decay.json` | no | Figure 11.1 data |
| `table_9_1.json` | no | Scheduler comparison (Table 9.1) |

## Reproducing

### Quick smoke run (about 30 seconds)

```bash
python benchmarks/run_benchmarks.py --smoke --out results/smoke.json

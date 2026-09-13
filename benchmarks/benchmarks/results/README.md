# Benchmark Results

This directory holds JSON output from `benchmarks/run_benchmarks.py`.

**Generated files are not committed** (see `.gitignore`) except for
`sample_benchmark.json` and `sample_benchmark.md`, which illustrate the
output schema.

## Files

| File                     | Committed? | Purpose                                    |
|--------------------------|------------|--------------------------------------------|
| `.gitkeep`               | yes        | Keep directory in Git                      |
| `README.md`              | yes        | This file                                  |
| `sample_benchmark.json`  | yes        | Example output (small, schematic)          |
| `sample_benchmark.md`    | yes        | Markdown table rendered from sample        |
| `benchmark.json`         | no         | Actual output from full benchmark run      |
| `benchmark_*.json`       | no         | Timestamped runs                           |

## Reproducing

```bash
# Fast smoke run
python benchmarks/run_benchmarks.py --smoke

# Full run
python benchmarks/run_benchmarks.py \
    --graphs mve ba er \
    --n 32 500 10000 \
    --seeds 0 1 2 \
    --out benchmarks/results/benchmark.json

# Format to markdown
python benchmarks/format_benchmark.py \
    --in benchmarks/results/benchmark.json \
    --stdout

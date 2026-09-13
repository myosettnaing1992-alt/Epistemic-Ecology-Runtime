# 1. Run smoke benchmark
python benchmarks/run_benchmarks.py --smoke \
    --out benchmarks/results/benchmark.json

# 2. Copy as sample
cp benchmarks/results/benchmark.json \
   benchmarks/results/sample_benchmark.json

# 3. Generate markdown
python benchmarks/format_benchmark.py \
    --in benchmarks/results/sample_benchmark.json \
    --stdout > benchmarks/results/sample_benchmark.md

# 4. Verify Git status
git status
# → benchmark.json: ignored
# → sample_benchmark.json: untracked (add it)
# → sample_benchmark.md: untracked (add it)

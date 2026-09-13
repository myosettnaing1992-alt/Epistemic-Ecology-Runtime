"""
Format benchmark JSON into markdown tables for the README.

Reads results/benchmark.json (produced by run_benchmarks.py), aggregates
across seeds, and emits a markdown table between BENCHMARK_TABLE_START
and BENCHMARK_TABLE_END markers in README.md.

Usage:
    python benchmarks/format_benchmark.py
    python benchmarks/format_benchmark.py --in results/benchmark.json
    python benchmarks/format_benchmark.py --stdout
    python benchmarks/format_benchmark.py --update-readme
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    """
    Aggregate records across seeds.

    Returns: dict keyed by (graph, n, method) -> dict with mean/std of
    wallclock and updates.
    """
    buckets: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["graph"], r["n"], r["method"])
        buckets[key].append(r)

    agg = {}
    for key, rs in buckets.items():
        wall = np.array([r["wallclock_sec"] for r in rs])
        upds = np.array([r["coordinate_updates"] for r in rs])
        res = np.array([r["final_residual"] for r in rs])
        agg[key] = {
            "wallclock_mean": float(wall.mean()),
            "wallclock_std": float(wall.std()),
            "updates_mean": float(upds.mean()),
            "updates_std": float(upds.std()),
            "residual_mean": float(res.mean()),
            "n_seeds": len(rs),
        }
    return agg


# ----------------------------------------------------------------------
# Markdown table
# ----------------------------------------------------------------------

METHOD_LABELS = {
    "cyclic_gs": "Cyclic GS",
    "random_priority": "Random priority",
    "hybrid_priority": "Hybrid priority",
}


def to_markdown(agg: dict) -> str:
    """
    Build markdown table with columns:
        Graph | n | Method | Wall-clock (s) | Updates | Speedup
    Speedup is relative to Cyclic GS within each (graph, n) row group.
    """
    keys = sorted(agg.keys(), key=lambda k: (k[0], k[1], k[2]))

    # Group by (graph, n)
    groups: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for (graph, n, method), stats in agg.items():
        groups[(graph, n)][method] = stats

    lines = []
    lines.append("| Graph | n | Method | Wall-clock (s) | Updates | Speedup |")
    lines.append("|---|---|---|---|---|---|")

    for (graph, n), methods in sorted(groups.items()):
        baseline = methods.get("cyclic_gs")
        if baseline is None:
            continue
        base_wall = baseline["wallclock_mean"]

        for method in ["cyclic_gs", "random_priority", "hybrid_priority"]:
            if method not in methods:
                continue
            s = methods[method]
            wall = s["wallclock_mean"]
            wall_std = s["wallclock_std"]
            upd = int(s["updates_mean"])
            speedup = base_wall / wall if wall > 0 else 0.0

            lines.append(
                f"| {graph.upper()} | {n} | {METHOD_LABELS[method]} "
                f"| {wall:.3f} ± {wall_std:.3f} "
                f"| {upd:,} "
                f"| {speedup:.2f}x |"
            )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# README integration
# ----------------------------------------------------------------------

START = "<!-- BENCHMARK_TABLE_START -->"
END = "<!-- BENCHMARK_TABLE_END -->"


def update_readme(readme_path: Path, table: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise RuntimeError(
            f"README markers not found. Add '{START}' and '{END}'."
        )
    pattern = re.compile(
        re.escape(START) + r".*?" + re.escape(END), re.DOTALL
    )
    replacement = f"{START}\n{table}\n{END}"
    new_text = pattern.sub(replacement, text)
    readme_path.write_text(new_text, encoding="utf-8")
    print(f"Updated {readme_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=str,
                   default="results/benchmark.json")
    p.add_argument("--readme", type=str, default="README.md")
    p.add_argument("--stdout", action="store_true",
                   help="Print table to stdout only.")
    p.add_argument("--update-readme", action="store_true",
                   help="Rewrite README between markers.")
    return p.parse_args()


def main():
    args = parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        print(f"[error] {in_path} not found. Run run_benchmarks.py first.",
              file=sys.stderr)
        sys.exit(1)

    payload = json.loads(in_path.read_text())
    records = payload.get("records", [])
    if not records:
        print("[error] No records in JSON.", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(records)
    table = to_markdown(agg)

    if args.stdout or not args.update_readme:
        print(table)

    if args.update_readme:
        readme_path = Path(args.readme)
        update_readme(readme_path, table)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Format benchmark CSV into Markdown and inject into README.md.

Reads benchmarks/results/benchmark_summary.csv (schema produced by
run_benchmarks.py) and replaces the block between:

    <!-- BENCHMARK_TABLE_START -->
    <!-- BENCHMARK_TABLE_END -->

Usage:
    python benchmarks/format_benchmark.py
    python benchmarks/format_benchmark.py --print-only
"""

import argparse
from pathlib import Path

import pandas as pd


TABLE_START = "<!-- BENCHMARK_TABLE_START -->"
TABLE_END = "<!-- BENCHMARK_TABLE_END -->"


def format_summary(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV and reshape it into a README-friendly DataFrame."""
    raw = pd.read_csv(csv_path)

    # Match the schema written by run_benchmarks.py
    required = {
        "Nodes", "CGS_Sweeps", "EER_Updates",
        "CGS_Time_s", "CGS_Time_std",
        "EER_Time_s", "EER_Time_std", "Speedup",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"CSV {csv_path} missing columns: {sorted(missing)}. "
            f"Re-run `python benchmarks/run_benchmarks.py`."
        )

    return pd.DataFrame({
        "Nodes ($n$)": raw["Nodes"].apply(lambda x: f"{int(x):,}"),
        "CGS Sweeps": raw["CGS_Sweeps"].astype(int),
        "EER Updates": raw["EER_Updates"].astype(int),
        "CGS Time (s)": raw.apply(
            lambda r: f"{r['CGS_Time_s']:.3f} ± {r['CGS_Time_std']:.3f}",
            axis=1,
        ),
        "EER Time (s)": raw.apply(
            lambda r: f"{r['EER_Time_s']:.3f} ± {r['EER_Time_std']:.3f}",
            axis=1,
        ),
        "Speedup": raw["Speedup"].apply(lambda x: f"**{x:.2f}×**"),
    })


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join([":---:"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def inject_into_readme(readme_path: Path, table_md: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if TABLE_START not in text or TABLE_END not in text:
        raise RuntimeError(
            f"Markers not found in {readme_path}. "
            f"Add '{TABLE_START}' and '{TABLE_END}' around the benchmark table."
        )
    pre, rest = text.split(TABLE_START, 1)
    _, post = rest.split(TABLE_END, 1)
    new_block = f"{TABLE_START}\n{table_md}\n{TABLE_END}"
    readme_path.write_text(pre + new_block + post, encoding="utf-8")
    print(f"[OK] Injected benchmark table into {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="benchmarks/results/benchmark_summary.csv",
        type=Path,
    )
    parser.add_argument("--readme", default="README.md", type=Path)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"{args.csv} not found. "
            f"Run `python benchmarks/run_benchmarks.py` first."
        )

    table_df = format_summary(args.csv)
    table_md = df_to_markdown(table_df)

    print("\n--- Generated Markdown Table ---")
    print(table_md)
    print("--------------------------------\n")

    if args.print_only:
        return
    inject_into_readme(args.readme, table_md)


if __name__ == "__main__":
    main()

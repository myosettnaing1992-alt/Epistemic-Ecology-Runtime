"""
Hyperparameter sensitivity for Section 7.5.

Sweeps over:

    - Aging rate        ε ∈ {1e-5, 1e-4, 1e-3, 1e-2}
    - Backbone period   M ∈ {10, 50, 100, 500}

and reports total coordinate updates to convergence on BA graphs.

Usage:
    python experiments/run_sensitivity.py
    python experiments/run_sensitivity.py --n 500
    python experiments/run_sensitivity.py --out results/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eer import (                       # noqa: E402
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
    run_hybrid_priority_scheduler_optimized,
)
from eer.utils import make_ba_graph, set_random_priors   # noqa: E402


@dataclass
class SensitivityRecord:
    n: int
    epsilon: float
    M: int
    seed: int
    wallclock_sec: float
    coordinate_updates: int
    final_residual: float
    converged: bool


def _make_system(n: int, seed: int = 0):
    G_nx = make_ba_graph(n, m=3, seed=seed)
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc, L_max=3)
    return g, H


def run_one(
    n: int, epsilon: float, M: int, seed: int = 0,
    tol: float = 1e-6, verbose: bool = False,
) -> SensitivityRecord:
    g, H = _make_system(n, seed=seed)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    x0 = np.full(n, 0.5, dtype=np.float64)

    if verbose:
        print(f"  n={n} eps={epsilon:.1e} M={M} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    _, upd, res = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        g.b, x0, M=M, epsilon=epsilon, tol=tol, max_sweeps=200_000,
    )
    t1 = time.perf_counter()
    if verbose:
        print(f"{t1 - t0:6.3f}s upd={upd:,}")

    return SensitivityRecord(
        n=n, epsilon=float(epsilon), M=int(M), seed=seed,
        wallclock_sec=t1 - t0,
        coordinate_updates=int(upd),
        final_residual=float(res),
        converged=res < tol,
    )


def to_markdown(records: list[SensitivityRecord], n: int) -> str:
    """Render as an (ε, M) heatmap table."""
    epsilons = sorted({r.epsilon for r in records})
    Ms = sorted({r.M for r in records})

    lines = [f"### Sensitivity to (ε, M) — BA, n = {n}", ""]
    header = "| ε \\ M | " + " | ".join(str(m) for m in Ms) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(Ms) + 1))

    for eps in epsilons:
        row = [f"| {eps:.0e} "]
        for M in Ms:
            match = [r for r in records if r.epsilon == eps and r.M == M]
            if match:
                row.append(f"| {match[0].coordinate_updates:,} ")
            else:
                row.append("| — ")
        row.append("|")
        lines.append("".join(row))

    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epsilons", nargs="+", type=float,
                   default=[1e-5, 1e-4, 1e-3, 1e-2])
    p.add_argument("--periods", nargs="+", type=int,
                   default=[10, 50, 100, 500])
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--out", type=str, default="results/sensitivity.json")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    print(f"=== Hyperparameter sensitivity (BA, n={args.n}) ===")
    records: list[SensitivityRecord] = []
    for eps in args.epsilons:
        for M in args.periods:
            for seed in args.seeds:
                try:
                    r = run_one(args.n, eps, M, seed=seed,
                                tol=args.tol, verbose=verbose)
                    records.append(r)
                except Exception as e:
                    print(f"  [FAIL] eps={eps} M={M} seed={seed}: {e}",
                          file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "python": sys.version,
                "platform": platform.platform(),
                "n": args.n,
                "epsilons": args.epsilons,
                "periods": args.periods,
                "seeds": args.seeds,
                "tol": args.tol,
            },
            "records": [asdict(r) for r in records],
        }, f, indent=2)
    print(f"\nSaved → {out_path}")

    md = to_markdown(records, args.n)
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved → {md_path}\n")
    print(md)


if __name__ == "__main__":
    main()

"""
Topology ablation for Section 7.3.

Compares the hybrid scheduler's speedup across three graph families:

    - MVE chain        (n = 32)
    - BA scale-free    (n = 500, 10^4)
    - ER random        (n = 500, 10^4)

For each (graph, n), reports:
    - Wall-clock time (s)
    - Coordinate updates
    - Update-count speedup vs. CGS
    - Wall-clock speedup vs. CGS

Also computes the residual mass distribution across degree classes,
explaining why the aging mechanism is topology-dependent.

Usage:
    python experiments/run_topology_ablation.py
    python experiments/run_topology_ablation.py --out results/topology_ablation.json
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
    run_cyclic_gs_scheduler,
    run_hybrid_priority_scheduler_optimized,
)
from eer.utils import (                 # noqa: E402
    make_ba_graph,
    make_er_graph,
    set_random_priors,
)


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------

@dataclass
class AblationRecord:
    graph: str
    n: int
    m_S: int
    method: str
    wallclock_sec: float
    coordinate_updates: int
    final_residual: float
    converged: bool
    seed: int


@dataclass
class ResidualDistribution:
    graph: str
    n: int
    degree_bins: list[int]
    residual_mass_per_bin: list[float]
    degree_counts: list[int]
    gini_coefficient: float


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_graph(kind: str, n: int, seed: int = 0) -> EpistemicGraph:
    if kind == "mve":
        import networkx as nx
        G_nx = nx.path_graph(min(n, 32))
    elif kind == "ba":
        G_nx = make_ba_graph(n, m=3, seed=seed)
    elif kind == "er":
        G_nx = make_er_graph(n, avg_deg=5.0, seed=seed)
    else:
        raise ValueError(kind)

    g = EpistemicGraph(num_nodes=G_nx.number_of_nodes())
    set_random_priors(g, seed=seed)
    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)
    return g


def _assemble(g: EpistemicGraph):
    Q_cyc = build_cycle_matrix_fundamental(g, K_max=1000)
    H = assemble_extended_hessian(g, gamma=0.05, Q_cycle=Q_cyc, L_max=3)
    return H


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array."""
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


# ----------------------------------------------------------------------
# Residual distribution analysis
# ----------------------------------------------------------------------

def compute_residual_distribution(
    g: EpistemicGraph,
    H,
    x_star: np.ndarray,
    n_bins: int = 10,
) -> ResidualDistribution:
    """
    Bin nodes by degree; sum |residual| within each bin.

    High concentration (large Gini) indicates that the residual mass
    concentrates on a few hubs, which is what the aging mechanism exploits.
    """
    import networkx as nx
    A = g.support_adjacency()
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(int)

    # Residual at equilibrium
    r = H @ x_star - g.b
    abs_r = np.abs(r)

    # Bin by degree
    d_min, d_max = int(degrees.min()), int(degrees.max())
    if d_max == d_min:
        bins = np.array([d_min, d_min + 1])
    else:
        bins = np.linspace(d_min, d_max + 1, n_bins + 1).astype(int)

    bin_idx = np.clip(np.digitize(degrees, bins) - 1, 0, len(bins) - 2)
    mass = np.zeros(len(bins) - 1)
    counts = np.zeros(len(bins) - 1, dtype=int)
    for k in range(len(bins) - 1):
        mask = bin_idx == k
        mass[k] = abs_r[mask].sum()
        counts[k] = int(mask.sum())

    return ResidualDistribution(
        graph="", n=g.num_nodes,
        degree_bins=[int(b) for b in bins],
        residual_mass_per_bin=[float(m) for m in mass],
        degree_counts=[int(c) for c in counts],
        gini_coefficient=_gini(abs_r),
    )


# ----------------------------------------------------------------------
# Main ablation
# ----------------------------------------------------------------------

def run_one(kind: str, n: int, seed: int = 0, tol: float = 1e-6, verbose: bool = True):
    g = _make_graph(kind, n, seed=seed)
    H = _assemble(g)
    H_csr = H.tocsr()
    H_csc = H.tocsc()
    b = g.b
    x0 = np.full(g.num_nodes, 0.5)

    m_S = int(len(g._src))
    records: list[AblationRecord] = []

    # --- CGS ---
    if verbose:
        print(f"  [CGS] {kind:>3} n={n:>6} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    x_cgs, upd, res = run_cyclic_gs_scheduler(H, b, x0, tol=tol, max_sweeps=100_000)
    t1 = time.perf_counter()
    records.append(AblationRecord(
        graph=kind, n=n, m_S=m_S, method="cyclic_gs",
        wallclock_sec=t1 - t0, coordinate_updates=int(upd),
        final_residual=float(res), converged=res < tol, seed=seed,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  upd={upd:>8,}  res={res:.2e}")

    # --- Hybrid ---
    if verbose:
        print(f"  [HYP] {kind:>3} n={n:>6} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    x_hyb, upd_h, res_h = run_hybrid_priority_scheduler_optimized(
        H_csr.indptr, H_csr.indices, H_csr.data,
        H_csc.indptr, H_csc.indices, H_csc.data,
        b, x0, M=g.num_nodes, epsilon=1e-3, tol=tol, max_sweeps=100_000,
    )
    t1 = time.perf_counter()
    records.append(AblationRecord(
        graph=kind, n=n, m_S=m_S, method="hybrid_priority",
        wallclock_sec=t1 - t0, coordinate_updates=int(upd_h),
        final_residual=float(res_h), converged=res_h < tol, seed=seed,
    ))
    if verbose:
        print(f"{t1 - t0:7.3f}s  upd={upd_h:>8,}  res={res_h:.2e}")

    # Residual distribution (using hybrid equilibrium)
    dist = compute_residual_distribution(g, H, x_hyb)
    dist.graph = kind
    dist.n = n

    return records, dist


def run_all(
    configs: list[tuple[str, int]],
    seeds: list[int],
    verbose: bool = True,
) -> tuple[list[AblationRecord], list[ResidualDistribution]]:
    all_records: list[AblationRecord] = []
    all_dists: list[ResidualDistribution] = []

    for kind, n in configs:
        for seed in seeds:
            if verbose:
                print(f"\n=== {kind.upper()}  n={n:,}  seed={seed} ===")
            try:
                recs, dist = run_one(kind, n, seed=seed, verbose=verbose)
                all_records.extend(recs)
                all_dists.append(dist)
            except Exception as e:
                print(f"  [FAIL] {kind} n={n} seed={seed}: {e}", file=sys.stderr)
    return all_records, all_dists


# ----------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------

def to_markdown(records: list[AblationRecord], dists: list[ResidualDistribution]) -> str:
    from collections import defaultdict

    groups: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        groups[(r.graph, r.n)][r.method] = r

    lines = ["## Topology Ablation (Section 7.3)", ""]
    lines.append("| Graph | n | Method | Wall-clock (s) | Updates | Wall speedup | Update speedup |")
    lines.append("|---|---|---|---|---|---|---|")
    for (kind, n), m in sorted(groups.items()):
        base = m.get("cyclic_gs")
        if base is None:
            continue
        for method in ["cyclic_gs", "hybrid_priority"]:
            if method not in m:
                continue
            r = m[method]
            wall_sp = base.wallclock_sec / r.wallclock_sec if r.wallclock_sec > 0 else 0.0
            upd_sp = base.coordinate_updates / r.coordinate_updates if r.coordinate_updates > 0 else 0.0
            lines.append(
                f"| {kind.upper()} | {n:,} | {method} "
                f"| {r.wallclock_sec:.3f} "
                f"| {r.coordinate_updates:,} "
                f"| {wall_sp:.2f}x "
                f"| {upd_sp:.2f}x |"
            )

    lines.append("")
    lines.append("## Residual Distribution (Gini Coefficient)")
    lines.append("")
    lines.append("| Graph | n | Gini |")
    lines.append("|---|---|---|")
    for d in dists:
        lines.append(f"| {d.graph.upper()} | {d.n:,} | {d.gini_coefficient:.4f} |")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results/topology_ablation.json")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--quick", action="store_true",
                   help="Quick run: skip n=10^4.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    configs = [("mve", 32), ("ba", 500), ("er", 500)]
    if not args.quick:
        configs += [("ba", 10_000), ("er", 10_000)]

    records, dists = run_all(configs, args.seeds, verbose=not args.quiet)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "seeds": args.seeds,
        },
        "records": [asdict(r) for r in records],
        "residual_distribution": [asdict(d) for d in dists],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Markdown
    md = to_markdown(records, dists)
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved → {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()

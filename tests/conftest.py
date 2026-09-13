"""
Shared pytest fixtures for the EER test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eer import (                       # noqa: E402
    EpistemicGraph,
    assemble_extended_hessian,
    build_cycle_matrix_fundamental,
)
from eer.utils import (                 # noqa: E402
    make_ba_graph,
    make_er_graph,
    set_random_priors,
)


# ----------------------------------------------------------------------
# Session-level config
# ----------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration (multi-module)"
    )


# ----------------------------------------------------------------------
# Graph fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def small_ba_graph():
    """BA graph with n=50, seed=0."""
    return make_ba_graph(50, m=3, seed=0)


@pytest.fixture
def medium_ba_graph():
    """BA graph with n=200, seed=0."""
    return make_ba_graph(200, m=3, seed=0)


@pytest.fixture
def small_er_graph():
    """ER graph with n=50, avg_deg=5, seed=0."""
    return make_er_graph(50, avg_deg=5.0, seed=0)


@pytest.fixture
def chain_graph():
    """Chain graph with n=20."""
    import networkx as nx
    return nx.path_graph(20)


# ----------------------------------------------------------------------
# EpistemicGraph fixtures
# ----------------------------------------------------------------------

def _make_eer_graph_from_nx(G_nx, seed: int = 0, with_derived: bool = True):
    """Helper: convert a networkx graph to an EpistemicGraph."""
    n = G_nx.number_of_nodes()
    g = EpistemicGraph(num_nodes=n)
    set_random_priors(g, seed=seed)

    for u, v in G_nx.edges():
        g.add_support_edge(int(u), int(v), 1.0)

    if with_derived:
        # Add 10% of support edges as derived-from (directed)
        rng = np.random.default_rng(seed)
        edges = list(G_nx.edges())
        n_derived = max(1, len(edges) // 10)
        for u, v in rng.choice(len(edges), size=n_derived, replace=False):
            uu, vv = edges[u]
            g.add_derived_from_edge(int(uu), int(vv), 0.5)

    return g


@pytest.fixture
def eer_small_ba():
    return _make_eer_graph_from_nx(make_ba_graph(50, m=3, seed=0), seed=0)


@pytest.fixture
def eer_medium_ba():
    return _make_eer_graph_from_nx(make_ba_graph(200, m=3, seed=0), seed=0)


@pytest.fixture
def eer_small_er():
    return _make_eer_graph_from_nx(
        make_er_graph(50, avg_deg=5.0, seed=0), seed=0
    )


@pytest.fixture
def eer_chain():
    import networkx as nx
    return _make_eer_graph_from_nx(nx.path_graph(20), seed=0)


# ----------------------------------------------------------------------
# Hessian fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def hessian_small_ba(eer_small_ba):
    """H_ext for small BA graph with alpha=0.5, gamma=0.5."""
    Q_cyc = build_cycle_matrix_fundamental(eer_small_ba, K_max=200)
    H = assemble_extended_hessian(
        eer_small_ba, alpha=0.5, gamma=0.5, Q_cycle=Q_cyc, L_max=3,
    )
    return H


@pytest.fixture
def hessian_medium_ba(eer_medium_ba):
    """H_ext for medium BA graph with alpha=0.5, gamma=0.5."""
    Q_cyc = build_cycle_matrix_fundamental(eer_medium_ba, K_max=500)
    H = assemble_extended_hessian(
        eer_medium_ba, alpha=0.5, gamma=0.5, Q_cycle=Q_cyc, L_max=3,
    )
    return H


# ----------------------------------------------------------------------
# RHS fixture
# ----------------------------------------------------------------------

@pytest.fixture
def rhs_vector(eer_small_ba):
    """Random RHS b for the small EER system."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(eer_small_ba.num_nodes)


# ----------------------------------------------------------------------
# Tolerance helpers
# ----------------------------------------------------------------------

@pytest.fixture
def numerical_tol():
    return 1e-9


@pytest.fixture
def eigen_tol():
    return 1e-8

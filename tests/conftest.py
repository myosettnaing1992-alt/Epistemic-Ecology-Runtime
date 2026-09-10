"""Shared pytest fixtures for the EER test suite."""

import numpy as np
import pytest

from eer.core_graph import EpistemicGraph


@pytest.fixture
def four_cycle():
    """4-node closed support cycle: 0-1-2-3-0."""
    eg = EpistemicGraph(4)
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        eg.add_support_edge(u, v, w=1.0)
    return eg


@pytest.fixture
def connected_tree():
    """4-node connected tree: 0-1-2-3."""
    eg = EpistemicGraph(4)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        eg.add_support_edge(u, v, w=1.0)
    return eg


@pytest.fixture
def triangle_with_tail():
    """Triangle 0-1-2-0 plus tail 2-3."""
    eg = EpistemicGraph(4)
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        eg.add_support_edge(u, v, w=1.0)
    return eg


@pytest.fixture
def random_graph():
    """Random sparse epistemic graph with fixed seed."""
    rng = np.random.default_rng(42)
    n = 30
    g = EpistemicGraph(n)
    for _ in range(60):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            g.add_support_edge(int(u), int(v), float(rng.uniform(0.5, 1.5)))
    for _ in range(20):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            g.add_contradiction_edge(int(u), int(v), float(rng.uniform(0.3, 1.0)))
    for _ in range(15):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            g.add_derivation_edge(int(u), int(v), float(rng.uniform(0.5, 1.0)))
    g.b = rng.uniform(0.0, 1.0, size=n)
    g.lambda_vec = rng.uniform(0.5, 2.0, size=n)
    return g


@pytest.fixture
def spd_problem():
    """Small SPD problem for scheduler tests."""
    rng = np.random.default_rng(7)
    n = 40
    g = EpistemicGraph(n)
    for _ in range(80):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            g.add_support_edge(int(u), int(v), 1.0)
    g.b = rng.uniform(0.2, 0.8, size=n)
    g.lambda_vec = np.full(n, 1.0)
    return g

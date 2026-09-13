# Data

This directory holds graph data for benchmarks and experiments.

**Large graph files are not committed.** Generate them on demand:

```python
import networkx as nx
G = nx.barabasi_albert_graph(10_000, m=3, seed=0)
nx.write_graphml(G, "data/ba_10k.graphml")

"""Community-detection wrappers. All return a list of disjoint node sets covering the graph."""
from __future__ import annotations

from typing import Callable

import networkx as nx

from .modularity import modularity_fixed_partition

METHODS = ("louvain", "greedy_modularity", "label_propagation", "girvan_newman", "leiden")


def _singletons(G: nx.Graph) -> list[set[int]]:
    return [{v} for v in G.nodes()]


def louvain(G: nx.Graph, seed: int | None = None, resolution: float = 1.0) -> list[set[int]]:
    """Louvain (Blondel et al., 2008) via NetworkX."""
    if G.number_of_edges() == 0:
        return _singletons(G)
    return [set(c) for c in nx.community.louvain_communities(G, seed=seed, resolution=resolution)]


def greedy_modularity(G: nx.Graph, seed: int | None = None) -> list[set[int]]:
    """Clauset-Newman-Moore greedy modularity maximisation (deterministic; ``seed`` ignored)."""
    if G.number_of_edges() == 0:
        return _singletons(G)
    return [set(c) for c in nx.community.greedy_modularity_communities(G)]


def label_propagation(G: nx.Graph, seed: int | None = None) -> list[set[int]]:
    """Asynchronous label propagation (stochastic)."""
    if G.number_of_edges() == 0:
        return _singletons(G)
    return [set(c) for c in nx.community.asyn_lpa_communities(G, seed=seed)]


def girvan_newman(G: nx.Graph, seed: int | None = None, max_levels: int | None = None) -> list[set[int]]:
    """Girvan-Newman: among the dendrogram levels, return the one with the highest modularity.

    Deterministic (``seed`` ignored). Cost grows quickly with edge count, so use sparingly on dense
    (noisy) graphs; ``max_levels`` caps the number of dendrogram levels examined.
    """
    if G.number_of_edges() == 0:
        return _singletons(G)
    best, best_q = None, -float("inf")
    for level, comms in enumerate(nx.community.girvan_newman(G)):
        part = [set(c) for c in comms]
        q = modularity_fixed_partition(G, part)
        if q > best_q:
            best, best_q = part, q
        if max_levels is not None and level + 1 >= max_levels:
            break
    return best if best is not None else _singletons(G)


def leiden(G: nx.Graph, seed: int | None = None) -> list[set[int]]:
    """Leiden (Traag et al., 2019); requires the optional ``python-igraph`` and ``leidenalg``."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Leiden needs `pip install python-igraph leidenalg`.") from exc
    if G.number_of_edges() == 0:
        return _singletons(G)
    nodes = sorted(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in G.edges()])
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return [{nodes[i] for i in block} for block in part]


_REGISTRY: dict[str, Callable[..., list[set[int]]]] = {
    "louvain": louvain, "greedy_modularity": greedy_modularity,
    "label_propagation": label_propagation, "girvan_newman": girvan_newman, "leiden": leiden,
}


def leiden_available() -> bool:
    """True if the optional Leiden dependencies can be imported."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
        return True
    except ImportError:
        return False


def detect_communities(G: nx.Graph, method: str = "louvain", seed: int | None = None, **kwargs
                       ) -> list[set[int]]:
    """Dispatch to a detection algorithm by name (see :data:`METHODS`)."""
    if method not in _REGISTRY:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    return _REGISTRY[method](G, seed=seed, **kwargs)

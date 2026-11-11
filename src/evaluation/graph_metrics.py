"""Descriptive graph statistics used to compare an original graph with a release."""
from __future__ import annotations

import networkx as nx
import numpy as np


def degree_sequence(G: nx.Graph) -> np.ndarray:
    return np.array(sorted((d for _, d in G.degree()), reverse=True), dtype=int)


def degree_histogram_normalized(G: nx.Graph, max_degree: int | None = None) -> np.ndarray:
    """Empirical degree distribution P(k) for k = 0..max_degree."""
    degs = degree_sequence(G)
    top = int(max_degree if max_degree is not None else (degs.max() if degs.size else 0))
    hist = np.bincount(np.clip(degs, 0, top), minlength=top + 1).astype(float)
    return hist / max(hist.sum(), 1.0)


def largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def average_path_length(G: nx.Graph) -> float:
    """Average shortest-path length on the largest connected component (nan if it has < 2 nodes).
    Defined per component; the LCC fraction should be reported alongside it."""
    lcc = largest_component(G)
    return float(nx.average_shortest_path_length(lcc)) if lcc.number_of_nodes() > 1 else float("nan")


def graph_summary(G: nx.Graph) -> dict[str, float]:
    """Density, degree, clustering, components and path-length summary of ``G``."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = list(nx.connected_components(G)) if n else []
    lcc = largest_component(G)
    return {
        "n_nodes": n, "n_edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "mean_degree": 2 * m / n if n else 0.0,
        "max_degree": int(degree_sequence(G).max()) if n else 0,
        "avg_clustering": float(nx.average_clustering(G)) if n else 0.0,
        "transitivity": float(nx.transitivity(G)) if n else 0.0,
        "n_components": len(comps),
        "lcc_fraction": lcc.number_of_nodes() / n if n else 0.0,
        "avg_path_length_lcc": average_path_length(G),
        "diameter_lcc": int(nx.diameter(lcc)) if lcc.number_of_nodes() > 1 else 0,
        "assortativity": _safe_assortativity(G),
    }


def _safe_assortativity(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return float("nan")
    with np.errstate(all="ignore"):
        val = nx.degree_assortativity_coefficient(G)
    return float(val) if np.isfinite(val) else float("nan")

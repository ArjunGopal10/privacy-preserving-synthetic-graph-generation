"""Distances between structural properties of an original graph and a release (lower = closer)."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .graph_metrics import degree_sequence, graph_summary


def degree_ks_distance(G: nx.Graph, H: nx.Graph) -> float:
    """Kolmogorov-Smirnov statistic between the degree samples of G and H (statistic only; the
    asymptotic method avoids exact-p-value warnings with tied integer degrees)."""
    return float(ks_2samp(degree_sequence(G), degree_sequence(H), method="asymp").statistic)


def degree_wasserstein(G: nx.Graph, H: nx.Graph) -> float:
    """1-Wasserstein (earth mover's) distance between degree samples."""
    return float(wasserstein_distance(degree_sequence(G), degree_sequence(H)))


def structural_distances(G: nx.Graph, H: nx.Graph) -> dict[str, float]:
    """Absolute differences in key statistics plus degree-distribution distances."""
    sg, sh = graph_summary(G), graph_summary(H)
    return {
        "degree_ks": degree_ks_distance(G, H),
        "degree_wasserstein": degree_wasserstein(G, H),
        "abs_density_diff": abs(sg["density"] - sh["density"]),
        "abs_clustering_diff": abs(sg["avg_clustering"] - sh["avg_clustering"]),
        "abs_path_length_diff": float(abs(sg["avg_path_length_lcc"] - sh["avg_path_length_lcc"])),
        "abs_components_diff": abs(sg["n_components"] - sh["n_components"]),
    }

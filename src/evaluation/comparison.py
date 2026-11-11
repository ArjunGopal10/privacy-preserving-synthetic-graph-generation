"""Compare an original graph with a generated graph across privacy, structure and community axes."""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.community.community_metrics import fixed_partition_report, redetected_report
from src.graph_generation.graph_preprocessing import graph_to_pair_vector
from src.privacy.privacy_metrics import all_edge_metrics
from .graph_metrics import graph_summary
from .structural_metrics import structural_distances


def compare_graphs(G: nx.Graph, H: nx.Graph, P0: list[set[int]], detection_seed: int | None = 0,
                   detect_method: str = "louvain") -> dict[str, float]:
    """One flat record of edge indicators, structure statistics/distances and community metrics.

    Community metrics use both protocols (fixed P0, and ``detect_method`` re-detected on H);
    the ``sizes`` list is dropped from the flat record.
    """
    rec: dict[str, float] = {}
    rec.update(all_edge_metrics(graph_to_pair_vector(G), graph_to_pair_vector(H)))
    rec.update({f"stat_{k}": v for k, v in graph_summary(H).items()})
    rec.update(structural_distances(G, H))
    rec.update(fixed_partition_report(H, P0))
    red = redetected_report(H, P0, detect_method, seed=detection_seed)
    red.pop("sizes", None)
    rec.update(red)
    return rec

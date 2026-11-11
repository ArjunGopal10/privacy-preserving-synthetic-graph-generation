"""Community-structure metrics and the two evaluation protocols.

Protocol A - FIXED reference partition P0
    P0 is computed once on the original graph. The released graph H is scored *under P0*
    (modularity, coverage, performance, conductance, intra/inter edge ratio). NMI/ARI are not
    meaningful here (comparing P0 with itself gives 1).

Protocol B - RE-DETECTED partition
    A detection algorithm is run on H giving P_H; P_H is compared with P0 (NMI, ARI, assignment
    consistency) and its own properties are reported (modularity of P_H on H, community count,
    sizes). This measures whether community structure is *recoverable* from the release.

The two protocols answer different questions and are reported in separate tables/columns.
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .community_detection import detect_communities
from .modularity import labels_from_partition, modularity_fixed_partition


def nmi(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Normalized Mutual Information in [0, 1] (arithmetic normalisation). Higher = more similar."""
    return float(normalized_mutual_info_score(labels_a, labels_b))


def ari(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    """Adjusted Rand Index (1 = identical, ~0 = chance, can be negative)."""
    return float(adjusted_rand_score(labels_a, labels_b))


def num_communities(partition: Sequence[set[int]]) -> int:
    return sum(1 for b in partition if len(b) > 0)


def community_sizes(partition: Sequence[set[int]]) -> list[int]:
    """Community sizes in descending order."""
    return sorted((len(b) for b in partition if len(b) > 0), reverse=True)


def _edge_counts(G: nx.Graph, labels: np.ndarray) -> tuple[int, int]:
    intra = sum(1 for u, v in G.edges() if labels[u] == labels[v])
    return intra, G.number_of_edges() - intra


def intra_inter_edges(G: nx.Graph, partition: Sequence[set[int]]) -> tuple[int, int]:
    """(# edges inside communities, # edges between communities)."""
    return _edge_counts(G, labels_from_partition(partition, G.number_of_nodes()))


def intra_inter_ratio(G: nx.Graph, partition: Sequence[set[int]]) -> tuple[float, float]:
    """(intra fraction, inter fraction) of edges; (nan, nan) if the graph has no edges."""
    intra, inter = intra_inter_edges(G, partition)
    m = intra + inter
    return (intra / m, inter / m) if m else (float("nan"), float("nan"))


def coverage(G: nx.Graph, partition: Sequence[set[int]]) -> float:
    """Fraction of edges inside communities (= intra-community edge ratio)."""
    return intra_inter_ratio(G, partition)[0]


def performance(G: nx.Graph, partition: Sequence[set[int]]) -> float:
    """(intra edges + inter non-edges) / all pairs: fraction of correctly 'classified' pairs."""
    n = G.number_of_nodes()
    total = n * (n - 1) // 2
    if total == 0:
        return float("nan")
    labels = labels_from_partition(partition, n)
    sizes = np.bincount(labels)
    intra_pairs = int((sizes * (sizes - 1) // 2).sum())
    inter_pairs = total - intra_pairs
    intra_e, inter_e = _edge_counts(G, labels)
    return (intra_e + (inter_pairs - inter_e)) / total


def conductance_per_community(G: nx.Graph, partition: Sequence[set[int]]) -> list[float]:
    """Conductance cut(S, V\\S) / min(vol(S), vol(V\\S)) per community; nan where undefined.
    Lower = better separated community."""
    out = []
    for block in partition:
        if not block or len(block) == G.number_of_nodes():
            out.append(float("nan"))
            continue
        vol_s = sum(d for _, d in G.degree(block))
        vol_c = 2 * G.number_of_edges() - vol_s
        cut = nx.cut_size(G, block)
        denom = min(vol_s, vol_c)
        out.append(cut / denom if denom > 0 else float("nan"))
    return out


def mean_conductance(G: nx.Graph, partition: Sequence[set[int]]) -> float:
    """Unweighted mean of defined per-community conductances (nan if none defined)."""
    vals = np.array(conductance_per_community(G, partition), dtype=float)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def assignment_consistency(ref_labels: Sequence[int], det_labels: Sequence[int]) -> float:
    """Fraction of nodes assigned consistently after the best one-to-one matching of communities
    (Hungarian algorithm on the contingency table). Penalises splits/merges; in [0, 1]."""
    a, b = np.asarray(ref_labels), np.asarray(det_labels)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ua.size, ub.size), dtype=int)
    np.add.at(table, (ia, ib), 1)
    r, c = linear_sum_assignment(-table)
    return float(table[r, c].sum() / a.size)


def fixed_partition_report(H: nx.Graph, P0: Sequence[set[int]]) -> dict[str, float]:
    """Protocol A: metrics of ``H`` under the fixed reference partition ``P0``."""
    intra, inter = intra_inter_edges(H, P0)
    intra_r, inter_r = intra_inter_ratio(H, P0)
    return {
        "modularity_fixed": modularity_fixed_partition(H, P0),
        "coverage_fixed": coverage(H, P0),
        "performance_fixed": performance(H, P0),
        "conductance_fixed": mean_conductance(H, P0),
        "intra_edges_fixed": intra, "inter_edges_fixed": inter,
        "intra_ratio_fixed": intra_r, "inter_ratio_fixed": inter_r,
    }


def redetected_report(H: nx.Graph, P0: Sequence[set[int]], method: str = "louvain",
                      seed: int | None = None, **kwargs) -> dict[str, float]:
    """Protocol B: run ``method`` on ``H`` and compare the result with ``P0``."""
    n = H.number_of_nodes()
    PH = detect_communities(H, method, seed=seed, **kwargs)
    ref, det = labels_from_partition(P0, n), labels_from_partition(PH, n)
    sizes = community_sizes(PH)
    return {
        "modularity_redetected": modularity_fixed_partition(H, PH),
        "nmi": nmi(ref, det), "ari": ari(ref, det),
        "assignment_consistency": assignment_consistency(ref, det),
        "n_communities": num_communities(PH),
        "largest_community": sizes[0] if sizes else 0,
        "conductance_redetected": mean_conductance(H, PH),
        "coverage_redetected": coverage(H, PH),
        "performance_redetected": performance(H, PH),
        "sizes": sizes,
    }

"""Modularity of a graph under a *given* (fixed) partition.

    Q = sum_c [ L_c / m - (d_c / 2m)^2 ]

L_c = edges inside community c, d_c = total degree of c, m = total edges; Q := 0 when m = 0.
This is the formulation used in the project (partition P0 is fixed, never re-detected).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import networkx as nx
import numpy as np

from src.graph_generation.graph_preprocessing import pair_indices


def labels_from_partition(partition: Sequence[set[int]], n_nodes: int) -> np.ndarray:
    """Integer label per node ``0..n-1``; validates that the partition covers every node once."""
    labels = np.full(n_nodes, -1, dtype=int)
    for c, block in enumerate(partition):
        for v in block:
            if not 0 <= v < n_nodes:
                raise ValueError(f"Node {v} outside 0..{n_nodes - 1}.")
            if labels[v] != -1:
                raise ValueError(f"Node {v} appears in more than one community.")
            labels[v] = c
    if (labels < 0).any():
        raise ValueError("Partition does not cover all nodes.")
    return labels


def partition_from_labels(labels: Sequence[int] | Mapping[int, int]) -> list[set[int]]:
    """Inverse of :func:`labels_from_partition` (communities ordered by first appearance)."""
    items = labels.items() if isinstance(labels, Mapping) else enumerate(labels)
    blocks: dict[int, set[int]] = {}
    for node, lab in items:
        blocks.setdefault(int(lab), set()).add(int(node))
    return list(blocks.values())


def modularity_fixed_partition(G: nx.Graph, partition: Sequence[set[int]]) -> float:
    """Q of ``G`` under ``partition`` (own implementation; cross-checked against NetworkX in tests)."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    labels = labels_from_partition(partition, G.number_of_nodes())
    n_comm = labels.max() + 1
    intra = np.zeros(n_comm)
    deg = np.zeros(n_comm)
    for u, v in G.edges():
        deg[labels[u]] += 1
        deg[labels[v]] += 1
        if labels[u] == labels[v]:
            intra[labels[u]] += 1
    return float(np.sum(intra / m - (deg / (2 * m)) ** 2))


class PairVectorModularity:
    """Vectorised modularity for many releases at once under a fixed partition.

    Precomputes, for the N node pairs, the matrices needed to evaluate Q for a ``(K, N)`` batch of
    0/1 pair vectors with three matrix products (fast enough for NSGA-II inner loops).
    """

    def __init__(self, labels: np.ndarray):
        self.labels = np.asarray(labels, dtype=int)
        n = self.labels.size
        rows, cols = pair_indices(n)
        self.n_comm = int(self.labels.max()) + 1
        N = rows.size
        incidence = np.zeros((N, n))
        incidence[np.arange(N), rows] = 1
        incidence[np.arange(N), cols] = 1
        onehot = np.eye(self.n_comm)[self.labels]               # (n, C)
        self._deg_by_comm = incidence @ onehot                  # (N, C): degree mass each pair adds
        same = self.labels[rows] == self.labels[cols]
        self._intra_by_comm = np.zeros((N, self.n_comm))
        self._intra_by_comm[np.flatnonzero(same), self.labels[rows][same]] = 1.0
        self._intra_mask = same

    def modularity(self, releases: np.ndarray) -> np.ndarray:
        """Q for each row of ``releases`` (``(K, N)`` or ``(N,)``); Q = 0 for empty graphs."""
        R = np.atleast_2d(releases).astype(float)
        m = R.sum(axis=1)
        intra = R @ self._intra_by_comm
        deg = R @ self._deg_by_comm
        with np.errstate(divide="ignore", invalid="ignore"):
            q = (intra / m[:, None] - (deg / (2 * m[:, None])) ** 2).sum(axis=1)
        return np.where(m > 0, q, 0.0)

    def intra_inter_counts(self, releases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(intra-community edge count, inter-community edge count) per release."""
        R = np.atleast_2d(releases).astype(float)
        intra = R @ self._intra_mask.astype(float)
        return intra, R.sum(axis=1) - intra

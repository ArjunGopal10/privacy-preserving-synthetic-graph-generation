"""Graph <-> pair-vector conversion.

Randomized response acts on the *upper triangle of the adjacency matrix*: one bit for each of the
N = n(n-1)/2 unordered node pairs. The ordering is ``numpy.triu_indices(n, k=1)`` everywhere.
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx
import numpy as np


def num_possible_pairs(n_nodes: int) -> int:
    """Number of unordered node pairs, ``n (n - 1) / 2`` (561 for n = 34)."""
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative.")
    return n_nodes * (n_nodes - 1) // 2


@lru_cache(maxsize=16)
def pair_indices(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column indices of all unordered pairs (upper triangle, k=1)."""
    rows, cols = np.triu_indices(n_nodes, k=1)
    rows.setflags(write=False)
    cols.setflags(write=False)
    return rows, cols


def to_simple_graph(G: nx.Graph) -> nx.Graph:
    """Undirected simple graph (no self-loops/multi-edges) with nodes relabelled ``0..n-1``."""
    H = nx.Graph()
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    H.add_nodes_from(range(len(mapping)))
    H.add_edges_from((mapping[u], mapping[v]) for u, v in G.edges() if u != v)
    return H


def graph_to_pair_vector(G: nx.Graph) -> np.ndarray:
    """0/1 ``uint8`` vector of length N; entry k is 1 iff pair k is an edge of ``G``."""
    n = G.number_of_nodes()
    if set(G.nodes()) != set(range(n)):
        raise ValueError("Nodes must be labelled 0..n-1 (use to_simple_graph).")
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.uint8)
    rows, cols = pair_indices(n)
    return (A[rows, cols] > 0).astype(np.uint8)


def pair_vector_to_graph(vec: np.ndarray, n_nodes: int) -> nx.Graph:
    """Inverse of :func:`graph_to_pair_vector`."""
    vec = np.asarray(vec)
    if vec.shape != (num_possible_pairs(n_nodes),):
        raise ValueError(f"Expected vector of length {num_possible_pairs(n_nodes)}, got {vec.shape}.")
    rows, cols = pair_indices(n_nodes)
    idx = np.flatnonzero(vec)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(rows[idx].tolist(), cols[idx].tolist()))
    return G

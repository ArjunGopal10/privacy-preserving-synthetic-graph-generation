"""Loading graphs and reference partitions."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from .graph_preprocessing import to_simple_graph


def load_karate_club() -> nx.Graph:
    """Zachary's Karate Club (34 nodes, 78 edges) with nodes labelled ``0..33``."""
    return to_simple_graph(nx.karate_club_graph())


def load_edge_list(path: str | Path) -> nx.Graph:
    """Load a whitespace-separated edge list; nodes are relabelled to ``0..n-1``.

    Lines starting with ``#`` are ignored. Self-loops and duplicate edges are dropped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Edge list not found: {path}")
    return to_simple_graph(nx.read_edgelist(path, comments="#", nodetype=int))


def load_partition(path: str | Path, n_nodes: int | None = None) -> list[set[int]]:
    """Load a partition from JSON (``[[0, 1, ...], [5, 6, ...], ...]``) and validate it."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    partition = [set(int(v) for v in block) for block in data]
    seen: set[int] = set()
    for block in partition:
        if seen & block:
            raise ValueError("Partition blocks must be disjoint.")
        seen |= block
    if n_nodes is not None and seen != set(range(n_nodes)):
        raise ValueError(f"Partition must cover exactly nodes 0..{n_nodes - 1}.")
    return partition


def save_partition(partition: list[set[int]], path: str | Path) -> None:
    """Save a partition as JSON."""
    Path(path).write_text(json.dumps([sorted(int(v) for v in b) for b in partition]), encoding="utf-8")

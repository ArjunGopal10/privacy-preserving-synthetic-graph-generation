"""Common input loading for the experiment scripts (graph + fixed reference partition)."""
from __future__ import annotations

import argparse
import sys
from typing import Callable

import networkx as nx

from src.common import OutputExistsError
from src.community.community_detection import detect_communities
from src.graph_generation.graph_loader import load_edge_list, load_karate_club, load_partition


def load_inputs(args: argparse.Namespace) -> tuple[nx.Graph, list[set[int]], str]:
    """Return ``(graph, reference partition P0, description of where P0 came from)``.

    P0 is Louvain run ONCE on the original graph with ``--seed`` unless ``--partition-file`` is given.
    NOTE: Louvain output depends on node/edge insertion order and the NetworkX version, so the
    partition is written to ``results/reports/reference_partition.json`` by each experiment.
    """
    G = load_edge_list(args.edge_list) if args.edge_list else load_karate_club()
    if args.partition_file:
        return G, load_partition(args.partition_file, G.number_of_nodes()), f"file:{args.partition_file}"
    return G, detect_communities(G, "louvain", seed=args.seed), f"louvain(seed={args.seed}) on original"


def run_cli(main_fn: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    """Run ``main_fn`` and translate expected errors into clear messages and exit codes."""
    try:
        return main_fn(argv)
    except OutputExistsError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


def write_reference_partition(path, partition: list[set[int]], overwrite: bool = False) -> None:
    """Write the shared reference-partition file.

    Several experiments use the same P0. If the file already holds the same partition this is a
    no-op; if it holds a DIFFERENT partition the write is refused unless ``overwrite`` is set.
    """
    import json
    from pathlib import Path

    from src.common import OutputExistsError
    from src.graph_generation.graph_loader import save_partition

    path = Path(path)
    if path.exists() and not overwrite:
        existing = sorted(sorted(b) for b in json.loads(path.read_text(encoding="utf-8")))
        if existing == sorted(sorted(int(v) for v in b) for b in partition):
            return
        raise OutputExistsError(f"{path} holds a different reference partition (pass --overwrite).")
    save_partition(partition, path)

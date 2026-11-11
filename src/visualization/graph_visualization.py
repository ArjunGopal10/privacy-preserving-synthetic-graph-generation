"""Figures 9, 24, 25: node-link drawings (shared layout, nodes coloured by the fixed partition P0)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.community.modularity import labels_from_partition

NODE_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]


def fixed_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Deterministic spring layout computed once on the ORIGINAL graph and reused for all panels."""
    return nx.spring_layout(G, seed=seed, k=0.55, iterations=200)


def draw_graph(ax: plt.Axes, H: nx.Graph, pos: dict, partition, title: str,
               original: nx.Graph | None = None, node_size: int = 150) -> None:
    """Draw ``H``. If ``original`` is given, colour edges: kept (grey), added (red), deleted (dashed)."""
    labels = labels_from_partition(partition, H.number_of_nodes())
    colors = [NODE_PALETTE[l % len(NODE_PALETTE)] for l in labels]
    if original is None:
        nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#888888", width=0.9, alpha=0.8)
    else:
        oe = {tuple(sorted(e)) for e in original.edges()}
        he = {tuple(sorted(e)) for e in H.edges()}
        nx.draw_networkx_edges(H, pos, edgelist=sorted(oe & he), ax=ax, edge_color="#888888", width=0.9,
                               alpha=0.85)
        nx.draw_networkx_edges(H, pos, edgelist=sorted(he - oe), ax=ax, edge_color="#D55E00", width=0.7,
                               alpha=0.55)
        nx.draw_networkx_edges(H, pos, edgelist=sorted(oe - he), ax=ax, edge_color="#222222", width=0.9,
                               style="dashed", alpha=0.6)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=node_size, edgecolors="white",
                           linewidths=0.8)
    nx.draw_networkx_labels(H, pos, ax=ax, font_size=5.5, font_color="white")
    ax.set_title(title, fontsize=9.5)
    ax.set_axis_off()
    ax.grid(False)


def _edge_legend(fig: plt.Figure) -> None:
    handles = [Line2D([0], [0], color="#888888", lw=1.5, label="kept real edge"),
               Line2D([0], [0], color="#D55E00", lw=1.5, label="added (fake) edge"),
               Line2D([0], [0], color="#222222", lw=1.5, ls="--", label="deleted real edge")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)


def graph_edge_table(G: nx.Graph, H: nx.Graph, label: str) -> dict:
    oe = {tuple(sorted(e)) for e in G.edges()}
    he = {tuple(sorted(e)) for e in H.edges()}
    return {"graph": label, "edges": len(he), "kept": len(oe & he), "added": len(he - oe),
            "deleted": len(oe - he)}


def fig_original(G: nx.Graph, P0, pos: dict):
    """Figure 24: the original Karate Club graph coloured by the fixed Louvain partition."""
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    draw_graph(ax, G, pos, P0, f"Zachary's Karate Club: {G.number_of_nodes()} nodes, "
               f"{G.number_of_edges()} edges (colour = fixed partition P0)", node_size=260)
    return fig, pd.DataFrame([graph_edge_table(G, G, "original")])


def fig_epsilon_panels(G: nx.Graph, releases: dict[float, nx.Graph], P0, pos: dict, title: str):
    """Figure 25 (and 9): original next to releases at selected epsilons (one release each)."""
    k = len(releases) + 1
    fig, axes = plt.subplots(1, k, figsize=(4.1 * k, 4.5))
    draw_graph(axes[0], G, pos, P0, f"Original ({G.number_of_edges()} edges)")
    rows = [graph_edge_table(G, G, "original")]
    for ax, (eps, H) in zip(axes[1:], releases.items()):
        draw_graph(ax, H, pos, P0, rf"Randomized, $\varepsilon$={eps:g} ({H.number_of_edges()} edges)", G)
        rows.append(graph_edge_table(G, H, f"eps={eps:g}"))
    fig.suptitle(title, fontsize=11, y=1.0)
    _edge_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig, pd.DataFrame(rows)

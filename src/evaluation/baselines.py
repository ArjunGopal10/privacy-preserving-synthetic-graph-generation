"""Baseline graph generators/perturbation methods.

Each baseline function returns a NetworkX graph on nodes ``0..n-1``. The registry in
:data:`BASELINES` records, for every method, whether it carries a FORMAL edge-DP guarantee.

| method                    | formal edge-DP? | notes                                                        |
|---------------------------|-----------------|--------------------------------------------------------------|
| original                  | no (eps = inf)  | unperturbed reference                                        |
| randomized_response       | yes, eps        | the project's mechanism                                      |
| laplace_topm              | yes, eps        | simplified top-m filter (TmF-style): eps/2 noisy edge count  |
|                           |                 | + eps/2 Laplace on every pair, keep the top noisy-m pairs    |
| erdos_renyi_matched       | n/a             | G(n, m) with the ORIGINAL m: uses the true edge count, so    |
|                           |                 | it is NOT claimed to be DP; it ignores the structure         |
| degree_preserving         | no              | double-edge swaps keep every degree: leaks the exact degree  |
|                           |                 | sequence, so no DP claim                                     |
| deletion_only             | no              | each real edge deleted w.p. p; every released edge is real,  |
|                           |                 | so it violates DP (output proves the edge existed)           |
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np

from src.graph_generation.graph_preprocessing import (
    graph_to_pair_vector, num_possible_pairs, pair_vector_to_graph)
from src.graph_generation.randomized_response import (
    flip_probability, randomized_response_graph, validate_epsilon)


def original(G: nx.Graph, rng: np.random.Generator, **_) -> nx.Graph:
    return G.copy()


def randomized_response(G: nx.Graph, rng: np.random.Generator, epsilon: float, **_) -> nx.Graph:
    return randomized_response_graph(G, epsilon, rng)


def erdos_renyi_matched(G: nx.Graph, rng: np.random.Generator, **_) -> nx.Graph:
    """Uniform random graph with the same node count and the ORIGINAL edge count."""
    return nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=int(rng.integers(2**31)))


def degree_preserving(G: nx.Graph, rng: np.random.Generator, swaps_per_edge: int = 10, **_) -> nx.Graph:
    """Randomise by repeated double-edge swaps (all node degrees are preserved exactly)."""
    H = G.copy()
    m = H.number_of_edges()
    if m >= 2:
        nx.double_edge_swap(H, nswap=swaps_per_edge * m, max_tries=100 * swaps_per_edge * m,
                            seed=int(rng.integers(2**31)))
    return H


def deletion_only(G: nx.Graph, rng: np.random.Generator, epsilon: float, **_) -> nx.Graph:
    """Delete every real edge with probability p = 1/(1+e^eps); never add edges. NOT private."""
    p = flip_probability(epsilon)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(e for e in G.edges() if rng.random() >= p)
    return H


def laplace_topm(G: nx.Graph, rng: np.random.Generator, epsilon: float, count_share: float = 0.5,
                 **_) -> nx.Graph:
    """Simplified Top-m-Filter-style mechanism (inspired by Nguyen et al., 2015; NOT an exact
    reproduction). Edge-DP with total budget ``epsilon``:

    1. m~ = m + Lap(1/eps_1)   (edge-count sensitivity 1),   eps_1 = count_share * eps
    2. a~_ij = a_ij + Lap(1/eps_2) for every pair (sensitivity 1), eps_2 = eps - eps_1
    3. release the m~ pairs with the largest noisy values (post-processing).
    Sequential composition gives (eps_1 + eps_2) = eps.
    """
    eps = validate_epsilon(epsilon)
    if eps <= 0:
        raise ValueError("laplace_topm needs epsilon > 0.")
    n, N = G.number_of_nodes(), num_possible_pairs(G.number_of_nodes())
    eps1, eps2 = count_share * eps, (1 - count_share) * eps
    vec = graph_to_pair_vector(G).astype(float)
    m_noisy = int(np.clip(round(vec.sum() + rng.laplace(0, 1 / eps1)), 0, N))
    noisy = vec + rng.laplace(0, 1 / eps2, size=N)
    out = np.zeros(N, dtype=np.uint8)
    if m_noisy > 0:
        out[np.argpartition(-noisy, m_noisy - 1)[:m_noisy]] = 1
    return pair_vector_to_graph(out, n)


@dataclass(frozen=True)
class Baseline:
    name: str
    fn: Callable[..., nx.Graph]
    uses_epsilon: bool
    formal_edge_dp: bool
    note: str


BASELINES: dict[str, Baseline] = {b.name: b for b in [
    Baseline("original", original, False, False, "No perturbation (reference)."),
    Baseline("randomized_response", randomized_response, True, True, "Edge-DP, one release."),
    Baseline("laplace_topm", laplace_topm, True, True, "Edge-DP (simplified TmF-style)."),
    Baseline("erdos_renyi_matched", erdos_renyi_matched, False, False,
             "Uses the true edge count; structure-free random graph."),
    Baseline("degree_preserving", degree_preserving, False, False,
             "Keeps exact degree sequence; no DP claim."),
    Baseline("deletion_only", deletion_only, True, False, "Only deletes edges; not DP."),
]}


def generate(name: str, G: nx.Graph, rng: np.random.Generator, epsilon: float | None = None) -> nx.Graph:
    """Generate one graph from baseline ``name`` (``epsilon`` required if the method uses it)."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {list(BASELINES)}.")
    b = BASELINES[name]
    if b.uses_epsilon and epsilon is None:
        raise ValueError(f"Baseline {name!r} requires epsilon.")
    return b.fn(G, rng, epsilon=epsilon)

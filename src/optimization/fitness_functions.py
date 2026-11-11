"""Objective (fitness) functions for the epsilon search.

Decision variable : epsilon in [eps_min, eps_max]
f1(eps)           = eps                       (privacy budget; minimise)
f2(eps)           = - E[ Q(H; P0) ]           (negated expected modularity; minimise)

pymoo *minimises* every objective, so utility (modularity, to be maximised) enters with a minus
sign: minimising -E[Q] is identical to maximising E[Q]. Using ``1 - Q`` or any decreasing function
would give the same Pareto set; the plain negative keeps the units interpretable.

E[Q] is estimated by averaging Q over K randomized releases. With ``common_random_numbers=True``
(default) the *same* K uniform matrices are re-used for every epsilon, so f2 is a deterministic
function of epsilon (no evaluation noise, and the estimate is monotone-coupled across epsilons).
With ``common_random_numbers=False`` fresh noise is drawn at every call: f2 becomes a noisy
objective, which can make nearly-equal solutions look non-dominated or dominated by chance.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from src.community.modularity import PairVectorModularity, labels_from_partition
from src.graph_generation.graph_preprocessing import graph_to_pair_vector, num_possible_pairs
from src.graph_generation.randomized_response import (
    draw_uniforms, randomized_response_vectors, validate_epsilon)


class ExpectedModularityEvaluator:
    """Monte-Carlo estimate of ``E[Q(H; P0)]`` for RR releases at a given epsilon."""

    def __init__(self, G: nx.Graph, partition: list[set[int]], n_releases: int = 20,
                 seed: int = 42, common_random_numbers: bool = True):
        if n_releases < 1:
            raise ValueError("n_releases must be >= 1.")
        self.vec = graph_to_pair_vector(G)
        self.n_nodes = G.number_of_nodes()
        self.n_pairs = num_possible_pairs(self.n_nodes)
        self.labels = labels_from_partition(partition, self.n_nodes)
        self._q = PairVectorModularity(self.labels)
        self.n_releases = n_releases
        self.seed = seed
        self.crn = common_random_numbers
        self._U = draw_uniforms(self.n_pairs, n_releases, seed)
        self._noise_rng = np.random.default_rng(seed + 1)
        self.n_evaluations = 0
        self.original_modularity = float(self._q.modularity(self.vec)[0])

    def _uniforms(self) -> np.ndarray:
        return self._U if self.crn else self._noise_rng.random((self.n_releases, self.n_pairs))

    def modularity_samples(self, epsilon: float) -> np.ndarray:
        """Q of each of the K releases at ``epsilon``."""
        self.n_evaluations += 1
        releases = randomized_response_vectors(self.vec, validate_epsilon(epsilon), self._uniforms())
        return self._q.modularity(releases)

    def expected_modularity(self, epsilon: float) -> float:
        return float(self.modularity_samples(epsilon).mean())

    def objectives(self, epsilon: float) -> np.ndarray:
        """``[f1, f2] = [epsilon, -E[Q]]`` (both to be minimised)."""
        return np.array([float(epsilon), -self.expected_modularity(epsilon)])

    def objectives_batch(self, epsilons: np.ndarray) -> np.ndarray:
        """Vector form: array of shape ``(len(epsilons), 2)``."""
        return np.vstack([self.objectives(e) for e in np.asarray(epsilons).ravel()])


def combined_fitness(epsilon: float, expected_q: float, q_original: float, weight: float = 0.5,
                     eps_min: float = 0.1, eps_max: float = 10.0) -> float:
    """Weighted-sum scalarisation used by the website's single-objective GA baseline:
    ``(1 - w) * eps_norm - w * Q / Q0`` (minimised). Not used by NSGA-II."""
    eps_norm = (epsilon - eps_min) / (eps_max - eps_min)
    return float((1 - weight) * eps_norm - weight * expected_q / q_original)

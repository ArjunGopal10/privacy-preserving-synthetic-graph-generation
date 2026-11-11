import math

import networkx as nx
import numpy as np
import pytest

from src.graph_generation.graph_loader import load_karate_club
from src.graph_generation.graph_preprocessing import (graph_to_pair_vector, num_possible_pairs,
                                                      pair_vector_to_graph)
from src.graph_generation.randomized_response import (draw_uniforms, flip_probability,
                                                      randomized_response_graph,
                                                      randomized_response_vectors)
from src.privacy.differential_privacy import (epsilon_from_flip_probability, max_log_likelihood_ratio)


def test_flip_probability_formula():
    for eps in [0.1, 0.5, 1, 2, 5, 8]:
        assert flip_probability(eps) == pytest.approx(1 / (1 + math.exp(eps)))


def test_epsilon_boundaries():
    assert flip_probability(0) == pytest.approx(0.5)
    assert flip_probability(math.inf) == 0.0
    assert flip_probability(1000) == pytest.approx(0.0, abs=1e-300)   # no overflow
    with pytest.raises(ValueError):
        flip_probability(-0.1)
    with pytest.raises(ValueError):
        flip_probability(float("nan"))


def test_number_of_possible_edges():
    assert num_possible_pairs(34) == 561
    assert num_possible_pairs(0) == 0 and num_possible_pairs(1) == 0 and num_possible_pairs(2) == 1
    with pytest.raises(ValueError):
        num_possible_pairs(-1)


def test_karate_dimensions_and_roundtrip():
    G = load_karate_club()
    assert (G.number_of_nodes(), G.number_of_edges()) == (34, 78)
    vec = graph_to_pair_vector(G)
    assert vec.shape == (561,) and vec.sum() == 78
    assert nx.utils.graphs_equal(G, pair_vector_to_graph(vec, 34))


def test_graph_generation_keeps_nodes_and_is_simple():
    G = load_karate_club()
    H = randomized_response_graph(G, 1.0, rng=0)
    assert set(H.nodes()) == set(G.nodes())
    assert nx.number_of_selfloops(H) == 0


def test_eps_infinity_is_identity_and_eps_zero_is_noise():
    G = load_karate_club()
    vec = graph_to_pair_vector(G)
    U = draw_uniforms(561, 50, seed=1)
    assert (randomized_response_vectors(vec, math.inf, U) == vec).all()
    flips = (randomized_response_vectors(vec, 0.0, U) != vec).mean()
    assert flips == pytest.approx(0.5, abs=0.02)


def test_empirical_flip_rate_matches_p():
    vec = graph_to_pair_vector(load_karate_club())
    U = draw_uniforms(561, 400, seed=3)
    for eps in (0.5, 2.0):
        rate = (randomized_response_vectors(vec, eps, U) != vec).mean()
        assert rate == pytest.approx(flip_probability(eps), abs=0.01)


def test_common_random_numbers_are_nested():
    """Flips at a larger epsilon are a subset of flips at a smaller epsilon (coupling)."""
    vec = graph_to_pair_vector(load_karate_club())
    U = draw_uniforms(561, 10, seed=5)
    low = randomized_response_vectors(vec, 0.5, U) != vec
    high = randomized_response_vectors(vec, 3.0, U) != vec
    assert not (high & ~low).any()


def test_dp_ratio_equals_epsilon():
    for eps in (0.1, 1, 5):
        assert max_log_likelihood_ratio(eps) == pytest.approx(eps)
        assert epsilon_from_flip_probability(flip_probability(eps)) == pytest.approx(eps)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        randomized_response_vectors(np.zeros(5, dtype=np.uint8), 1.0, np.zeros((1, 6)))
    with pytest.raises(ValueError):
        pair_vector_to_graph(np.zeros(5), 34)
    with pytest.raises(ValueError):
        epsilon_from_flip_probability(0.7)

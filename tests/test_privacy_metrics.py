import numpy as np
import pytest

from src.graph_generation.graph_loader import load_karate_club
from src.graph_generation.graph_preprocessing import graph_to_pair_vector
from src.graph_generation.randomized_response import draw_uniforms, randomized_response_vectors
from src.privacy.differential_privacy import (basic_composition, expected_release_counts,
                                              posterior_edge_belief)
from src.privacy.privacy_analysis import privacy_budget_table, sweep_privacy
from src.privacy.privacy_metrics import (all_edge_metrics, analytic_rr_metrics, edge_f1,
                                         edge_jaccard, edge_precision, edge_recall, graph_density)

ORIG = np.array([1, 1, 1, 0, 0, 0, 0, 0])
REL = np.array([1, 1, 0, 1, 1, 0, 0, 0])      # TP=2, FN=1, FP=2


def test_hand_computed_metrics():
    assert edge_precision(ORIG, REL) == pytest.approx(2 / 4)
    assert edge_recall(ORIG, REL) == pytest.approx(2 / 3)
    assert edge_f1(ORIG, REL) == pytest.approx(2 * 2 / (3 + 4))
    assert edge_jaccard(ORIG, REL) == pytest.approx(2 / 5)
    assert graph_density(REL) == pytest.approx(0.5)


def test_identical_and_disjoint():
    assert edge_jaccard(ORIG, ORIG) == 1 and edge_f1(ORIG, ORIG) == 1
    other = 1 - ORIG
    assert edge_jaccard(ORIG, other) == 0 and edge_precision(ORIG, other) == 0


def test_zero_denominators_are_zero():
    z = np.zeros(8, dtype=int)
    assert edge_precision(ORIG, z) == 0 and edge_recall(z, REL) == 0
    assert edge_f1(z, z) == 0 and edge_jaccard(z, z) == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        edge_precision(np.zeros(3), np.zeros(4))


def test_all_edge_metrics_counts():
    m = all_edge_metrics(ORIG, REL)
    assert m["edges_added"] == 2 and m["edges_deleted"] == 1 and m["n_edges"] == 4


def test_empirical_matches_analytic_on_karate():
    vec = graph_to_pair_vector(load_karate_club())
    U = draw_uniforms(561, 300, seed=7)
    for eps in (1.0, 3.0):
        rel = randomized_response_vectors(vec, eps, U)
        emp = np.mean([edge_precision(vec, r) for r in rel])
        assert emp == pytest.approx(analytic_rr_metrics(561, 78, eps)["precision"], abs=0.02)


def test_expected_counts_and_posterior():
    c = expected_release_counts(561, 78, 0.0)
    assert c["total_edges"] == pytest.approx(561 / 2)
    assert posterior_edge_belief(78 / 561, 0.0) == pytest.approx(78 / 561)     # eps=0: no information
    assert posterior_edge_belief(78 / 561, 8.0) > 0.99
    with pytest.raises(ValueError):
        posterior_edge_belief(1.5, 1.0)


def test_composition_and_budget_table():
    assert basic_composition([1, 2, 0.5]) == pytest.approx(3.5)
    t = privacy_budget_table([0.1, 1.0], 561, 78)
    assert list(t["epsilon"]) == [0.1, 1.0]
    assert t["total_epsilon_10_releases"].tolist() == pytest.approx([1.0, 10.0])


def test_sweep_shapes():
    G = load_karate_club()
    per, summ = sweep_privacy(G, [0.5, 2.0], 5, seed=0)
    assert len(per) == 10 and len(summ) == 2
    assert {"precision_mean", "precision_analytic", "precision_ci95"} <= set(summ.columns)

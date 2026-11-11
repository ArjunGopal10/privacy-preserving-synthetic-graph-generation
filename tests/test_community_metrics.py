import networkx as nx
import numpy as np
import pytest

from src.community.community_detection import detect_communities, METHODS, leiden_available
from src.community.community_metrics import (ari, assignment_consistency, conductance_per_community,
                                             coverage, fixed_partition_report, intra_inter_edges,
                                             intra_inter_ratio, nmi, performance, redetected_report)
from src.community.modularity import (PairVectorModularity, labels_from_partition,
                                      modularity_fixed_partition, partition_from_labels)
from src.graph_generation.graph_loader import load_karate_club
from src.graph_generation.graph_preprocessing import graph_to_pair_vector


@pytest.fixture(scope="module")
def karate():
    G = load_karate_club()
    return G, detect_communities(G, "louvain", seed=42)


def test_modularity_matches_networkx(karate):
    G, P = karate
    assert modularity_fixed_partition(G, P) == pytest.approx(nx.community.modularity(G, P))


def test_modularity_two_cliques():
    G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    assert modularity_fixed_partition(G, [{0, 1, 2, 3}, {4, 5, 6, 7}]) == pytest.approx(0.5)
    assert modularity_fixed_partition(G, [set(range(8))]) == pytest.approx(0.0)


def test_modularity_empty_graph_is_zero():
    G = nx.empty_graph(5)
    assert modularity_fixed_partition(G, [{0, 1}, {2, 3, 4}]) == 0.0


def test_vectorised_modularity_matches_scalar(karate):
    G, P = karate
    pv = PairVectorModularity(labels_from_partition(P, 34))
    rng = np.random.default_rng(0)
    from src.graph_generation.graph_preprocessing import pair_vector_to_graph
    vecs = (rng.random((5, 561)) < 0.2).astype(np.uint8)
    expected = [modularity_fixed_partition(pair_vector_to_graph(v, 34), P) for v in vecs]
    assert pv.modularity(vecs) == pytest.approx(expected)
    assert pv.modularity(np.zeros(561)) == pytest.approx([0.0])
    intra, inter = pv.intra_inter_counts(graph_to_pair_vector(G))
    assert (intra[0], inter[0]) == intra_inter_edges(G, P)


def test_partition_validation():
    with pytest.raises(ValueError):
        labels_from_partition([{0, 1}, {1, 2}], 3)          # overlap
    with pytest.raises(ValueError):
        labels_from_partition([{0, 1}], 3)                  # uncovered node
    assert partition_from_labels([0, 0, 1]) == [{0, 1}, {2}]


def test_nmi_ari_basics():
    a = [0, 0, 1, 1, 2, 2]
    assert nmi(a, [5, 5, 7, 7, 9, 9]) == pytest.approx(1.0)      # label permutation invariant
    assert ari(a, [5, 5, 7, 7, 9, 9]) == pytest.approx(1.0)
    assert nmi(a, [0] * 6) == pytest.approx(0.0)
    assert ari(a, [0, 1, 0, 1, 0, 1]) <= 0.1


def test_assignment_consistency():
    assert assignment_consistency([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert assignment_consistency([0, 0, 1, 1], [0, 0, 0, 0]) == 0.5


def test_fixed_partition_quantities(karate):
    G, P = karate
    intra, inter = intra_inter_edges(G, P)
    assert intra + inter == 78
    r = intra_inter_ratio(G, P)
    assert r[0] + r[1] == pytest.approx(1.0) and coverage(G, P) == pytest.approx(r[0])
    assert 0 < performance(G, P) < 1
    cond = conductance_per_community(G, P)
    assert len(cond) == len(P) and all(0 <= c <= 1 for c in cond)
    nx_cov, nx_perf = nx.community.partition_quality(G, P)
    assert coverage(G, P) == pytest.approx(nx_cov) and performance(G, P) == pytest.approx(nx_perf)
    rep = fixed_partition_report(G, P)
    assert rep["modularity_fixed"] == pytest.approx(0.3853550295857988)


def test_redetected_on_original_recovers_reference(karate):
    G, P = karate
    rep = redetected_report(G, P, "louvain", seed=42)
    assert rep["nmi"] == pytest.approx(1.0) and rep["n_communities"] == len(P)


@pytest.mark.parametrize("method", [m for m in METHODS])
def test_all_detectors_return_partitions(method, karate):
    if method == "leiden" and not leiden_available():
        pytest.skip("leiden dependencies not installed")
    G, _ = karate
    part = detect_communities(G, method, seed=1)
    labels_from_partition(part, 34)       # raises if not a valid partition
    assert modularity_fixed_partition(G, part) > 0.3


def test_detectors_handle_edgeless_graph():
    part = detect_communities(nx.empty_graph(4), "louvain", seed=0)
    assert len(part) == 4
    with pytest.raises(ValueError):
        detect_communities(nx.empty_graph(4), "nope")

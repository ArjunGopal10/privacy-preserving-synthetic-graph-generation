import numpy as np
import pytest
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.community.community_detection import detect_communities
from src.graph_generation.graph_loader import load_karate_club
from src.optimization.fitness_functions import ExpectedModularityEvaluator, combined_fitness
from src.optimization.nsga2_components import (assign_rank_and_crowding, crowding_distance,
                                               environmental_selection, fast_non_dominated_sort,
                                               polynomial_mutation, sbx_crossover)
from src.optimization.nsga2_optimizer import NSGA2Config, grid_search, run_nsga2, run_nsga2_scratch
from src.optimization.pareto_analysis import (compare_with_grid, extract_pareto_front, hypervolume_2d,
                                              non_dominated_mask, select_solutions)


@pytest.fixture(scope="module")
def setup():
    G = load_karate_club()
    return G, detect_communities(G, "louvain", seed=42)


def evaluator(setup, **kw):
    return ExpectedModularityEvaluator(setup[0], setup[1], n_releases=kw.pop("n_releases", 10),
                                       seed=kw.pop("seed", 42), **kw)


def test_objective_shape_and_signs(setup):
    ev = evaluator(setup)
    f = ev.objectives(2.0)
    assert f.shape == (2,) and f[0] == 2.0 and f[1] == pytest.approx(-ev.expected_modularity(2.0))
    assert ev.objectives_batch(np.array([0.5, 1.0, 3.0])).shape == (3, 2)
    assert ev.original_modularity == pytest.approx(0.3853550295857988)
    assert ev.expected_modularity(10.0) == pytest.approx(ev.original_modularity, abs=0.01)


def test_crn_makes_objective_deterministic_and_noise_does_not(setup):
    crn, noisy = evaluator(setup), evaluator(setup, common_random_numbers=False)
    assert crn.expected_modularity(1.0) == crn.expected_modularity(1.0)
    assert noisy.expected_modularity(1.0) != noisy.expected_modularity(1.0)


def test_invalid_config():
    with pytest.raises(ValueError):
        NSGA2Config(pop_size=2)
    with pytest.raises(ValueError):
        NSGA2Config(eps_min=5, eps_max=1)


def test_non_dominated_sort_matches_pymoo():
    F = np.random.default_rng(0).random((80, 2))
    mine = [sorted(f.tolist()) for f in fast_non_dominated_sort(F)]
    ref = [sorted(f.tolist()) for f in NonDominatedSorting().do(F)]
    assert mine == ref


def test_pareto_mask_and_front_extraction():
    F = np.array([[1, 5], [2, 3], [3, 4], [4, 1], [4, 1]])
    assert non_dominated_mask(F).tolist() == [True, True, False, True, True]
    front = extract_pareto_front(F[:, :1], F)
    assert len(front) == 3 and front["epsilon"].is_monotonic_increasing    # duplicates removed


def test_hypervolume_hand_computed():
    assert hypervolume_2d(np.array([[1.0, 1.0]]), (3.0, 3.0)) == pytest.approx(4.0)
    assert hypervolume_2d(np.array([[1.0, 2.0], [2.0, 1.0]]), (3.0, 3.0)) == pytest.approx(3.0)
    assert hypervolume_2d(np.array([[5.0, 5.0]]), (3.0, 3.0)) == 0.0


def test_crowding_distance_boundaries_infinite():
    cd = crowding_distance(np.array([[0, 4], [1, 3], [2, 2], [4, 0]], float))
    assert np.isinf(cd[0]) and np.isinf(cd[3]) and np.isfinite(cd[1:3]).all()


def test_variation_operators_respect_bounds():
    rng = np.random.default_rng(0)
    xl, xu = np.array([0.1]), np.array([10.0])
    p1, p2 = rng.uniform(0.1, 10, (200, 1)), rng.uniform(0.1, 10, (200, 1))
    c1, c2 = sbx_crossover(p1, p2, xl, xu, 15, 0.9, rng=rng)
    m = polynomial_mutation(c1, xl, xu, 20, rng=rng)
    for arr in (c1, c2, m):
        assert arr.min() >= 0.1 and arr.max() <= 10.0
    assert not np.allclose(c1, p1)


def test_environmental_selection_size_and_elitism():
    rng = np.random.default_rng(1)
    F = rng.random((30, 2))
    keep = environmental_selection(F, 10)
    assert len(keep) == len(set(keep.tolist())) == 10
    front0 = set(fast_non_dominated_sort(F)[0].tolist())
    assert len(front0) > 10 or front0 <= set(keep.tolist())


def test_pymoo_run_structure_and_bounds(setup):
    cfg = NSGA2Config(pop_size=16, n_gen=6, n_releases=5)
    res = run_nsga2(evaluator(setup, n_releases=5), cfg)
    assert len(res.generations) == 6 and res.final_F.shape == (16, 2)
    assert res.final_X.min() >= cfg.eps_min and res.final_X.max() <= cfg.eps_max
    assert (res.populations.groupby("generation").size() == 16).all()
    hv = res.generations["hypervolume"]
    # Not strictly monotone: once the first front exceeds the population, crowding truncation can
    # drop points. Search must still improve on the random initial population.
    assert hv.iloc[-1] >= hv.iloc[0]


def test_scratch_run_matches_pymoo_quality(setup):
    cfg = NSGA2Config(pop_size=24, n_gen=15, n_releases=8)
    a = run_nsga2(evaluator(setup, n_releases=8), cfg)
    b = run_nsga2_scratch(evaluator(setup, n_releases=8), cfg)
    assert b.generations["hypervolume"].iloc[-1] == pytest.approx(a.generations["hypervolume"].iloc[-1], rel=0.03)


def test_grid_and_comparison(setup):
    ev = evaluator(setup, n_releases=8)
    grid = grid_search(ev, n_points=60)
    assert len(grid) == 60 and grid["on_pareto_front"].any()
    res = run_nsga2(evaluator(setup, n_releases=8), NSGA2Config(pop_size=16, n_gen=8, n_releases=8))
    stats = compare_with_grid(res.front, grid, (10.0, 0.1))
    assert 0.9 < stats["hv_ratio"] <= 1.05
    sel = select_solutions(res.front, ev.original_modularity)
    assert (sel["utility_fraction"] > 0).all()


def test_combined_fitness_bounds():
    assert combined_fitness(0.1, 0.0, 0.385) == pytest.approx(0.0)
    assert combined_fitness(10.0, 0.385, 0.385, 0.5) == pytest.approx(0.5 - 0.5)

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import OutputExistsError, check_overwrite, parse_epsilons
from src.community.community_detection import detect_communities
from src.graph_generation.graph_loader import load_karate_club
from src.graph_generation.randomized_response import draw_uniforms, randomized_response_graph
from src.optimization.fitness_functions import ExpectedModularityEvaluator
from src.optimization.nsga2_optimizer import NSGA2Config, run_nsga2
from src.privacy.privacy_analysis import sweep_privacy

ROOT = Path(__file__).resolve().parents[1]


def test_same_seed_same_release_different_seed_differs():
    G = load_karate_club()
    a, b = randomized_response_graph(G, 1.0, 5), randomized_response_graph(G, 1.0, 5)
    c = randomized_response_graph(G, 1.0, 6)
    assert set(map(frozenset, a.edges())) == set(map(frozenset, b.edges()))
    assert set(map(frozenset, a.edges())) != set(map(frozenset, c.edges()))
    assert (draw_uniforms(10, 3, 1) == draw_uniforms(10, 3, 1)).all()


def test_louvain_reference_partition_reproducible():
    G = load_karate_club()
    p1, p2 = detect_communities(G, "louvain", seed=42), detect_communities(G, "louvain", seed=42)
    assert sorted(map(sorted, p1)) == sorted(map(sorted, p2))


def test_sweep_and_nsga2_reproducible():
    G = load_karate_club()
    P = detect_communities(G, "louvain", seed=42)
    s1 = sweep_privacy(G, [1.0, 3.0], 5, 0)[1]
    s2 = sweep_privacy(G, [1.0, 3.0], 5, 0)[1]
    pd.testing.assert_frame_equal(s1, s2)
    cfg = NSGA2Config(pop_size=12, n_gen=5, n_releases=4)
    r1 = run_nsga2(ExpectedModularityEvaluator(G, P, 4, 42), cfg)
    r2 = run_nsga2(ExpectedModularityEvaluator(G, P, 4, 42), cfg)
    assert np.allclose(r1.final_F, r2.final_F)


def test_overwrite_guard(tmp_path):
    f = tmp_path / "x.csv"
    check_overwrite([f], overwrite=False)
    f.write_text("a")
    with pytest.raises(OutputExistsError):
        check_overwrite([f], overwrite=False)
    check_overwrite([f], overwrite=True)


def test_parse_epsilons():
    assert parse_epsilons("1, 0.5,1") == [0.5, 1.0]
    with pytest.raises(ValueError):
        parse_epsilons("-1")


def test_script_runs_writes_metadata_and_refuses_overwrite(tmp_path):
    cmd = [sys.executable, str(ROOT / "experiments" / "run_privacy_comparison.py"),
           "--results-dir", str(tmp_path), "--epsilons", "1,3", "--n-releases", "3"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    meta = json.loads((tmp_path / "reports" / "privacy_comparison_metadata.json").read_text())
    assert meta["config"]["seed"] == 42 and "numpy" in meta["library_versions"]
    assert (tmp_path / "figures" / "fig07_privacy_metric_dashboard.png").exists()
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 2 and "overwrite" in second.stderr.lower()

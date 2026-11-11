# Data

## Included
`sample/karate_club_edges.txt` — Zachary's Karate Club (34 nodes, 78 undirected edges, nodes `0..33`).
Source: W. W. Zachary (1977), "An information flow model for conflict and fission in small groups", *Journal of Anthropological Research* 33(4):452–473. The same graph ships with NetworkX (`networkx.karate_club_graph()`), which is what the experiments use by default. The file is 78 lines of node pairs (a published, widely redistributed network benchmark); cite the original paper when using it.

## Using your own graph
```bash
python experiments/run_complete_experiment.py --edge-list path/to/edges.txt --results-dir results_custom
```
Format: whitespace-separated integer node ids, one edge per line, `#` comments allowed; self-loops and duplicates are dropped and nodes relabelled to `0..n-1`. Note: randomized response is dense (O(n²) pairs) — keep graphs to a few hundred nodes.

## Not included
Large graphs (e.g. SNAP ego-Facebook, ~4,039 nodes) are **not** redistributed. If you add them, download from https://snap.stanford.edu/data/ ( check the licence) into `data/raw/` (git-ignored).

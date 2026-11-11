# Contributing

Issues and pull requests are welcome.

1. Fork, create a branch, `pip install -r requirements.txt`.
2. Keep the privacy-claim discipline: never describe empirical indicators (precision, recall, F1, Jaccard, NMI, …) as differential-privacy metrics, and never claim node-level or end-to-end DP.
3. Add or update tests in `tests/` and run `python -m pytest -q` (all tests must pass).
4. Use type hints and docstrings; keep functions small and reusable.
5. Do not add fabricated results. Result files must come from the scripts in `experiments/` (with metadata).
6. If you regenerate `results/`, run `python experiments/run_complete_experiment.py --overwrite` and commit the CSVs, figures and metadata together.

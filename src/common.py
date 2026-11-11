"""Shared utilities: configuration defaults, seeds, metadata, overwrite protection, CLI parsing."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as importlib_metadata
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR: Path = REPO_ROOT / "results"
DEFAULT_EPSILONS: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
DEFAULT_SEED: int = 42
DEFAULT_N_RELEASES: int = 20

TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "scipy", "pandas", "networkx", "matplotlib", "scikit-learn",
    "pymoo", "python-igraph", "leidenalg", "pytest",
)


class OutputExistsError(FileExistsError):
    """Raised when an experiment would overwrite existing results without ``--overwrite``."""


def library_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions of tracked packages (``"not installed"`` if absent)."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def parse_epsilons(text: str | Sequence[float]) -> list[float]:
    """Parse ``"0.1,0.5,1"`` (or a sequence) into a sorted list of positive floats."""
    if isinstance(text, str):
        values = [float(t) for t in text.replace(" ", "").split(",") if t]
    else:
        values = [float(t) for t in text]
    if not values:
        raise ValueError("At least one epsilon value is required.")
    if any(v < 0 for v in values):
        raise ValueError("Epsilon values must be non-negative.")
    return sorted(set(values))


def check_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    """Abort with a clear message if any of ``paths`` exist and ``overwrite`` is False."""
    existing = [p for p in paths if Path(p).exists()]
    if existing and not overwrite:
        listing = "\n  ".join(str(p) for p in existing[:10])
        more = f"\n  ... and {len(existing) - 10} more" if len(existing) > 10 else ""
        raise OutputExistsError(
            "Refusing to overwrite existing results (pass --overwrite to replace them):\n  "
            f"{listing}{more}"
        )


def ensure_results_dirs(results_dir: Path) -> dict[str, Path]:
    """Create and return the standard results sub-directories."""
    dirs = {name: Path(results_dir) / name for name in ("tables", "figures", "csv", "reports")}
    dirs["figure_data"] = dirs["csv"] / "figure_data"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_metadata(path: Path, script: str, config: dict, extra: dict | None = None) -> Path:
    """Write experiment metadata (config, seeds, library versions, platform) as JSON."""
    payload = {
        "script": script,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "library_versions": library_versions(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    text = json.dumps(payload, indent=2, default=str)
    # Do not leak absolute local paths into committed metadata.
    for prefix, repl in ((str(REPO_ROOT), "<repo>"), (str(Path.home()), "<home>")):
        text = text.replace(prefix, repl)
    path.write_text(text, encoding="utf-8")
    return path


def base_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser with the options shared by every experiment script."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Master random seed.")
    p.add_argument("--epsilons", type=str, default=",".join(str(e) for e in DEFAULT_EPSILONS),
                   help="Comma-separated epsilon values.")
    p.add_argument("--n-releases", type=int, default=DEFAULT_N_RELEASES,
                   help="Randomized releases per epsilon.")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                   help="Output directory (tables/, figures/, csv/, reports/ are created inside).")
    p.add_argument("--edge-list", type=Path, default=None,
                   help="Optional edge-list file; default is the NetworkX Karate Club graph.")
    p.add_argument("--partition-file", type=Path, default=None,
                   help="Optional JSON file with a fixed reference partition (list of node lists). "
                        "Default: Louvain run once on the original graph with --seed.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing results.")
    return p


def dataframe_to_markdown(df, float_fmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies)."""
    def fmt(v):
        if isinstance(v, float):
            return "nan" if v != v else float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"

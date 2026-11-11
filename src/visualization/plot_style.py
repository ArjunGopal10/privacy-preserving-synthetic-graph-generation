"""Shared plotting style, colours and the figure-saving helper (PNG + PDF + underlying CSV data)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import check_overwrite  # noqa: E402

# Okabe-Ito colour-blind-safe palette; the same colour means the same thing in every figure.
COLORS = {
    "original": "#000000", "randomized_response": "#0072B2", "laplace_topm": "#D55E00",
    "erdos_renyi_matched": "#009E73", "degree_preserving": "#CC79A7", "deletion_only": "#E69F00",
    "fixed": "#0072B2", "redetected": "#D55E00", "nsga2": "#0072B2", "grid": "#7F7F7F",
    "scratch": "#009E73", "analytic": "#7F7F7F", "selected": "#D55E00",
    "louvain": "#0072B2", "greedy_modularity": "#009E73", "label_propagation": "#E69F00",
    "girvan_newman": "#CC79A7", "leiden": "#D55E00",
}
LABELS = {
    "original": "Original", "randomized_response": "Randomized response (edge-DP)",
    "laplace_topm": "Laplace top-m (edge-DP)", "erdos_renyi_matched": "Matched G(n,m) (no DP)",
    "degree_preserving": "Degree-preserving (no DP)", "deletion_only": "Deletion-only (no DP)",
    "louvain": "Louvain", "greedy_modularity": "Greedy modularity",
    "label_propagation": "Label propagation", "girvan_newman": "Girvan–Newman", "leiden": "Leiden",
}


def apply_style() -> None:
    """Consistent research-paper style (call once per script)."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "lines.markersize": 5, "legend.frameon": False,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def eps_axis(ax: plt.Axes, log: bool = True) -> None:
    """Label the epsilon axis (log-scale by default; the tested grid spans 0.1 to 10)."""
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    if log:
        ax.set_xscale("log")
        ticks = [0.1, 0.5, 1, 2, 3, 5, 8]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()


def errorbar_line(ax: plt.Axes, x, mean, ci, color: str, label: str | None = None, marker: str = "o",
                  ls: str = "-", **kw) -> None:
    """Line with 95% CI error bars."""
    ax.errorbar(x, mean, yerr=ci, color=color, marker=marker, ls=ls, capsize=2.5, elinewidth=1,
                label=label, **kw)


class FigureSaver:
    """Save a figure as PNG and PDF in ``fig_dir`` and its plotted data as CSV in ``data_dir``."""

    def __init__(self, fig_dir: Path, data_dir: Path, overwrite: bool = False):
        self.fig_dir, self.data_dir, self.overwrite = Path(fig_dir), Path(data_dir), overwrite
        self.manifest: list[dict] = []

    def save(self, fig: plt.Figure, name: str, data: pd.DataFrame | None = None) -> list[Path]:
        paths = [self.fig_dir / f"{name}.png", self.fig_dir / f"{name}.pdf"]
        data_path = self.data_dir / f"{name}.csv"
        check_overwrite(paths + ([data_path] if data is not None else []), self.overwrite)
        for p in paths:
            fig.savefig(p)
        if data is not None:
            data.to_csv(data_path, index=False)
        plt.close(fig)
        self.manifest.append({"name": name, "png": paths[0].name, "pdf": paths[1].name,
                              "data": data_path.name if data is not None else None})
        return paths


def nice_note(ax: plt.Axes, text: str, loc: str = "lower right") -> None:
    """Small italic note inside an axes (e.g. 'higher = more disclosure')."""
    xy = {"lower right": (0.98, 0.04, "right", "bottom"), "upper left": (0.02, 0.96, "left", "top"),
          "upper right": (0.98, 0.96, "right", "top"), "lower left": (0.02, 0.04, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8,
            style="italic", color="#444444")

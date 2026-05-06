"""
Generate figures for A2.2.

Figure 1:
    Runtime scaling of greedy vs exact Z3 baseline on a log-log plot.

Run from repo root:

    python experiments/make_figures.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MAIN_RESULTS = PROJECT_ROOT / "results" / "main_results.csv"

FIGURE1_PDF = PROJECT_ROOT / "results" / "figure1_runtime_scaling.pdf"
FIGURE1_PNG = PROJECT_ROOT / "results" / "figure1_runtime_scaling.png"


def load_main_results() -> pd.DataFrame:
    """Load the main experiment results."""

    if not MAIN_RESULTS.exists():
        raise FileNotFoundError(f"Missing {MAIN_RESULTS}")

    return pd.read_csv(MAIN_RESULTS)


def make_runtime_scaling_figure(df: pd.DataFrame) -> None:
    """
    Make Figure 1: runtime scaling, greedy vs exact, log-log.

    One panel per dataset.
    """

    summary = (
        df.groupby(["dataset", "num_constraints"])
        .agg(
            greedy_mean=("greedy_runtime_sec", "mean"),
            greedy_std=("greedy_runtime_sec", "std"),
            exact_mean=("exact_runtime_sec", "mean"),
            exact_std=("exact_runtime_sec", "std"),
        )
        .reset_index()
    )

    datasets = ["A", "B", "C"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.6),
        sharey=True,
    )

    for ax, dataset in zip(axes, datasets):
        part = summary[summary["dataset"] == dataset].sort_values(
            "num_constraints"
        )

        ax.errorbar(
            part["num_constraints"],
            part["greedy_mean"],
            yerr=part["greedy_std"],
            marker="o",
            capsize=3,
            label="Greedy",
        )

        ax.errorbar(
            part["num_constraints"],
            part["exact_mean"],
            yerr=part["exact_std"],
            marker="s",
            capsize=3,
            label="Exact Z3",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(f"Dataset {dataset}")
        ax.set_xlabel("Number of constraints |C|")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("Runtime seconds")
    axes[-1].legend(loc="best")

    fig.suptitle("Figure 1: Runtime scaling of greedy and exact baselines")
    fig.tight_layout()

    FIGURE1_PDF.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE1_PDF, bbox_inches="tight")
    fig.savefig(FIGURE1_PNG, dpi=300, bbox_inches="tight")

    print(f"Saved {FIGURE1_PDF}")
    print(f"Saved {FIGURE1_PNG}")


def main() -> None:
    df = load_main_results()
    make_runtime_scaling_figure(df)


if __name__ == "__main__":
    main()

"""
Generate figures for A2.2.

Figure 1: Runtime scaling of greedy vs exact Z3 baseline.
Figure 2: Runtime ratio and MUS discovery count.

Notes:
- Structural explanation for ratio = 1.0 is described in the Results text.
- Brute-force vs Z3 agreement is reported as an inline sanity check or LaTeX table,
  not as a generated figure.

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
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURE1_PDF = FIGURES_DIR / "figure1_runtime_scaling.pdf"
FIGURE1_PNG = FIGURES_DIR / "figure1_runtime_scaling.png"

FIGURE2_PDF = FIGURES_DIR / "figure2_runtime_ratio_mus_count.pdf"
FIGURE2_PNG = FIGURES_DIR / "figure2_runtime_ratio_mus_count.png"


def load_main_results() -> pd.DataFrame:
    """Load the main experiment results."""

    if not MAIN_RESULTS.exists():
        raise FileNotFoundError(f"Missing {MAIN_RESULTS}")

    return pd.read_csv(MAIN_RESULTS)


def make_runtime_scaling_figure(df: pd.DataFrame) -> None:
    """
    Figure 1: absolute runtime scaling.

    This figure shows how greedy and exact Z3 runtime change as |C| grows.
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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

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
        ax.set_xlabel(r"Number of constraints $|C|$")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("Runtime seconds")
    axes[-1].legend(loc="best")

    fig.suptitle("Figure 1: Runtime scaling of greedy and exact baselines")
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE1_PDF, bbox_inches="tight")
    fig.savefig(FIGURE1_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE1_PDF}")
    print(f"Saved {FIGURE1_PNG}")


def make_runtime_ratio_and_mus_figure(df: pd.DataFrame) -> None:
    """
    Figure 2: runtime ratio and MUS discovery count.

    This figure explains why Dataset B is slower for lazy greedy:
    it discovers more MUSes before halting.
    """

    summary = (
        df.groupby(["dataset", "num_constraints"])
        .agg(
            greedy_mean=("greedy_runtime_sec", "mean"),
            exact_mean=("exact_runtime_sec", "mean"),
            mus_count_mean=("num_discovered_muses", "mean"),
            mus_count_std=("num_discovered_muses", "std"),
        )
        .reset_index()
    )

    summary["runtime_ratio"] = (
        summary["greedy_mean"] / summary["exact_mean"]
    )

    datasets = ["A", "B", "C"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1, ax2 = axes

    for dataset in datasets:
        part = summary[summary["dataset"] == dataset].sort_values(
            "num_constraints"
        )

        ax1.plot(
            part["num_constraints"],
            part["runtime_ratio"],
            marker="o",
            label=f"Dataset {dataset}",
        )

        ax2.errorbar(
            part["num_constraints"],
            part["mus_count_mean"],
            yerr=part["mus_count_std"],
            marker="o",
            capsize=3,
            label=f"Dataset {dataset}",
        )

    ax1.axhline(1.0, linestyle="--", linewidth=1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Number of constraints $|C|$")
    ax1.set_ylabel("Greedy runtime / exact runtime")
    ax1.set_title("(a) Relative runtime")
    ax1.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Number of constraints $|C|$")
    ax2.set_ylabel("Discovered MUS count")
    ax2.set_title("(b) MUSes discovered by lazy greedy")
    ax2.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Figure 2: Lazy greedy overhead is driven by MUS discovery")
    fig.tight_layout()

    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE2_PDF, bbox_inches="tight")
    fig.savefig(FIGURE2_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE2_PDF}")
    print(f"Saved {FIGURE2_PNG}")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    main_df = load_main_results()

    make_runtime_scaling_figure(main_df)
    make_runtime_ratio_and_mus_figure(main_df)


if __name__ == "__main__":
    main()

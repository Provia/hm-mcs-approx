"""
Generate figures for A2.2.

Figure 1: Runtime scaling of greedy vs exact Z3 baseline, log-log.
Figure 2: MUS discovery and greedy iterations vs |C|.
Figure 3: Per-dataset summary table.
Figure 4: Brute-force vs Z3 agreement sanity confirmation.

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
BRUTE_RESULTS = PROJECT_ROOT / "results" / "brute_sanity_results.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURE1_PDF = FIGURES_DIR / "figure1_runtime_scaling.pdf"
FIGURE1_PNG = FIGURES_DIR / "figure1_runtime_scaling.png"

FIGURE2_PDF = FIGURES_DIR / "figure2_mus_discovery.pdf"
FIGURE2_PNG = FIGURES_DIR / "figure2_mus_discovery.png"

FIGURE3_PDF = FIGURES_DIR / "figure3_dataset_summary.pdf"
FIGURE3_PNG = FIGURES_DIR / "figure3_dataset_summary.png"

FIGURE4_PDF = FIGURES_DIR / "figure4_brute_z3_agreement.pdf"
FIGURE4_PNG = FIGURES_DIR / "figure4_brute_z3_agreement.png"


def load_main_results() -> pd.DataFrame:
    """Load the main experiment results."""

    if not MAIN_RESULTS.exists():
        raise FileNotFoundError(f"Missing {MAIN_RESULTS}")

    return pd.read_csv(MAIN_RESULTS)


def load_brute_results() -> pd.DataFrame:
    """Load brute-force sanity results."""

    if not BRUTE_RESULTS.exists():
        raise FileNotFoundError(f"Missing {BRUTE_RESULTS}")

    return pd.read_csv(BRUTE_RESULTS)


def make_runtime_scaling_figure(df: pd.DataFrame) -> None:
    """
    Figure 1: runtime scaling, greedy vs exact, log-log.

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
        ax.set_xlabel(r"Number of constraints $|C|$")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("Runtime seconds")
    axes[-1].legend(loc="best")

    fig.suptitle("Figure 1: Runtime scaling of greedy and exact baselines")
    fig.tight_layout()

    fig.savefig(FIGURE1_PDF, bbox_inches="tight")
    fig.savefig(FIGURE1_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE1_PDF}")
    print(f"Saved {FIGURE1_PNG}")


def make_mus_discovery_figure(df: pd.DataFrame) -> None:
    """
    Figure 2: MUS discovery and greedy iterations vs |C|.

    Left panel:
        discovered MUS count and greedy iteration count

    Right panel:
        mean discovered MUS size
    """

    summary = (
        df.groupby(["dataset", "num_constraints"])
        .agg(
            mus_count_mean=("num_discovered_muses", "mean"),
            mus_count_std=("num_discovered_muses", "std"),
            iter_mean=("greedy_iterations", "mean"),
            iter_std=("greedy_iterations", "std"),
            mus_size_mean=("avg_discovered_mus_size", "mean"),
            mus_size_std=("avg_discovered_mus_size", "std"),
        )
        .reset_index()
    )

    datasets = ["A", "B", "C"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 3.8),
    )

    ax1, ax2 = axes

    for dataset in datasets:
        part = summary[summary["dataset"] == dataset].sort_values(
            "num_constraints"
        )

        ax1.errorbar(
            part["num_constraints"],
            part["mus_count_mean"],
            yerr=part["mus_count_std"],
            marker="o",
            capsize=3,
            label=f"Dataset {dataset}: MUSes",
        )

        ax1.errorbar(
            part["num_constraints"],
            part["iter_mean"],
            yerr=part["iter_std"],
            marker="s",
            capsize=3,
            linestyle="--",
            label=f"Dataset {dataset}: iterations",
        )

        ax2.errorbar(
            part["num_constraints"],
            part["mus_size_mean"],
            yerr=part["mus_size_std"],
            marker="o",
            capsize=3,
            label=f"Dataset {dataset}",
        )

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Number of constraints $|C|$")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) MUS discovery and greedy iterations")
    ax1.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax1.legend(loc="best", fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Number of constraints $|C|$")
    ax2.set_ylabel("Mean discovered MUS size")
    ax2.set_title("(b) Mean discovered MUS size")
    ax2.grid(True, which="both", linewidth=0.5, alpha=0.4)
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Figure 2: MUS discovery characteristics of lazy greedy")
    fig.tight_layout()

    fig.savefig(FIGURE2_PDF, bbox_inches="tight")
    fig.savefig(FIGURE2_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE2_PDF}")
    print(f"Saved {FIGURE2_PNG}")


def make_dataset_summary_table(df: pd.DataFrame) -> None:
    """
    Figure 3: per-dataset summary table.

    This table summarises approximation quality, runtime, MUS discovery,
    and overlap across all sizes and seeds.
    """

    summary = (
        df.groupby("dataset")
        .agg(
            instances=("approximation_ratio", "count"),
            mean_ratio=("approximation_ratio", "mean"),
            max_ratio=("approximation_ratio", "max"),
            failure_rate_2x=("failure_2x", "mean"),
            greedy_runtime=("greedy_runtime_sec", "mean"),
            exact_runtime=("exact_runtime_sec", "mean"),
            muses=("num_discovered_muses", "mean"),
            iterations=("greedy_iterations", "mean"),
            overlap=("overlap_measure", "mean"),
        )
        .reset_index()
    )

    display = summary.copy()
    display["mean_ratio"] = display["mean_ratio"].map("{:.3f}".format)
    display["max_ratio"] = display["max_ratio"].map("{:.3f}".format)
    display["failure_rate_2x"] = display["failure_rate_2x"].map("{:.3f}".format)
    display["greedy_runtime"] = display["greedy_runtime"].map("{:.4f}".format)
    display["exact_runtime"] = display["exact_runtime"].map("{:.4f}".format)
    display["muses"] = display["muses"].map("{:.2f}".format)
    display["iterations"] = display["iterations"].map("{:.2f}".format)
    display["overlap"] = display["overlap"].map("{:.3f}".format)

    display = display.rename(
        columns={
            "dataset": "Dataset",
            "instances": "n",
            "mean_ratio": "Mean ratio",
            "max_ratio": "Max ratio",
            "failure_rate_2x": "Failure >2x",
            "greedy_runtime": "Greedy sec",
            "exact_runtime": "Exact sec",
            "muses": "Mean MUSes",
            "iterations": "Mean iters",
            "overlap": "Overlap",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    ax.set_title("Figure 3: Per-dataset summary of approximation and structure")

    fig.tight_layout()
    fig.savefig(FIGURE3_PDF, bbox_inches="tight")
    fig.savefig(FIGURE3_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE3_PDF}")
    print(f"Saved {FIGURE3_PNG}")


def make_brute_z3_agreement_figure(brute_df: pd.DataFrame) -> None:
    """
    Figure 4: brute-force vs Z3 agreement.

    The figure compares optimal weights from exhaustive brute force and Z3
    on tiny instances.
    """

    summary = (
        brute_df.groupby("dataset")
        .agg(
            brute_weight=("brute_weight", "mean"),
            exact_weight=("exact_weight", "mean"),
            match_rate=("weights_match", "mean"),
            checked_subsets=("brute_checked_subsets", "mean"),
        )
        .reset_index()
    )

    x = range(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.8))

    ax.bar(
        [i - width / 2 for i in x],
        summary["brute_weight"],
        width=width,
        label="Brute force",
    )

    ax.bar(
        [i + width / 2 for i in x],
        summary["exact_weight"],
        width=width,
        label="Exact Z3",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["dataset"])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean optimal correction weight")
    ax.set_title("Figure 4: Brute-force and Z3 exact baseline agreement")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    for i, row in summary.iterrows():
        ax.text(
            i,
            max(row["brute_weight"], row["exact_weight"]),
            f"match={row['match_rate']:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(FIGURE4_PDF, bbox_inches="tight")
    fig.savefig(FIGURE4_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {FIGURE4_PDF}")
    print(f"Saved {FIGURE4_PNG}")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    main_df = load_main_results()
    brute_df = load_brute_results()

    make_runtime_scaling_figure(main_df)
    make_mus_discovery_figure(main_df)
    make_dataset_summary_table(main_df)
    make_brute_z3_agreement_figure(brute_df)


if __name__ == "__main__":
    main()

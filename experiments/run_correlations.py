"""
Re-pivoted analysis for A2.2 Phase 3.2.

Because approximation ratio is constant at 1.0 on the current synthetic
generators, this script focuses on:

- runtime scaling
- greedy iteration scaling
- MUS-discovery scaling
- weight statistics
- brute-force versus Z3 agreement

Run from repo root:

    python experiments/run_correlations.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


MAIN_RESULTS = PROJECT_ROOT / "results" / "main_results.csv"
BRUTE_RESULTS = PROJECT_ROOT / "results" / "brute_sanity_results.csv"

SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "summary_by_dataset_size.csv"
CORRELATIONS_OUTPUT = PROJECT_ROOT / "results" / "correlations.csv"
BRUTE_SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "brute_sanity_summary.csv"


def safe_corr(df, x_col, y_col, method):
    """Compute Pearson or Spearman correlation without requiring scipy."""

    usable = df[[x_col, y_col]].dropna()

    if len(usable) < 2:
        return math.nan

    if usable[x_col].nunique() <= 1:
        return math.nan

    if usable[y_col].nunique() <= 1:
        return math.nan

    if method == "pearson":
        x = usable[x_col].astype(float).to_numpy()
        y = usable[y_col].astype(float).to_numpy()

    elif method == "spearman":
        x = usable[x_col].rank(method="average").astype(float).to_numpy()
        y = usable[y_col].rank(method="average").astype(float).to_numpy()

    else:
        raise ValueError(f"unknown correlation method: {method}")

    if len(x) < 2:
        return math.nan

    value = np.corrcoef(x, y)[0, 1]

    if np.isnan(value):
        return math.nan

    return float(value)


def log_log_slope(group, x_col, y_col):
    """Fit log y against log x and return the slope."""

    usable = group[[x_col, y_col]].dropna()
    usable = usable[(usable[x_col] > 0) & (usable[y_col] > 0)]

    if len(usable) < 2:
        return math.nan

    grouped = (
        usable.groupby(x_col)[y_col]
        .mean()
        .reset_index()
    )

    if len(grouped) < 2:
        return math.nan

    x = np.log(grouped[x_col].to_numpy(dtype=float))
    y = np.log(grouped[y_col].to_numpy(dtype=float))

    slope, _intercept = np.polyfit(x, y, deg=1)
    return float(slope)


def make_summary(main_df):
    """Create grouped summary by dataset and size."""

    return (
        main_df.groupby(["dataset", "num_constraints"])
        .agg(
            n=("approximation_ratio", "count"),
            exact_success_rate=("exact_success", "mean"),
            mean_ratio=("approximation_ratio", "mean"),
            max_ratio=("approximation_ratio", "max"),
            failure_rate_2x=("failure_2x", "mean"),
            mean_greedy_runtime=("greedy_runtime_sec", "mean"),
            mean_exact_runtime=("exact_runtime_sec", "mean"),
            runtime_ratio_greedy_over_exact=(
                "greedy_runtime_sec",
                lambda s: float(s.mean()),
            ),
            mean_greedy_iterations=("greedy_iterations", "mean"),
            mean_discovered_muses=("num_discovered_muses", "mean"),
            mean_mus_size=("avg_discovered_mus_size", "mean"),
            mean_overlap=("overlap_measure", "mean"),
            mean_weight_mean=("weight_mean", "mean"),
            mean_weight_variance=("weight_variance", "mean"),
        )
        .reset_index()
    )


def fix_runtime_ratio(summary):
    """Compute greedy runtime divided by exact runtime on summary rows."""

    summary = summary.copy()

    summary["runtime_ratio_greedy_over_exact"] = (
        summary["mean_greedy_runtime"] / summary["mean_exact_runtime"]
    )

    return summary


def make_correlations(main_df):
    """
    Create correlation table.

    Approximation ratio is intentionally not used as the main target because it
    has zero variance in the current experiment.
    """

    rows = []

    targets = [
        "greedy_runtime_sec",
        "exact_runtime_sec",
        "num_discovered_muses",
        "greedy_iterations",
        "weight_variance",
    ]

    features = [
        "num_constraints",
        "conflict_density",
        "overlap_measure",
        "avg_discovered_mus_size",
        "weight_mean",
        "weight_variance",
    ]

    for dataset, group in main_df.groupby("dataset"):
        for target in targets:
            for feature in features:
                if target == feature:
                    continue

                for method in ["pearson", "spearman"]:
                    rows.append(
                        {
                            "analysis_type": "correlation",
                            "dataset": dataset,
                            "target": target,
                            "feature": feature,
                            "method": method,
                            "value": safe_corr(group, feature, target, method),
                            "note": "",
                        }
                    )

        for runtime_col in ["greedy_runtime_sec", "exact_runtime_sec"]:
            rows.append(
                {
                    "analysis_type": "log_log_slope",
                    "dataset": dataset,
                    "target": runtime_col,
                    "feature": "num_constraints",
                    "method": "log-log linear fit",
                    "value": log_log_slope(
                        group,
                        x_col="num_constraints",
                        y_col=runtime_col,
                    ),
                    "note": "Slope of log runtime against log number of constraints.",
                }
            )

        for structural_col in ["num_discovered_muses", "greedy_iterations"]:
            rows.append(
                {
                    "analysis_type": "log_log_slope",
                    "dataset": dataset,
                    "target": structural_col,
                    "feature": "num_constraints",
                    "method": "log-log linear fit",
                    "value": log_log_slope(
                        group,
                        x_col="num_constraints",
                        y_col=structural_col,
                    ),
                    "note": "Slope of log structural count against log number of constraints.",
                }
            )

        rows.append(
            {
                "analysis_type": "constant_ratio_check",
                "dataset": dataset,
                "target": "approximation_ratio",
                "feature": "all_features",
                "method": "variance check",
                "value": float(group["approximation_ratio"].var(ddof=0)),
                "note": "Approximation ratio has zero variance, so ratio correlations are not meaningful.",
            }
        )

    return pd.DataFrame(rows)


def make_brute_summary(brute_df):
    """Summarise brute-force and Z3 agreement."""

    return (
        brute_df.groupby("dataset")
        .agg(
            n=("weights_match", "count"),
            all_match=("weights_match", "all"),
            match_rate=("weights_match", "mean"),
            mean_brute_weight=("brute_weight", "mean"),
            mean_exact_weight=("exact_weight", "mean"),
            mean_checked_subsets=("brute_checked_subsets", "mean"),
        )
        .reset_index()
    )


def main():
    if not MAIN_RESULTS.exists():
        raise FileNotFoundError(f"Missing {MAIN_RESULTS}")

    if not BRUTE_RESULTS.exists():
        raise FileNotFoundError(f"Missing {BRUTE_RESULTS}")

    main_df = pd.read_csv(MAIN_RESULTS)
    brute_df = pd.read_csv(BRUTE_RESULTS)

    summary = fix_runtime_ratio(make_summary(main_df))
    correlations = make_correlations(main_df)
    brute_summary = make_brute_summary(brute_df)

    summary.to_csv(SUMMARY_OUTPUT, index=False)
    correlations.to_csv(CORRELATIONS_OUTPUT, index=False)
    brute_summary.to_csv(BRUTE_SUMMARY_OUTPUT, index=False)

    print(f"Saved {SUMMARY_OUTPUT}")
    print(f"Saved {CORRELATIONS_OUTPUT}")
    print(f"Saved {BRUTE_SUMMARY_OUTPUT}")

    print()
    print("Dataset and size summary:")
    print(summary.to_string(index=False))

    print()
    print("Brute sanity summary:")
    print(brute_summary.to_string(index=False))

    print()
    print("Log-log runtime slopes:")
    slope_rows = correlations[
        correlations["analysis_type"].eq("log_log_slope")
        & correlations["target"].isin(["greedy_runtime_sec", "exact_runtime_sec"])
    ]
    print(
        slope_rows[
            ["dataset", "target", "feature", "value"]
        ].to_string(index=False)
    )

    print()
    print("Approximation-ratio variance check:")
    ratio_rows = correlations[
        correlations["analysis_type"].eq("constant_ratio_check")
    ]
    print(
        ratio_rows[
            ["dataset", "target", "value", "note"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

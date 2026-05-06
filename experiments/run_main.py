"""
Main experiment runner for A2.2.

This script runs the full Phase 3 experiment:

    Dataset A/B/C
    sizes = 10, 20, 50, 100, 200
    seeds = 0..19

It also produces a brute-force sanity artefact on tiny instances, comparing
the Z3 exact baseline against exhaustive search.

Run from repo root:

    python experiments/run_main.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.brute import minimum_weight_mcs_bruteforce
from src.exact import minimum_weight_mcs_z3
from src.generator import (
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.metrics import evaluate_instance


DATASETS = ["A", "B", "C"]
SIZES = [10, 20, 50, 100, 200]
SEEDS = range(20)


def make_instance(dataset: str, size: int, seed: int):
    """Generate one synthetic instance."""

    if dataset == "A":
        return generate_dataset_a(
            size=size,
            seed=seed,
        )

    if dataset == "B":
        return generate_dataset_b(
            size=size,
            seed=seed,
            num_conflicting_types=4,
        )

    if dataset == "C":
        return generate_dataset_c(
            size=size,
            seed=seed,
            max_depth=3,
        )

    raise ValueError(f"unknown dataset: {dataset}")


def run_main_experiment(
    exact_timeout_ms: int | None = 30_000,
) -> pd.DataFrame:
    """Run the full main experiment grid."""

    rows = []
    total = len(DATASETS) * len(SIZES) * len(list(SEEDS))
    current = 0
    start_all = time.perf_counter()

    for dataset in DATASETS:
        for size in SIZES:
            for seed in SEEDS:
                current += 1

                instance = make_instance(
                    dataset=dataset,
                    size=size,
                    seed=seed,
                )

                result = evaluate_instance(
                    instance,
                    exact_timeout_ms=exact_timeout_ms,
                    run_exact=True,
                    extra_metadata={
                        "seed": seed,
                    },
                )

                rows.append(result.__dict__)

                print(
                    f"[{current:>3}/{total}] "
                    f"dataset={dataset} "
                    f"size={size:>3} "
                    f"seed={seed:>2} "
                    f"ratio={result.approximation_ratio:.3f} "
                    f"greedy_w={result.greedy_weight:.1f} "
                    f"exact_w={result.exact_weight:.1f} "
                    f"greedy_t={result.greedy_runtime_sec:.4f}s "
                    f"exact_t={result.exact_runtime_sec:.4f}s "
                    f"muses={result.num_discovered_muses} "
                    f"iters={result.greedy_iterations} "
                    f"overlap={result.overlap_measure:.3f} "
                    f"exact_success={result.exact_success}"
                )

    elapsed_all = time.perf_counter() - start_all
    print()
    print(f"Main experiment finished in {elapsed_all:.2f}s")

    return pd.DataFrame(rows)


def run_brute_sanity_check() -> pd.DataFrame:
    """
    Compare Z3 exact baseline with brute force on tiny instances.

    This is a small experiment artefact, separate from the main scaling run.
    """

    rows = []

    for dataset in DATASETS:
        for seed in range(5):
            instance = make_instance(
                dataset=dataset,
                size=10,
                seed=seed,
            )

            brute = minimum_weight_mcs_bruteforce(
                instance.constraints,
                max_constraints=12,
            )

            exact = minimum_weight_mcs_z3(instance.constraints)

            weights_match = abs(brute.total_weight - exact.total_weight) < 1e-9

            rows.append(
                {
                    "dataset": dataset,
                    "size": 10,
                    "seed": seed,
                    "brute_weight": brute.total_weight,
                    "exact_weight": exact.total_weight,
                    "weights_match": weights_match,
                    "brute_cardinality": brute.cardinality,
                    "exact_cardinality": exact.cardinality,
                    "brute_checked_subsets": brute.checked_subsets,
                    "brute_selected_indices": str(brute.selected_indices),
                    "exact_selected_indices": str(exact.selected_indices),
                }
            )

            print(
                f"[brute sanity] "
                f"dataset={dataset} "
                f"seed={seed} "
                f"brute_w={brute.total_weight:.1f} "
                f"exact_w={exact.total_weight:.1f} "
                f"match={weights_match}"
            )

    return pd.DataFrame(rows)


def print_main_summary(df: pd.DataFrame) -> None:
    """Print aggregate main experiment summary."""

    summary = (
        df.groupby(["dataset", "num_constraints"])
        .agg(
            n=("approximation_ratio", "count"),
            exact_success_rate=("exact_success", "mean"),
            mean_ratio=("approximation_ratio", "mean"),
            max_ratio=("approximation_ratio", "max"),
            failure_rate_2x=("failure_2x", "mean"),
            mean_greedy_runtime=("greedy_runtime_sec", "mean"),
            mean_exact_runtime=("exact_runtime_sec", "mean"),
            mean_greedy_iterations=("greedy_iterations", "mean"),
            mean_muses=("num_discovered_muses", "mean"),
            mean_mus_size=("avg_discovered_mus_size", "mean"),
            mean_overlap=("overlap_measure", "mean"),
            mean_weight_variance=("weight_variance", "mean"),
        )
        .reset_index()
    )

    print()
    print("Main summary:")
    print(summary.to_string(index=False))


def print_brute_summary(df: pd.DataFrame) -> None:
    """Print brute-force sanity summary."""

    summary = (
        df.groupby("dataset")
        .agg(
            n=("weights_match", "count"),
            all_match=("weights_match", "all"),
            mean_brute_weight=("brute_weight", "mean"),
            mean_exact_weight=("exact_weight", "mean"),
            mean_checked_subsets=("brute_checked_subsets", "mean"),
        )
        .reset_index()
    )

    print()
    print("Brute-force sanity summary:")
    print(summary.to_string(index=False))


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    main_df = run_main_experiment(
        exact_timeout_ms=30_000,
    )

    main_output = results_dir / "main_results.csv"
    main_df.to_csv(main_output, index=False)

    print()
    print(f"Saved main results to {main_output}")

    print_main_summary(main_df)

    brute_df = run_brute_sanity_check()

    brute_output = results_dir / "brute_sanity_results.csv"
    brute_df.to_csv(brute_output, index=False)

    print()
    print(f"Saved brute sanity results to {brute_output}")

    print_brute_summary(brute_df)


if __name__ == "__main__":
    main()

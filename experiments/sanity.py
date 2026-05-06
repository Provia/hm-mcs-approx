"""
Small end-to-end sanity experiment.

This script runs a small grid over Dataset A/B/C before the full Phase 3
experiment. It checks whether the pipeline works end-to-end:

    generator -> greedy -> exact Z3 -> metrics -> CSV

Run from repo root:

    python experiments/sanity.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Allow `python experiments/sanity.py` from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import (
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.metrics import evaluate_instance


def make_instance(dataset: str, size: int, seed: int):
    """Generate one instance for the requested dataset."""

    if dataset == "A":
        return generate_dataset_a(size=size, seed=seed)

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


def main() -> None:
    datasets = ["A", "B", "C"]
    sizes = [10, 15, 20]
    seeds = range(5)

    rows = []

    for dataset in datasets:
        for size in sizes:
            for seed in seeds:
                instance = make_instance(dataset, size, seed)

                result = evaluate_instance(
                    instance,
                    exact_timeout_ms=10_000,
                    run_exact=True,
                    extra_metadata={
                        "seed": seed,
                    },
                )

                rows.append(result.__dict__)

                print(
                    f"dataset={dataset} "
                    f"size={size:>3} "
                    f"seed={seed:>2} "
                    f"ratio={result.approximation_ratio:.3f} "
                    f"greedy_w={result.greedy_weight:.1f} "
                    f"exact_w={result.exact_weight:.1f} "
                    f"greedy_t={result.greedy_runtime_sec:.4f}s "
                    f"exact_t={result.exact_runtime_sec:.4f}s "
                    f"muses={result.num_discovered_muses} "
                    f"overlap={result.overlap_measure:.3f} "
                    f"exact_success={result.exact_success}"
                )

    df = pd.DataFrame(rows)

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "sanity_results.csv"
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved sanity results to {output_path}")
    print()

    summary = (
        df.groupby(["dataset", "num_constraints"])
        .agg(
            n=("approximation_ratio", "count"),
            mean_ratio=("approximation_ratio", "mean"),
            max_ratio=("approximation_ratio", "max"),
            failure_rate_2x=("failure_2x", "mean"),
            mean_greedy_runtime=("greedy_runtime_sec", "mean"),
            mean_exact_runtime=("exact_runtime_sec", "mean"),
            mean_muses=("num_discovered_muses", "mean"),
            mean_overlap=("overlap_measure", "mean"),
            exact_success_rate=("exact_success", "mean"),
        )
        .reset_index()
    )

    print("Summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

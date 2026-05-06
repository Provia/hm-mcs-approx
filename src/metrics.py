"""
Evaluation metrics for greedy MCS experiments.

This module runs the greedy approximation algorithm and the exact Z3 baseline
on generated HM-style constraint instances, then records approximation quality,
runtime, and structural features.

It is intended to produce rows that can be saved to CSV in Phase 3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
import time
from typing import Any, Iterable

import pandas as pd

from .constraints import EqConstraint
from .exact import ExactResult, minimum_weight_mcs_z3
from .generator import Instance
from .greedy import GreedyResult, minimum_weight_mcs_greedy


@dataclass(frozen=True)
class EvaluationResult:
    """One row of evaluation output."""

    instance_name: str
    dataset: str
    num_constraints: int

    greedy_weight: float
    exact_weight: float
    approximation_ratio: float

    greedy_runtime_sec: float
    exact_runtime_sec: float
    exact_success: bool
    exact_error: str

    failure_2x: bool

    greedy_cardinality: int
    exact_cardinality: float
    greedy_iterations: int

    num_discovered_muses: int
    avg_discovered_mus_size: float
    max_discovered_mus_size: int
    conflict_density: float
    overlap_measure: float

    weight_min: float
    weight_max: float
    weight_mean: float
    weight_variance: float

    target_size: Any = None
    seed: Any = None
    num_conflicting_types: Any = None
    max_depth: Any = None


def approximation_ratio(
    greedy_weight: float,
    exact_weight: float,
    exact_success: bool = True,
) -> float:
    """Return greedy_weight divided by exact_weight, with safe edge cases."""

    if not exact_success:
        return math.nan

    if exact_weight == 0:
        if greedy_weight == 0:
            return 1.0
        return math.inf

    return greedy_weight / exact_weight


def failure_2x(ratio: float) -> bool:
    """Return True if the approximation ratio is greater than 2."""

    if math.isnan(ratio):
        return False

    return ratio > 2.0


def weight_statistics(constraints: Iterable[EqConstraint]) -> dict[str, float]:
    """Compute simple population statistics over constraint weights."""

    weights = [float(constraint.weight) for constraint in constraints]

    if not weights:
        return {
            "weight_min": 0.0,
            "weight_max": 0.0,
            "weight_mean": 0.0,
            "weight_variance": 0.0,
        }

    mean = sum(weights) / len(weights)
    variance = sum((weight - mean) ** 2 for weight in weights) / len(weights)

    return {
        "weight_min": min(weights),
        "weight_max": max(weights),
        "weight_mean": mean,
        "weight_variance": variance,
    }


def average_pairwise_jaccard_overlap(
    muses: Iterable[tuple[int, ...]],
) -> float:
    """
    Compute average pairwise Jaccard overlap over discovered MUSes.

    For two MUSes A and B:
        overlap = size of intersection divided by size of union

    If there are fewer than two MUSes, the overlap is defined as 0.
    """

    mus_sets = [
        set(mus)
        for mus in muses
    ]

    if len(mus_sets) < 2:
        return 0.0

    overlaps: list[float] = []

    for left, right in combinations(mus_sets, 2):
        union = left | right

        if not union:
            overlaps.append(0.0)
            continue

        overlaps.append(len(left & right) / len(union))

    return sum(overlaps) / len(overlaps)


def discovered_mus_statistics(
    muses: Iterable[tuple[int, ...]],
    num_constraints: int,
) -> dict[str, float | int]:
    """Compute statistics from MUSes discovered by the greedy algorithm."""

    mus_list = list(muses)
    mus_sizes = [
        len(mus)
        for mus in mus_list
    ]

    num_muses = len(mus_list)

    if mus_sizes:
        avg_size = sum(mus_sizes) / len(mus_sizes)
        max_size = max(mus_sizes)
    else:
        avg_size = 0.0
        max_size = 0

    if num_constraints == 0:
        density = 0.0
    else:
        density = num_muses / num_constraints

    return {
        "num_discovered_muses": num_muses,
        "avg_discovered_mus_size": avg_size,
        "max_discovered_mus_size": max_size,
        "conflict_density": density,
        "overlap_measure": average_pairwise_jaccard_overlap(mus_list),
    }


def _time_greedy(constraints: list[EqConstraint]) -> tuple[GreedyResult, float]:
    """Run greedy and return result plus elapsed wall-clock time."""

    start = time.perf_counter()
    result = minimum_weight_mcs_greedy(constraints)
    elapsed = time.perf_counter() - start

    return result, elapsed


def _time_exact(
    constraints: list[EqConstraint],
    timeout_ms: int | None,
) -> tuple[ExactResult | None, float, str]:
    """
    Run exact Z3 baseline.

    Returns:
        exact result or None
        elapsed time
        error string, empty on success
    """

    start = time.perf_counter()

    try:
        result = minimum_weight_mcs_z3(
            constraints,
            timeout_ms=timeout_ms,
        )
        elapsed = time.perf_counter() - start
        return result, elapsed, ""

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return None, elapsed, f"{type(exc).__name__}: {exc}"


def evaluate_instance(
    instance: Instance,
    exact_timeout_ms: int | None = None,
    run_exact: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """
    Evaluate one generated instance.

    Args:
        instance: Generated benchmark instance.
        exact_timeout_ms: Optional Z3 timeout.
        run_exact: If False, skip exact solving and record NaN for exact fields.
        extra_metadata: Optional metadata, such as seed.

    Returns:
        EvaluationResult suitable for CSV export.
    """

    constraints = list(instance.constraints)
    metadata = dict(instance.metadata)

    if extra_metadata:
        metadata.update(extra_metadata)

    greedy_result, greedy_time = _time_greedy(constraints)

    if run_exact:
        exact_result, exact_time, exact_error = _time_exact(
            constraints,
            timeout_ms=exact_timeout_ms,
        )
    else:
        exact_result = None
        exact_time = 0.0
        exact_error = "exact baseline skipped"

    exact_success = exact_result is not None

    if exact_result is None:
        exact_weight = math.nan
        exact_cardinality = math.nan
    else:
        exact_weight = exact_result.total_weight
        exact_cardinality = float(exact_result.cardinality)

    ratio = approximation_ratio(
        greedy_weight=greedy_result.total_weight,
        exact_weight=exact_weight,
        exact_success=exact_success,
    )

    mus_stats = discovered_mus_statistics(
        greedy_result.discovered_muses,
        num_constraints=len(constraints),
    )

    weight_stats = weight_statistics(constraints)

    return EvaluationResult(
        instance_name=instance.name,
        dataset=str(metadata.get("dataset", "unknown")),
        num_constraints=len(constraints),

        greedy_weight=greedy_result.total_weight,
        exact_weight=exact_weight,
        approximation_ratio=ratio,

        greedy_runtime_sec=greedy_time,
        exact_runtime_sec=exact_time,
        exact_success=exact_success,
        exact_error=exact_error,

        failure_2x=failure_2x(ratio),

        greedy_cardinality=greedy_result.cardinality,
        exact_cardinality=exact_cardinality,
        greedy_iterations=greedy_result.iterations,

        num_discovered_muses=int(mus_stats["num_discovered_muses"]),
        avg_discovered_mus_size=float(mus_stats["avg_discovered_mus_size"]),
        max_discovered_mus_size=int(mus_stats["max_discovered_mus_size"]),
        conflict_density=float(mus_stats["conflict_density"]),
        overlap_measure=float(mus_stats["overlap_measure"]),

        weight_min=weight_stats["weight_min"],
        weight_max=weight_stats["weight_max"],
        weight_mean=weight_stats["weight_mean"],
        weight_variance=weight_stats["weight_variance"],

        target_size=metadata.get("target_size"),
        seed=metadata.get("seed"),
        num_conflicting_types=metadata.get("num_conflicting_types"),
        max_depth=metadata.get("max_depth"),
    )


def evaluate_instances(
    instances: Iterable[Instance],
    exact_timeout_ms: int | None = None,
    run_exact: bool = True,
) -> pd.DataFrame:
    """Evaluate multiple instances and return a pandas DataFrame."""

    rows = [
        asdict(
            evaluate_instance(
                instance,
                exact_timeout_ms=exact_timeout_ms,
                run_exact=run_exact,
            )
        )
        for instance in instances
    ]

    return pd.DataFrame(rows)

import math

import pandas as pd
import pytest

from src.generator import (
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.metrics import (
    approximation_ratio,
    average_pairwise_jaccard_overlap,
    evaluate_instance,
    evaluate_instances,
    failure_2x,
    weight_statistics,
)


def test_approximation_ratio_normal_case():
    assert approximation_ratio(6.0, 3.0) == pytest.approx(2.0)


def test_approximation_ratio_zero_exact_weight():
    assert approximation_ratio(0.0, 0.0) == pytest.approx(1.0)
    assert math.isinf(approximation_ratio(1.0, 0.0))


def test_failure_2x_logic():
    assert not failure_2x(1.0)
    assert not failure_2x(2.0)
    assert failure_2x(2.01)
    assert not failure_2x(math.nan)


def test_average_pairwise_jaccard_overlap():
    muses = [
        (0, 1),
        (1, 2),
        (2, 3),
    ]

    overlap = average_pairwise_jaccard_overlap(muses)

    assert overlap == pytest.approx(2.0 / 9.0)


def test_average_pairwise_jaccard_overlap_with_one_mus():
    assert average_pairwise_jaccard_overlap([(0, 1)]) == 0.0


def test_weight_statistics():
    instance = generate_dataset_b(
        size=10,
        seed=2,
        num_conflicting_types=4,
    )

    stats = weight_statistics(instance.constraints)

    assert stats["weight_min"] > 0
    assert stats["weight_max"] >= stats["weight_min"]
    assert stats["weight_mean"] > 0
    assert stats["weight_variance"] >= 0


def test_evaluate_dataset_a_has_ratio_one():
    instance = generate_dataset_a(size=8, seed=1)

    result = evaluate_instance(instance)

    assert result.dataset == "A"
    assert result.num_constraints == 8
    assert result.exact_success
    assert result.approximation_ratio == pytest.approx(1.0)
    assert not result.failure_2x
    assert result.greedy_weight == pytest.approx(result.exact_weight)


def test_evaluate_dataset_b_contains_structural_features():
    instance = generate_dataset_b(
        size=10,
        seed=2,
        num_conflicting_types=4,
    )

    result = evaluate_instance(instance)

    assert result.dataset == "B"
    assert result.num_constraints == 10
    assert result.num_discovered_muses >= 1
    assert result.avg_discovered_mus_size > 0
    assert result.max_discovered_mus_size >= result.avg_discovered_mus_size
    assert 0.0 <= result.conflict_density <= 1.0
    assert 0.0 <= result.overlap_measure <= 1.0


def test_evaluate_dataset_c_records_runtime_and_weights():
    instance = generate_dataset_c(size=10, seed=3)

    result = evaluate_instance(instance)

    assert result.dataset == "C"
    assert result.greedy_runtime_sec >= 0.0
    assert result.exact_runtime_sec >= 0.0
    assert result.weight_mean > 0.0
    assert result.weight_variance >= 0.0


def test_evaluate_instance_can_skip_exact():
    instance = generate_dataset_a(size=8, seed=1)

    result = evaluate_instance(
        instance,
        run_exact=False,
    )

    assert not result.exact_success
    assert math.isnan(result.exact_weight)
    assert math.isnan(result.approximation_ratio)
    assert result.exact_error == "exact baseline skipped"


def test_evaluate_instances_returns_dataframe():
    instances = [
        generate_dataset_a(size=8, seed=0),
        generate_dataset_b(size=10, seed=1, num_conflicting_types=3),
        generate_dataset_c(size=10, seed=2),
    ]

    df = evaluate_instances(instances)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3

    required_columns = {
        "instance_name",
        "dataset",
        "num_constraints",
        "greedy_weight",
        "exact_weight",
        "approximation_ratio",
        "greedy_runtime_sec",
        "exact_runtime_sec",
        "exact_success",
        "failure_2x",
        "num_discovered_muses",
        "avg_discovered_mus_size",
        "weight_variance",
        "conflict_density",
        "overlap_measure",
    }

    assert required_columns.issubset(set(df.columns))

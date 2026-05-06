from src.generator import (
    generate_all_small,
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.solver import is_sat


def test_dataset_a_is_unsatisfiable():
    instance = generate_dataset_a(size=10, seed=1)

    assert instance.metadata["dataset"] == "A"
    assert len(instance.constraints) == 10
    assert not is_sat(instance.constraints)


def test_dataset_b_is_unsatisfiable():
    instance = generate_dataset_b(size=12, seed=1, num_conflicting_types=4)

    assert instance.metadata["dataset"] == "B"
    assert len(instance.constraints) == 12
    assert not is_sat(instance.constraints)


def test_dataset_c_is_unsatisfiable():
    instance = generate_dataset_c(size=15, seed=1)

    assert instance.metadata["dataset"] == "C"
    assert len(instance.constraints) == 15
    assert not is_sat(instance.constraints)


def test_dataset_a_has_single_planted_conflict_labels():
    instance = generate_dataset_a(size=8, seed=2)

    labels = {constraint.label for constraint in instance.constraints}

    assert "A_conflict_int" in labels
    assert "A_conflict_bool" in labels


def test_dataset_b_has_overlapping_conflict_labels():
    instance = generate_dataset_b(size=9, seed=2, num_conflicting_types=3)

    labels = {constraint.label for constraint in instance.constraints}

    assert "B_overlap_0" in labels
    assert "B_overlap_1" in labels
    assert "B_overlap_2" in labels
    assert instance.metadata["overlap_variable"] == "x_overlap"


def test_dataset_c_uses_positive_depth_based_weights():
    instance = generate_dataset_c(size=10, seed=2)

    weights = [constraint.weight for constraint in instance.constraints]

    assert all(weight > 0 for weight in weights)
    assert max(weights) > 1.0


def test_generators_are_deterministic_with_same_seed():
    first = generate_dataset_c(size=10, seed=123)
    second = generate_dataset_c(size=10, seed=123)

    first_rendered = [str(constraint) for constraint in first.constraints]
    second_rendered = [str(constraint) for constraint in second.constraints]

    assert first_rendered == second_rendered


def test_generate_all_small_returns_three_unsat_instances():
    instances = generate_all_small(seed=0)

    assert len(instances) == 3

    for instance in instances:
        assert not is_sat(instance.constraints)

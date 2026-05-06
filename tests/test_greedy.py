import pytest

from src.constraints import BOOL, INT, EqConstraint, TyVar
from src.exact import minimum_weight_mcs_z3
from src.generator import (
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.greedy import (
    extract_one_mus,
    is_mus,
    minimum_weight_mcs_greedy,
    weighted_greedy_hitting_set,
)
from src.solver import is_sat


def remaining_after_drop(constraints, selected_indices):
    selected = set(selected_indices)

    return [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in selected
    ]


def assert_greedy_repairs_instance(constraints):
    result = minimum_weight_mcs_greedy(constraints)
    remaining = remaining_after_drop(constraints, result.selected_indices)

    assert is_sat(remaining)

    return result


def test_sat_input_returns_empty_correction_subset():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_int", weight=1.0),
    ]

    result = minimum_weight_mcs_greedy(constraints)

    assert result.selected_indices == ()
    assert result.selected_constraints == ()
    assert result.total_weight == 0.0
    assert result.discovered_muses == ()
    assert result.iterations == 0


def test_extract_one_mus_from_simple_conflict():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_int", weight=1.0),
        EqConstraint(a, BOOL, label="a_bool", weight=1.0),
    ]

    mus = extract_one_mus(constraints)

    assert mus == (0, 1)
    assert is_mus(constraints, mus)


def test_weighted_greedy_hitting_set_prefers_shared_constraint():
    muses = [
        (0, 1),
        (0, 2),
    ]

    weights = [
        1.0,
        1.0,
        1.0,
    ]

    selected = weighted_greedy_hitting_set(muses, weights)

    assert selected == (0,)


def test_dataset_a_greedy_matches_exact_weight():
    instance = generate_dataset_a(size=8, seed=1)

    greedy = minimum_weight_mcs_greedy(instance.constraints)
    exact = minimum_weight_mcs_z3(instance.constraints)

    assert greedy.total_weight == pytest.approx(exact.total_weight)
    assert greedy.total_weight == 1.0
    assert is_sat(remaining_after_drop(instance.constraints, greedy.selected_indices))


def test_greedy_repairs_dataset_b():
    instance = generate_dataset_b(
        size=10,
        seed=2,
        num_conflicting_types=4,
    )

    result = assert_greedy_repairs_instance(instance.constraints)

    assert result.total_weight > 0
    assert len(result.discovered_muses) >= 1


def test_greedy_repairs_dataset_c():
    instance = generate_dataset_c(size=10, seed=3)

    result = assert_greedy_repairs_instance(instance.constraints)

    assert result.total_weight > 0
    assert len(result.discovered_muses) >= 1


def test_greedy_weight_is_at_least_exact_on_small_instances():
    instances = [
        generate_dataset_a(size=8, seed=0),
        generate_dataset_b(size=10, seed=1, num_conflicting_types=3),
        generate_dataset_c(size=10, seed=2),
    ]

    for instance in instances:
        greedy = minimum_weight_mcs_greedy(instance.constraints)
        exact = minimum_weight_mcs_z3(instance.constraints)

        assert greedy.total_weight + 1e-9 >= exact.total_weight
        assert is_sat(
            remaining_after_drop(
                instance.constraints,
                greedy.selected_indices,
            )
        )


def test_discovered_muses_are_subset_minimal_unsat():
    instance = generate_dataset_b(
        size=10,
        seed=4,
        num_conflicting_types=4,
    )

    result = minimum_weight_mcs_greedy(instance.constraints)

    assert len(result.discovered_muses) >= 1

    for mus in result.discovered_muses:
        assert is_mus(instance.constraints, mus)


def test_greedy_is_deterministic_on_same_instance():
    instance = generate_dataset_c(size=10, seed=123)

    first = minimum_weight_mcs_greedy(instance.constraints)
    second = minimum_weight_mcs_greedy(instance.constraints)

    assert first.selected_indices == second.selected_indices
    assert first.total_weight == pytest.approx(second.total_weight)
    assert first.discovered_muses == second.discovered_muses

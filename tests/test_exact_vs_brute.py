import pytest

from src.brute import minimum_weight_mcs_bruteforce
from src.constraints import BOOL, INT, EqConstraint, TyList, TyVar
from src.exact import minimum_weight_mcs_z3
from src.generator import (
    generate_dataset_a,
    generate_dataset_b,
    generate_dataset_c,
)
from src.solver import is_sat


def remaining_after_drop(constraints, selected_indices):
    selected = set(selected_indices)

    return [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in selected
    ]


def assert_z3_matches_brute(constraints):
    brute = minimum_weight_mcs_bruteforce(constraints, max_constraints=12)
    exact = minimum_weight_mcs_z3(constraints)

    assert exact.total_weight == pytest.approx(brute.total_weight)

    exact_remaining = remaining_after_drop(
        constraints,
        exact.selected_indices,
    )

    assert is_sat(exact_remaining)


def test_sat_input_has_empty_exact_mcs():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_int", weight=1.0),
    ]

    exact = minimum_weight_mcs_z3(constraints)

    assert exact.selected_indices == ()
    assert exact.total_weight == 0.0


def test_simple_conflict_matches_brute():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_int", weight=1.0),
        EqConstraint(a, BOOL, label="a_bool", weight=1.0),
    ]

    assert_z3_matches_brute(constraints)


def test_weighted_conflict_matches_brute():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="expensive_int", weight=10.0),
        EqConstraint(a, BOOL, label="cheap_bool", weight=1.0),
    ]

    exact = minimum_weight_mcs_z3(constraints)
    brute = minimum_weight_mcs_bruteforce(constraints)

    assert exact.total_weight == pytest.approx(brute.total_weight)
    assert exact.total_weight == 1.0


def test_occurs_check_like_recursive_type_matches_brute():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, TyList(a), label="recursive", weight=3.0),
    ]

    assert_z3_matches_brute(constraints)


def test_dataset_a_matches_brute():
    instance = generate_dataset_a(size=8, seed=1)

    assert_z3_matches_brute(instance.constraints)


def test_dataset_b_matches_brute():
    instance = generate_dataset_b(
        size=10,
        seed=2,
        num_conflicting_types=4,
    )

    assert_z3_matches_brute(instance.constraints)


def test_dataset_c_matches_brute():
    instance = generate_dataset_c(size=10, seed=3)

    assert_z3_matches_brute(instance.constraints)


def test_z3_matches_brute_on_50_random_tiny_instances():
    for seed in range(50):
        selector = seed % 3

        if selector == 0:
            instance = generate_dataset_a(size=8, seed=seed)
        elif selector == 1:
            instance = generate_dataset_b(
                size=10,
                seed=seed,
                num_conflicting_types=3,
            )
        else:
            instance = generate_dataset_c(size=10, seed=seed)

        assert_z3_matches_brute(instance.constraints)

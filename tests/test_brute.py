import pytest

from src.brute import minimum_weight_mcs_bruteforce
from src.constraints import BOOL, INT, EqConstraint, TyVar
from src.generator import generate_dataset_a
from src.solver import is_sat


def test_sat_input_returns_empty_correction_subset():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_is_int", weight=1.0),
    ]

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.selected_indices == ()
    assert result.selected_constraints == ()
    assert result.total_weight == 0.0
    assert result.cardinality == 0


def test_simple_conflict_needs_one_removed_constraint():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="a_is_int", weight=1.0),
        EqConstraint(a, BOOL, label="a_is_bool", weight=1.0),
    ]

    assert not is_sat(constraints)

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.total_weight == 1.0
    assert result.cardinality == 1
    assert result.selected_indices == (0,)


def test_weighted_conflict_chooses_cheaper_deletion():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="expensive_int", weight=10.0),
        EqConstraint(a, BOOL, label="cheap_bool", weight=1.0),
    ]

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.total_weight == 1.0
    assert result.selected_indices == (1,)
    assert result.selected_constraints[0].label == "cheap_bool"


def test_dataset_a_has_optimum_weight_one():
    instance = generate_dataset_a(size=8, seed=1)

    result = minimum_weight_mcs_bruteforce(instance.constraints)

    assert result.total_weight == 1.0
    assert result.cardinality == 1

    remaining = [
        constraint
        for i, constraint in enumerate(instance.constraints)
        if i not in result.selected_indices
    ]

    assert is_sat(remaining)


def test_two_independent_conflicts_need_two_deletions():
    a = TyVar("a")
    b = TyVar("b")

    constraints = [
        EqConstraint(a, INT, label="a_int", weight=1.0),
        EqConstraint(a, BOOL, label="a_bool", weight=1.0),
        EqConstraint(b, INT, label="b_int", weight=1.0),
        EqConstraint(b, BOOL, label="b_bool", weight=1.0),
    ]

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.total_weight == 2.0
    assert result.cardinality == 2

    remaining = [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in result.selected_indices
    ]

    assert is_sat(remaining)


def test_tie_breaking_is_deterministic_by_index_order():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="first", weight=1.0),
        EqConstraint(a, BOOL, label="second", weight=1.0),
    ]

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.selected_indices == (0,)
    assert result.selected_constraints[0].label == "first"


def test_too_many_constraints_raises_value_error():
    constraints = [
        EqConstraint(TyVar(f"a{i}"), INT, label=f"c{i}", weight=1.0)
        for i in range(13)
    ]

    with pytest.raises(ValueError):
        minimum_weight_mcs_bruteforce(constraints, max_constraints=12)


def test_checked_subsets_count_for_three_constraints():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT, label="c0", weight=1.0),
        EqConstraint(a, BOOL, label="c1", weight=1.0),
        EqConstraint(TyVar("b"), INT, label="c2", weight=1.0),
    ]

    result = minimum_weight_mcs_bruteforce(constraints)

    assert result.checked_subsets == 8

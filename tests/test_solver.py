import pytest

from src.constraints import (
    BOOL,
    CHAR,
    INT,
    STRING,
    EqConstraint,
    TyFun,
    TyList,
    TyVar,
    UnificationError,
)
from src.solver import is_sat, solve


def test_variable_unifies_with_base_type():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT),
    ]

    assert is_sat(constraints)


def test_different_base_types_are_unsat():
    constraints = [
        EqConstraint(INT, BOOL),
    ]

    assert not is_sat(constraints)


def test_transitive_variable_conflict_is_unsat():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, INT),
        EqConstraint(a, BOOL),
    ]

    assert not is_sat(constraints)


def test_function_types_can_unify():
    a = TyVar("a")
    b = TyVar("b")

    constraints = [
        EqConstraint(TyFun(a, b), TyFun(INT, BOOL)),
    ]

    subst = solve(constraints)

    assert subst["a"] == INT
    assert subst["b"] == BOOL


def test_function_argument_conflict_is_unsat():
    a = TyVar("a")

    constraints = [
        EqConstraint(TyFun(a, INT), TyFun(BOOL, INT)),
        EqConstraint(a, INT),
    ]

    assert not is_sat(constraints)


def test_list_element_unification_is_sat():
    a = TyVar("a")

    constraints = [
        EqConstraint(TyList(a), TyList(INT)),
    ]

    subst = solve(constraints)

    assert subst["a"] == INT


def test_list_element_conflict_is_unsat():
    a = TyVar("a")

    constraints = [
        EqConstraint(TyList(a), TyList(INT)),
        EqConstraint(a, BOOL),
    ]

    assert not is_sat(constraints)


def test_occurs_check_rejects_infinite_type():
    a = TyVar("a")

    constraints = [
        EqConstraint(a, TyList(a)),
    ]

    assert not is_sat(constraints)


def test_nested_function_and_list_type_is_sat():
    a = TyVar("a")
    b = TyVar("b")

    constraints = [
        EqConstraint(
            TyFun(TyList(a), b),
            TyFun(TyList(INT), BOOL),
        ),
    ]

    subst = solve(constraints)

    assert subst["a"] == INT
    assert subst["b"] == BOOL


def test_string_alias_is_list_of_char():
    constraints = [
        EqConstraint(STRING, TyList(CHAR)),
    ]

    assert is_sat(constraints)


def test_solve_raises_on_unsat():
    constraints = [
        EqConstraint(INT, BOOL),
    ]

    with pytest.raises(UnificationError):
        solve(constraints)

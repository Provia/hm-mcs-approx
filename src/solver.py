"""
Satisfiability wrapper for HM-style type-equality constraints.

For Phase 1.2, satisfiability means:
a set of type-equality constraints is satisfiable exactly when unification succeeds.
"""

from __future__ import annotations

from typing import Iterable

from .constraints import EqConstraint, Substitution, UnificationError, unify_all


def is_sat(constraints: Iterable[EqConstraint]) -> bool:
    """Return True if the constraints are satisfiable."""

    try:
        unify_all(constraints)
        return True
    except UnificationError:
        return False


def solve(constraints: Iterable[EqConstraint]) -> Substitution:
    """
    Return a unifying substitution.

    Raises UnificationError if the constraints are unsatisfiable.
    """

    return unify_all(constraints)

"""
Z3 exact baseline for minimum-weight correction subsets.

Given a set of HM-style type-equality constraints C, this module solves:

    minimize weight(S)
    such that C without S is satisfiable

Each source constraint is guarded by a Boolean keep variable. Z3 minimizes
the total weight of constraints whose keep variable is false.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import re
from typing import Any

from z3 import (
    Bool,
    Const,
    Datatype,
    If,
    Implies,
    IntSort,
    IntVal,
    Optimize,
    Sum,
    is_true,
    sat,
)

from .constraints import EqConstraint, TyBase, TyFun, TyList, TyVar, TypeExpr


_SORT_COUNTER = count()


@dataclass(frozen=True)
class ExactResult:
    """Result returned by the Z3 exact MCS solver."""

    selected_indices: tuple[int, ...]
    selected_constraints: tuple[EqConstraint, ...]
    total_weight: float
    objective_weight: int
    solver_status: str

    @property
    def cardinality(self) -> int:
        """Number of constraints removed."""

        return len(self.selected_indices)


@dataclass
class _TypeEncoding:
    """Internal Z3 encoding for HM-style type expressions."""

    sort: Any
    base_ctor: Any
    list_ctor: Any
    fun_ctor: Any
    base_codes: dict[str, int]
    var_terms: dict[str, Any]
    tag: int

    def encode(self, typ: TypeExpr) -> Any:
        """Encode a type expression as a Z3 term."""

        if isinstance(typ, TyVar):
            if typ.name not in self.var_terms:
                safe = _safe_name(typ.name)
                self.var_terms[typ.name] = Const(
                    f"v_{self.tag}_{safe}",
                    self.sort,
                )
            return self.var_terms[typ.name]

        if isinstance(typ, TyBase):
            if typ.name not in self.base_codes:
                raise ValueError(f"unknown base type name: {typ.name}")
            return self.base_ctor(IntVal(self.base_codes[typ.name]))

        if isinstance(typ, TyList):
            return self.list_ctor(self.encode(typ.elem))

        if isinstance(typ, TyFun):
            return self.fun_ctor(
                self.encode(typ.arg),
                self.encode(typ.ret),
            )

        raise TypeError(f"unknown type expression: {typ!r}")


def _safe_name(name: str) -> str:
    """Make a string safe for use inside a Z3 symbol name."""

    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not safe:
        return "unnamed"
    return safe


def _collect_type_names(
    typ: TypeExpr,
    base_names: set[str],
    var_names: set[str],
) -> None:
    """Collect base type names and type variable names from a type expression."""

    if isinstance(typ, TyVar):
        var_names.add(typ.name)
        return

    if isinstance(typ, TyBase):
        base_names.add(typ.name)
        return

    if isinstance(typ, TyList):
        _collect_type_names(typ.elem, base_names, var_names)
        return

    if isinstance(typ, TyFun):
        _collect_type_names(typ.arg, base_names, var_names)
        _collect_type_names(typ.ret, base_names, var_names)
        return

    raise TypeError(f"unknown type expression: {typ!r}")


def _make_type_encoding(constraints: list[EqConstraint]) -> _TypeEncoding:
    """
    Create a fresh Z3 datatype encoding for this exact-solver call.

    The datatype has three constructors:
        Base(Int)
        List(Type)
        Fun(Type, Type)

    Base types such as Int, Bool, and Char are encoded using integer codes.
    """

    base_names: set[str] = set()
    var_names: set[str] = set()

    for constraint in constraints:
        _collect_type_names(constraint.left, base_names, var_names)
        _collect_type_names(constraint.right, base_names, var_names)

    base_codes = {
        name: i
        for i, name in enumerate(sorted(base_names))
    }

    tag = next(_SORT_COUNTER)

    type_dt = Datatype(f"TypeExpr_{tag}")
    type_dt.declare(
        f"Base_{tag}",
        (f"base_code_{tag}", IntSort()),
    )
    type_dt.declare(
        f"List_{tag}",
        (f"elem_{tag}", type_dt),
    )
    type_dt.declare(
        f"Fun_{tag}",
        (f"arg_{tag}", type_dt),
        (f"ret_{tag}", type_dt),
    )

    type_sort = type_dt.create()

    return _TypeEncoding(
        sort=type_sort,
        base_ctor=getattr(type_sort, f"Base_{tag}"),
        list_ctor=getattr(type_sort, f"List_{tag}"),
        fun_ctor=getattr(type_sort, f"Fun_{tag}"),
        base_codes=base_codes,
        var_terms={},
        tag=tag,
    )


def _weight_to_int(weight: float) -> int:
    """
    Convert a positive constraint weight to an integer objective coefficient.

    Current generators use integer-like float weights such as 1.0 and 2.0.
    """

    if weight <= 0:
        raise ValueError(f"constraint weight must be positive, got {weight}")

    rounded = round(weight)

    if abs(weight - rounded) > 1e-9:
        raise ValueError(
            "non-integer weights are not supported yet in exact.py; "
            f"got {weight}"
        )

    return int(rounded)


def _remaining_constraints(
    constraints: list[EqConstraint],
    selected_indices: tuple[int, ...],
) -> list[EqConstraint]:
    """Return constraints after dropping selected indices."""

    selected = set(selected_indices)

    return [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in selected
    ]


def minimum_weight_mcs_z3(
    constraints: list[EqConstraint],
    timeout_ms: int | None = None,
) -> ExactResult:
    """
    Compute an exact minimum-weight correction subset using Z3 Optimize.

    Args:
        constraints: HM-style type-equality constraints.
        timeout_ms: Optional timeout in milliseconds.

    Returns:
        ExactResult containing one optimal correction subset.

    Raises:
        RuntimeError: if Z3 does not return sat.
    """

    constraints = list(constraints)
    encoding = _make_type_encoding(constraints)

    opt = Optimize()

    if timeout_ms is not None:
        opt.set(timeout=timeout_ms)

    keep = [
        Bool(f"keep_{encoding.tag}_{i}")
        for i in range(len(constraints))
    ]

    for i, constraint in enumerate(constraints):
        left = encoding.encode(constraint.left)
        right = encoding.encode(constraint.right)
        opt.add(Implies(keep[i], left == right))

    scaled_weights = [
        _weight_to_int(constraint.weight)
        for constraint in constraints
    ]

    objective = Sum(
        [
            If(keep[i], 0, scaled_weights[i])
            for i in range(len(constraints))
        ]
    )

    opt.minimize(objective)

    status = opt.check()

    if status != sat:
        raise RuntimeError(f"Z3 Optimize did not return sat, got {status}")

    model = opt.model()

    selected_indices = tuple(
        i
        for i, keep_var in enumerate(keep)
        if not is_true(model.eval(keep_var, model_completion=True))
    )

    selected_constraints = tuple(
        constraints[i]
        for i in selected_indices
    )

    return ExactResult(
        selected_indices=selected_indices,
        selected_constraints=selected_constraints,
        total_weight=float(
            sum(constraint.weight for constraint in selected_constraints)
        ),
        objective_weight=sum(
            scaled_weights[i]
            for i in selected_indices
        ),
        solver_status=str(status),
    )


# Shorter alias used by later modules and experiments.
minimum_weight_mcs_exact = minimum_weight_mcs_z3

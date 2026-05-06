"""
Synthetic instance generators for HM-style type-equality constraint systems.

Dataset A:
    Single planted conflict.

Dataset B:
    Overlapping conflicts, where several inconsistent constraints share the same
    type variable.

Dataset C:
    HM-shaped instances using variables, base types, function types, list types,
    and weights derived from type-expression depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .constraints import (
    BOOL,
    CHAR,
    INT,
    STRING,
    EqConstraint,
    TyBase,
    TyFun,
    TyList,
    TyVar,
    TypeExpr,
)


@dataclass(frozen=True)
class Instance:
    """A generated benchmark instance."""

    name: str
    constraints: list[EqConstraint]
    metadata: dict[str, Any] = field(default_factory=dict)


def type_depth(typ: TypeExpr) -> int:
    """Return a simple structural depth measure for a type expression."""

    if isinstance(typ, TyVar):
        return 0

    if isinstance(typ, TyBase):
        return 0

    if isinstance(typ, TyList):
        return 1 + type_depth(typ.elem)

    if isinstance(typ, TyFun):
        return 1 + max(type_depth(typ.arg), type_depth(typ.ret))

    raise TypeError(f"unknown type expression: {typ!r}")


def constraint_weight(left: TypeExpr, right: TypeExpr) -> float:
    """
    Assign a positive weight using type depth.

    Deeper type expressions receive larger weights. This is a simple proxy for
    AST-depth-based weights in later experiments.
    """

    return float(1 + max(type_depth(left), type_depth(right)))


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _base_type_candidates() -> list[TypeExpr]:
    return [INT, BOOL, CHAR, STRING]


def _random_base_type(rng: random.Random) -> TypeExpr:
    return rng.choice(_base_type_candidates())


def _random_type_expr(
    rng: random.Random,
    depth: int,
    var_name: str,
) -> TypeExpr:
    """
    Generate a small HM-style type expression.

    The generated expression may use the supplied variable name at leaves.
    """

    if depth <= 0:
        choices: list[TypeExpr] = [
            TyVar(var_name),
            INT,
            BOOL,
            CHAR,
        ]
        return rng.choice(choices)

    shape = rng.choice(["base", "var", "list", "fun"])

    if shape == "base":
        return _random_base_type(rng)

    if shape == "var":
        return TyVar(var_name)

    if shape == "list":
        return TyList(_random_type_expr(rng, depth - 1, var_name))

    left = _random_type_expr(rng, depth - 1, var_name + "_a")
    right = _random_type_expr(rng, depth - 1, var_name + "_b")
    return TyFun(left, right)


def _make_constraint(
    left: TypeExpr,
    right: TypeExpr,
    label: str,
    weight: float | None = None,
) -> EqConstraint:
    if weight is None:
        weight = constraint_weight(left, right)

    return EqConstraint(
        left=left,
        right=right,
        label=label,
        weight=weight,
    )


def _append_satisfiable_fillers(
    constraints: list[EqConstraint],
    target_size: int,
    seed: int | None,
    prefix: str,
) -> None:
    """
    Add satisfiable filler constraints using fresh variables.

    Each filler uses fresh variables, so these constraints should not introduce
    conflicts with the planted unsatisfiable core.
    """

    rng = _rng(seed)
    i = 0

    while len(constraints) < target_size:
        v = TyVar(f"{prefix}_fill_{i}")
        label = f"{prefix}_fill_{i}"

        pattern = rng.choice(["base", "list", "fun", "list_eq"])

        if pattern == "base":
            right = _random_base_type(rng)
            constraints.append(_make_constraint(v, right, label))

        elif pattern == "list":
            right = TyList(_random_base_type(rng))
            constraints.append(_make_constraint(v, right, label))

        elif pattern == "fun":
            right = TyFun(_random_base_type(rng), _random_base_type(rng))
            constraints.append(_make_constraint(v, right, label))

        else:
            elem = TyVar(f"{prefix}_fill_elem_{i}")
            right = TyList(_random_base_type(rng))
            constraints.append(_make_constraint(TyList(elem), right, label))

        i += 1


def generate_dataset_a(size: int = 10, seed: int | None = None) -> Instance:
    """
    Generate Dataset A: a single planted type conflict.

    Core conflict:
        a = Int
        a = Bool
    """

    if size < 2:
        raise ValueError("Dataset A requires size >= 2")

    a = TyVar("a_conflict")

    constraints = [
        _make_constraint(a, INT, "A_conflict_int", weight=1.0),
        _make_constraint(a, BOOL, "A_conflict_bool", weight=1.0),
    ]

    _append_satisfiable_fillers(
        constraints=constraints,
        target_size=size,
        seed=seed,
        prefix="A",
    )

    return Instance(
        name="dataset_a_single_conflict",
        constraints=constraints,
        metadata={
            "dataset": "A",
            "description": "single planted conflict",
            "target_size": size,
            "planted_conflict_labels": ["A_conflict_int", "A_conflict_bool"],
        },
    )


def generate_dataset_b(
    size: int = 12,
    seed: int | None = None,
    num_conflicting_types: int = 4,
) -> Instance:
    """
    Generate Dataset B: overlapping conflicts.

    Several constraints bind the same variable to incompatible types.

    Example:
        x = Int
        x = Bool
        x = Char
        x = [Char]

    Multiple minimal conflicts share x, so this dataset is intended to create
    overlapping MUS structure.
    """

    if num_conflicting_types < 2:
        raise ValueError("num_conflicting_types must be >= 2")

    if size < num_conflicting_types:
        raise ValueError("size must be >= num_conflicting_types")

    x = TyVar("x_overlap")

    candidates: list[TypeExpr] = [
        INT,
        BOOL,
        CHAR,
        STRING,
        TyList(INT),
        TyFun(INT, INT),
        TyFun(BOOL, BOOL),
    ]

    if num_conflicting_types > len(candidates):
        extra_needed = num_conflicting_types - len(candidates)
        for i in range(extra_needed):
            candidates.append(TyBase(f"Custom{i}"))

    chosen = candidates[:num_conflicting_types]

    constraints: list[EqConstraint] = []

    for i, typ in enumerate(chosen):
        constraints.append(
            _make_constraint(
                x,
                typ,
                label=f"B_overlap_{i}",
                weight=1.0 + i,
            )
        )

    _append_satisfiable_fillers(
        constraints=constraints,
        target_size=size,
        seed=seed,
        prefix="B",
    )

    return Instance(
        name="dataset_b_overlapping_conflicts",
        constraints=constraints,
        metadata={
            "dataset": "B",
            "description": "overlapping conflicts sharing one type variable",
            "target_size": size,
            "num_conflicting_types": num_conflicting_types,
            "overlap_variable": "x_overlap",
            "planted_conflict_labels": [f"B_overlap_{i}" for i in range(num_conflicting_types)],
        },
    )


def generate_dataset_c(
    size: int = 15,
    seed: int | None = None,
    max_depth: int = 3,
) -> Instance:
    """
    Generate Dataset C: HM-shaped constraints.

    This uses type variables, base types, function types, and list types.
    Weights are derived from type-expression depth.

    Planted conflict:
        f = [a] -> b
        a = Int
        b = Bool
        f = [Char]

    The variable f is forced to be both a function type and a list type.
    """

    if size < 4:
        raise ValueError("Dataset C requires size >= 4")

    rng = _rng(seed)

    f = TyVar("f")
    a = TyVar("a")
    b = TyVar("b")

    function_shape = TyFun(TyList(a), b)
    list_shape = TyList(CHAR)

    constraints = [
        _make_constraint(f, function_shape, "C_shape_function"),
        _make_constraint(a, INT, "C_arg_int"),
        _make_constraint(b, BOOL, "C_ret_bool"),
        _make_constraint(f, list_shape, "C_conflict_list"),
    ]

    i = 0
    while len(constraints) < size:
        v = TyVar(f"C_fill_{i}")
        typ = _random_type_expr(
            rng=rng,
            depth=max_depth,
            var_name=f"C_inner_{i}",
        )
        constraints.append(
            _make_constraint(
                v,
                typ,
                label=f"C_fill_{i}",
            )
        )
        i += 1

    return Instance(
        name="dataset_c_hm_shaped",
        constraints=constraints,
        metadata={
            "dataset": "C",
            "description": "HM-shaped function/list constraint structure",
            "target_size": size,
            "max_depth": max_depth,
            "planted_conflict_labels": ["C_shape_function", "C_conflict_list"],
            "weight_rule": "1 + max type-expression depth",
        },
    )


def generate_all_small(seed: int | None = 0) -> list[Instance]:
    """Generate one small instance from each dataset."""

    return [
        generate_dataset_a(size=10, seed=seed),
        generate_dataset_b(size=12, seed=seed),
        generate_dataset_c(size=15, seed=seed),
    ]

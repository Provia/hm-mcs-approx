"""
Core type-expression and type-equality constraint definitions.

This module implements a small Hindley-Milner-style type language:

- type variables: a, b, x
- base types: Int, Bool, Char
- function types: t1 -> t2
- list types: [t]

It also provides first-order unification with an occurs check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias


@dataclass(frozen=True)
class TyVar:
    """A type variable, such as a or b."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyBase:
    """A base type, such as Int, Bool, or Char."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TyFun:
    """A function type, such as Int -> Bool."""

    arg: TypeExpr
    ret: TypeExpr

    def __str__(self) -> str:
        return f"({self.arg} -> {self.ret})"


@dataclass(frozen=True)
class TyList:
    """A list type, such as [Int]."""

    elem: TypeExpr

    def __str__(self) -> str:
        return f"[{self.elem}]"


TypeExpr: TypeAlias = TyVar | TyBase | TyFun | TyList


@dataclass(frozen=True)
class EqConstraint:
    """A type-equality constraint t1 = t2."""

    left: TypeExpr
    right: TypeExpr
    label: str = ""
    weight: float = 1.0

    def __str__(self) -> str:
        if self.label:
            return f"{self.label}: {self.left} = {self.right}"
        return f"{self.left} = {self.right}"


Substitution: TypeAlias = dict[str, TypeExpr]


class UnificationError(Exception):
    """Raised when a set of type-equality constraints is unsatisfiable."""


def apply_subst(typ: TypeExpr, subst: Substitution) -> TypeExpr:
    """Apply a substitution to a type expression."""

    if isinstance(typ, TyVar):
        if typ.name not in subst:
            return typ
        return apply_subst(subst[typ.name], subst)

    if isinstance(typ, TyBase):
        return typ

    if isinstance(typ, TyFun):
        return TyFun(
            apply_subst(typ.arg, subst),
            apply_subst(typ.ret, subst),
        )

    if isinstance(typ, TyList):
        return TyList(apply_subst(typ.elem, subst))

    raise TypeError(f"unknown type expression: {typ!r}")


def occurs_in(var_name: str, typ: TypeExpr, subst: Substitution) -> bool:
    """Check whether a type variable occurs inside a type expression."""

    typ = apply_subst(typ, subst)

    if isinstance(typ, TyVar):
        return typ.name == var_name

    if isinstance(typ, TyBase):
        return False

    if isinstance(typ, TyFun):
        return occurs_in(var_name, typ.arg, subst) or occurs_in(
            var_name,
            typ.ret,
            subst,
        )

    if isinstance(typ, TyList):
        return occurs_in(var_name, typ.elem, subst)

    raise TypeError(f"unknown type expression: {typ!r}")


def bind_var(var: TyVar, typ: TypeExpr, subst: Substitution) -> Substitution:
    """Bind a type variable to a type expression."""

    typ = apply_subst(typ, subst)

    if typ == var:
        return subst

    if occurs_in(var.name, typ, subst):
        raise UnificationError(f"occurs check failed: {var} occurs in {typ}")

    new_subst = dict(subst)
    new_subst[var.name] = typ

    single_binding = {var.name: typ}
    for key, value in list(new_subst.items()):
        if key != var.name:
            new_subst[key] = apply_subst(value, single_binding)

    return new_subst


def unify(left: TypeExpr, right: TypeExpr, subst: Substitution | None = None) -> Substitution:
    """Unify two type expressions under the current substitution."""

    if subst is None:
        subst = {}

    left = apply_subst(left, subst)
    right = apply_subst(right, subst)

    if left == right:
        return subst

    if isinstance(left, TyVar):
        return bind_var(left, right, subst)

    if isinstance(right, TyVar):
        return bind_var(right, left, subst)

    if isinstance(left, TyBase) and isinstance(right, TyBase):
        if left.name == right.name:
            return subst
        raise UnificationError(f"base type mismatch: {left} != {right}")

    if isinstance(left, TyList) and isinstance(right, TyList):
        return unify(left.elem, right.elem, subst)

    if isinstance(left, TyFun) and isinstance(right, TyFun):
        subst = unify(left.arg, right.arg, subst)
        return unify(left.ret, right.ret, subst)

    raise UnificationError(f"type constructor mismatch: {left} != {right}")


def unify_all(constraints: Iterable[EqConstraint]) -> Substitution:
    """Unify all constraints and return the final substitution."""

    subst: Substitution = {}

    for constraint in constraints:
        subst = unify(constraint.left, constraint.right, subst)

    return subst


# Convenient base-type constants.
INT = TyBase("Int")
BOOL = TyBase("Bool")
CHAR = TyBase("Char")
STRING = TyList(CHAR)

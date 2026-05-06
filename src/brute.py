"""
Brute-force exact solver for minimum-weight correction subsets.

Given a set of type-equality constraints C, a correction subset S is a subset
of constraints such that C without S is satisfiable.

This module enumerates all candidate correction subsets and returns one
deterministic minimum-weight solution. It is intended only for tiny instances
and for sanity-checking the Z3 exact baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .constraints import EqConstraint
from .solver import is_sat


@dataclass(frozen=True)
class BruteForceResult:
    """Result returned by the brute-force MCS solver."""

    selected_indices: tuple[int, ...]
    selected_constraints: tuple[EqConstraint, ...]
    total_weight: float
    checked_subsets: int
    max_constraints: int

    @property
    def cardinality(self) -> int:
        """Number of constraints removed."""

        return len(self.selected_indices)


def _subset_weight(constraints: list[EqConstraint], indices: tuple[int, ...]) -> float:
    """Return the total weight of the selected constraint indices."""

    return float(sum(constraints[i].weight for i in indices))


def _remaining_after_drop(
    constraints: list[EqConstraint],
    drop_indices: tuple[int, ...],
) -> list[EqConstraint]:
    """Return constraints after dropping the selected indices."""

    drop_set = set(drop_indices)
    return [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in drop_set
    ]


def minimum_weight_mcs_bruteforce(
    constraints: list[EqConstraint],
    max_constraints: int = 12,
) -> BruteForceResult:
    """
    Find a minimum-weight correction subset by exhaustive enumeration.

    Tie breaking is deterministic:
        1. smaller total weight
        2. fewer removed constraints
        3. lexicographically smaller tuple of indices

    Raises:
        ValueError: if the number of constraints exceeds max_constraints.
    """

    constraints = list(constraints)
    n = len(constraints)

    if n > max_constraints:
        raise ValueError(
            f"brute force is capped at {max_constraints} constraints, got {n}"
        )

    best_indices: tuple[int, ...] | None = None
    best_key: tuple[float, int, tuple[int, ...]] | None = None
    checked_subsets = 0

    for subset_size in range(n + 1):
        for indices in combinations(range(n), subset_size):
            candidate_indices = tuple(indices)
            remaining = _remaining_after_drop(constraints, candidate_indices)

            checked_subsets += 1

            if not is_sat(remaining):
                continue

            total_weight = _subset_weight(constraints, candidate_indices)
            candidate_key = (
                total_weight,
                len(candidate_indices),
                candidate_indices,
            )

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_indices = candidate_indices

    if best_indices is None:
        raise RuntimeError(
            "no correction subset found, this should be impossible because "
            "dropping all constraints must be satisfiable"
        )

    selected_constraints = tuple(constraints[i] for i in best_indices)

    return BruteForceResult(
        selected_indices=best_indices,
        selected_constraints=selected_constraints,
        total_weight=_subset_weight(constraints, best_indices),
        checked_subsets=checked_subsets,
        max_constraints=max_constraints,
    )


# Shorter alias used by later modules and experiments.
minimum_weight_mcs = minimum_weight_mcs_bruteforce

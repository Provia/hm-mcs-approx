"""
Lazy MUS-guided greedy algorithm for minimum-weight correction subsets.

The algorithm repeatedly finds one minimal unsatisfiable subset, then computes
a weighted greedy hitting set over the MUSes discovered so far.

This is an approximation algorithm. The exact optimum is provided separately
by exact.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import Bool, Solver, unsat

from .constraints import EqConstraint
from .exact import _make_type_encoding
from .solver import is_sat


@dataclass(frozen=True)
class GreedyResult:
    """Result returned by the greedy MCS approximation algorithm."""

    selected_indices: tuple[int, ...]
    selected_constraints: tuple[EqConstraint, ...]
    total_weight: float
    discovered_muses: tuple[tuple[int, ...], ...]
    iterations: int

    @property
    def cardinality(self) -> int:
        """Number of constraints removed."""

        return len(self.selected_indices)


def _constraints_at(
    constraints: list[EqConstraint],
    indices: tuple[int, ...],
) -> list[EqConstraint]:
    """Return constraints at the given indices."""

    return [constraints[i] for i in indices]


def _remaining_after_drop(
    constraints: list[EqConstraint],
    drop_indices: set[int],
) -> list[EqConstraint]:
    """Return constraints after dropping selected indices."""

    return [
        constraint
        for i, constraint in enumerate(constraints)
        if i not in drop_indices
    ]


def _subset_is_unsat(
    constraints: list[EqConstraint],
    indices: tuple[int, ...],
) -> bool:
    """Return True if the selected subset of constraints is unsatisfiable."""

    return not is_sat(_constraints_at(constraints, indices))


def extract_unsat_core_indices(
    constraints: list[EqConstraint],
    candidate_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """
    Extract one unsatisfiable core from candidate constraints using Z3.

    If Z3 cannot provide a useful core, this falls back to returning all
    candidate indices. The returned core is not guaranteed to be minimal.
    """

    constraints = list(constraints)

    if candidate_indices is None:
        candidate_indices = tuple(range(len(constraints)))
    else:
        candidate_indices = tuple(candidate_indices)

    if is_sat(_constraints_at(constraints, candidate_indices)):
        raise ValueError("cannot extract an unsat core from satisfiable constraints")

    encoding = _make_type_encoding(constraints)
    solver = Solver()

    tracker_to_index: dict[str, int] = {}

    for i in candidate_indices:
        constraint = constraints[i]
        tracker = Bool(f"track_{encoding.tag}_{i}")

        left = encoding.encode(constraint.left)
        right = encoding.encode(constraint.right)

        solver.assert_and_track(left == right, tracker)
        tracker_to_index[tracker.decl().name()] = i

    status = solver.check()

    if status != unsat:
        return candidate_indices

    core = solver.unsat_core()
    core_indices = tuple(
        sorted(
            tracker_to_index[tracker.decl().name()]
            for tracker in core
            if tracker.decl().name() in tracker_to_index
        )
    )

    if not core_indices:
        return candidate_indices

    if is_sat(_constraints_at(constraints, core_indices)):
        return candidate_indices

    return core_indices


def shrink_core_to_mus(
    constraints: list[EqConstraint],
    core_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Shrink an unsatisfiable core to a subset-minimal unsatisfiable subset.

    This uses deletion-based minimisation:
    try removing each constraint, and keep it removed if the subset remains
    unsatisfiable.
    """

    constraints = list(constraints)
    mus = list(dict.fromkeys(core_indices))
    mus.sort()

    if is_sat(_constraints_at(constraints, tuple(mus))):
        raise ValueError("cannot shrink a satisfiable core to a MUS")

    for index in list(mus):
        candidate = tuple(i for i in mus if i != index)

        if candidate and _subset_is_unsat(constraints, candidate):
            mus.remove(index)

    return tuple(mus)


def extract_one_mus(
    constraints: list[EqConstraint],
    candidate_indices: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """
    Extract one subset-minimal unsatisfiable subset.

    This first asks Z3 for an unsat core, then shrinks the core using the
    project unification-based satisfiability checker.
    """

    core = extract_unsat_core_indices(constraints, candidate_indices)
    return shrink_core_to_mus(constraints, core)


def is_mus(
    constraints: list[EqConstraint],
    indices: tuple[int, ...],
) -> bool:
    """Return True if the selected indices form a MUS."""

    indices = tuple(sorted(indices))

    if not indices:
        return False

    if is_sat(_constraints_at(constraints, indices)):
        return False

    for index in indices:
        candidate = tuple(i for i in indices if i != index)

        if candidate and _subset_is_unsat(constraints, candidate):
            return False

        if not candidate and not is_sat([]):
            return False

    return True


def weighted_greedy_hitting_set(
    muses: list[tuple[int, ...]],
    weights: list[float],
) -> tuple[int, ...]:
    """
    Compute a weighted greedy hitting set over discovered MUSes.

    Score:
        weight(c) / number of currently unhit MUSes containing c

    Tie breaking:
        smaller score
        smaller weight
        smaller original index
    """

    selected: set[int] = set()
    unhit = [set(mus) for mus in muses]

    while unhit:
        candidate_indices = sorted(set().union(*unhit))

        best_index: int | None = None
        best_key: tuple[float, float, int] | None = None

        for index in candidate_indices:
            coverage = sum(1 for mus in unhit if index in mus)

            if coverage == 0:
                continue

            score = weights[index] / coverage
            key = (score, weights[index], index)

            if best_key is None or key < best_key:
                best_key = key
                best_index = index

        if best_index is None:
            raise RuntimeError("failed to choose a hitting-set element")

        selected.add(best_index)
        unhit = [
            mus
            for mus in unhit
            if best_index not in mus
        ]

    return tuple(sorted(selected))


def minimum_weight_mcs_greedy(
    constraints: list[EqConstraint],
    max_iterations: int | None = None,
) -> GreedyResult:
    """
    Approximate a minimum-weight correction subset using lazy MUS-guided greedy.

    Args:
        constraints: HM-style type-equality constraints.
        max_iterations: Optional safety cap.

    Returns:
        GreedyResult containing one correction subset.
    """

    constraints = list(constraints)
    weights = [constraint.weight for constraint in constraints]

    selected: set[int] = set()
    discovered_muses: list[tuple[int, ...]] = []
    iterations = 0

    if is_sat(constraints):
        return GreedyResult(
            selected_indices=(),
            selected_constraints=(),
            total_weight=0.0,
            discovered_muses=(),
            iterations=0,
        )

    while not is_sat(_remaining_after_drop(constraints, selected)):
        if max_iterations is not None and iterations >= max_iterations:
            raise RuntimeError(
                f"greedy exceeded max_iterations={max_iterations}"
            )

        active_indices = tuple(
            i
            for i in range(len(constraints))
            if i not in selected
        )

        mus = extract_one_mus(constraints, active_indices)

        if mus not in discovered_muses:
            discovered_muses.append(mus)

        selected = set(
            weighted_greedy_hitting_set(discovered_muses, weights)
        )

        iterations += 1

    selected_indices = tuple(sorted(selected))
    selected_constraints = tuple(
        constraints[i]
        for i in selected_indices
    )

    return GreedyResult(
        selected_indices=selected_indices,
        selected_constraints=selected_constraints,
        total_weight=float(
            sum(constraint.weight for constraint in selected_constraints)
        ),
        discovered_muses=tuple(discovered_muses),
        iterations=iterations,
    )


# Shorter alias used by later modules and experiments.
minimum_weight_mcs = minimum_weight_mcs_greedy

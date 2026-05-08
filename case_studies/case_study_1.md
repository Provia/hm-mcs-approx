# Case Study 1 — Single Conflict (Dataset A analogue)

**Purpose**: Illustrate the HM program → unsatisfiable constraint set → MCS mapping
in the simplest possible setting. Used in the **Formulation** section of D1.

---

## Haskell Program

```haskell
let f x = x + 1 in f True
```

## HM Constraint Generation

Hindley–Milner assigns a fresh type variable α to `x`. Two uses of `x` impose
incompatible constraints:

| Constraint | Source | Weight |
|---|---|---|
| C0: α = Int  | `x + 1` requires the argument of `(+)` to be `Int` | 1.0 |
| C1: α = Bool | `f True` passes `True :: Bool` where `f` expects α | 1.0 |
| C2: β = Int  | satisfiable filler (another binding in scope) | 1.0 |
| C3: γ = Bool | satisfiable filler | 1.0 |

The full set {C0, C1, C2, C3} is **unsatisfiable**: unifying α = Int and α = Bool
requires Int = Bool, which fails.

## MUS Structure

There is exactly **one** Minimal Unsatisfiable Subset:

```
MUS = { C0, C1 }
```

Removing either C0 or C1 restores satisfiability. C2 and C3 play no role in the conflict.

## Minimum-Weight Correction Set

Both elements of the MUS have equal weight (1.0), so either singleton is optimal:

```
MCS* = { C0 }   or   { C1 }     total weight = 1.0
```

**Greedy result**: discovers MUS = {C0, C1}, selects C0 (first in index order among
equal-weight candidates).  
**Approximation ratio**: 1.0 / 1.0 = **1.00**

## Interpretation

This case corresponds to **Dataset A** in the experiments. The single MUS of size 2
with equal-weight constraints means greedy and exact necessarily agree. The example
illustrates the core MTECS formulation: a type error in a program maps to an
unsatisfiable constraint set, and fixing the error corresponds to finding a
minimum-weight correction subset.

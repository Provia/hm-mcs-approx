# Case Study 2 — Overlapping Conflicts (Dataset B analogue)

**Purpose**: Illustrate the MUS overlap structure that arises when a single type variable
is constrained by multiple incompatible uses. Used in the **Methodology** section of D1
to motivate Dataset B.

---

## Haskell Program

```haskell
let x = 5 in (x ++ "hello", x && True)
```

## HM Constraint Generation

The variable `x` is inferred to have type τ. Three uses impose three mutually
incompatible constraints on τ:

| Constraint | Source | Weight |
|---|---|---|
| C0: τ = Int   | `x = 5` — integer literal | 1.0 |
| C1: τ = [Char] | `x ++ "hello"` — `(++)` requires a list type | 2.0 |
| C2: τ = Bool  | `x && True` — `(&&)` requires `Bool` | 3.0 |
| C3: δ = Int   | satisfiable filler | 1.0 |

The full set {C0, C1, C2, C3} is **unsatisfiable**: no single type can simultaneously
be `Int`, `[Char]`, and `Bool`.

## MUS Structure

Any two constraints involving τ form a MUS (each pair is minimal):

```
MUS₁ = { C1, C2 }   (τ = [Char]  vs  τ = Bool)
MUS₂ = { C0, C2 }   (τ = Int     vs  τ = Bool)
```

*(MUS₃ = {C0, C1} is also present but not discovered by lazy greedy before halting.)*

All MUSes share the variable τ — this is the **overlap structure** that characterises
Dataset B.

## Greedy Execution (Step by Step)

**Iteration 1**: constraint set {C0,C1,C2,C3} minus selected=∅ is unsat.  
— discover MUS₁ = {C1, C2}  
— hits: C1 appears in 1 MUS, C2 appears in 1 MUS  
— ratio w/hits: C1 → 2.0/1 = 2.0, C2 → 3.0/1 = 3.0  
— **select C1** (smaller ratio)

**Iteration 2**: remaining set {C0, C2, C3} is still unsat (τ=Int vs τ=Bool).  
— discover MUS₂ = {C0, C2}  
— hits updated: C0 → 1, C2 → 1  
— ratio: C0 → 1.0/1 = 1.0, C2 → 3.0/1 = 3.0  
— **select C0** (smaller ratio)

**Termination**: {C2, C3} is satisfiable (τ = Bool and δ = Int are independent). ✓

```
Greedy MCS = { C0, C1 }    total weight = 1.0 + 2.0 = 3.0
```

## Minimum-Weight Correction Set

Brute-force enumeration confirms the optimum:

```
MCS* = { C0, C1 }    total weight = 3.0
```

(Alternatives: {C0, C2} → 4.0, {C1, C2} → 5.0 — all heavier.)

**Approximation ratio**: 3.0 / 3.0 = **1.00**

## Interpretation

This case corresponds to **Dataset B** in the experiments. The shared-variable overlap
structure produces multiple MUSes (all pairs of the three τ-constraints). Despite this
overlap, the weighted greedy algorithm reaches the optimum because the clique-like
structure on a single variable makes the hitting-set instance easy: the cheapest
combination of constraints that covers all MUSes coincides with the optimal MCS.

This case illustrates why Dataset B does not provoke an approximation gap, and why
future generator designs should introduce MUSes with *disjoint* support to separate
the cheap-greedy path from the globally optimal solution.

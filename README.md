# hm-mcs-approx

An empirical study of greedy approximation for **Minimum-Weight Minimum
Correction Subset** on synthetic Hindley–Milner-style type-equality
constraint systems.

This repository accompanies the report submitted as Deliverable D1 for
Assignment 2.2 of COMP4600/8460 Advanced Algorithms (ANU, Semester 1, 2026).

---

## What this project investigates

When a Hindley–Milner type checker rejects a program, the rejection corresponds
to an unsatisfiable set of type-equality constraints. A *correction subset* is
a subset whose removal restores satisfiability; a *minimum-weight* correction
subset is the most parsimonious explanation under a chosen weighting.

The project asks an empirical question:

> How well does a polynomial-time greedy hitting-set algorithm approximate
> minimum-weight correction subsets on synthetic HM-style instances, compared
> with an exact baseline, and which structural properties of the instances
> drive the approximation behaviour?

I implement a lazy MUS-guided greedy algorithm, an exact Z3 Optimize baseline,
a brute-force sanity checker, and three families of synthetic HM-style
generators, then evaluate them on 300 instances spanning sizes
|C| ∈ {10, 20, 50, 100, 200} and 20 random seeds per cell.

The headline empirical finding is that the greedy algorithm matches the exact
optimum on every tested instance (approximation ratio uniformly 1.0). The
report interprets this as a property of the synthetic generators rather than
of the algorithm: the generated MUS hypergraphs are too structurally regular
to exercise the standard worst-case behaviour of weighted greedy hitting set.

---

## Repository structure

```
hm-mcs-approx/
├── src/                        # core library
│   ├── constraints.py          # type expressions, equality constraints, unification
│   ├── solver.py               # satisfiability check via unification
│   ├── generator.py            # synthetic instance generators (Datasets A/B/C)
│   ├── greedy.py               # lazy MUS-guided greedy algorithm
│   ├── exact.py                # Z3 Optimize baseline
│   ├── brute.py                # brute-force exhaustive search (sanity)
│   └── metrics.py              # evaluation harness and structural metrics
│
├── tests/                      # pytest suite
│   ├── test_solver.py
│   ├── test_generator.py
│   ├── test_brute.py
│   ├── test_exact_vs_brute.py  # 50-instance Z3 vs brute fuzz check
│   ├── test_greedy.py
│   └── test_metrics.py
│
├── experiments/                # experiment and analysis scripts
│   ├── sanity.py               # small end-to-end pipeline check
│   ├── run_main.py             # full grid: 3 datasets × 5 sizes × 20 seeds
│   ├── run_correlations.py     # grouped summaries and analysis artefacts
│   └── make_figures.py         # regenerates paper figures from CSV outputs
│
├── results/                    # CSV outputs from experiments and analysis
│   ├── main_results.csv
│   ├── brute_sanity_results.csv
│   ├── brute_sanity_summary.csv
│   ├── summary_by_dataset_size.csv
│   └── correlations.csv
│
├── figures/                    # plots produced from the CSVs
│   ├── figure1_runtime_scaling.pdf
│   ├── figure1_runtime_scaling.png
│   ├── figure2_runtime_ratio_mus_count.pdf
│   └── figure2_runtime_ratio_mus_count.png
│
├── case_studies/               # worked Haskell-like examples translated to constraints
│
├── paper/                      # LaTeX source for the D1 report
│   ├── sections/               # sectioned source files
│   ├── main.tex                # top-level document
│   ├── refs.bib                # bibliography
│   └── main.pdf                # compiled report
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

The project targets Python 3.11+. A virtual environment is recommended.

```bash
git clone https://github.com/Provia/hm-mcs-approx.git
cd hm-mcs-approx
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `z3-solver`, `numpy`, `pandas`, `matplotlib`, `pytest`.

---

## Reproducing the experiments

All experiments are deterministic given the seeds in `experiments/`.

**1. Run the test suite** (fast, ~10 seconds):

```bash
pytest
```

This includes a 50-instance fuzz check confirming that the Z3 exact baseline
agrees with brute-force enumeration on small instances.

**2. Run the sanity experiment** (small end-to-end check, ~30 seconds):

```bash
python experiments/sanity.py
```

Outputs `results/sanity_results.csv` and an aggregate summary printed to
stdout. Useful for verifying the pipeline before running the full grid.

**3. Run the main experiment** (full 300-instance grid, a few minutes):

```bash
python experiments/run_main.py
```

Outputs:

- `results/main_results.csv` — one row per instance with greedy weight, exact
  weight, approximation ratio, runtime, and structural features.
- `results/brute_sanity_results.csv` — Z3 vs brute-force agreement on 15
  small instances (used as an internal validation artefact).

**4. Generate analysis summaries**:

```bash
python experiments/run_correlations.py
```

Outputs:

- `results/summary_by_dataset_size.csv`
- `results/correlations.csv`
- `results/brute_sanity_summary.csv`

**5. Regenerate figures**:

```bash
python experiments/make_figures.py
```

Figures are saved under `figures/`.

---

## Reading the results

`main_results.csv` columns of interest:

| Column | Meaning |
|---|---|
| `dataset` | A, B, or C |
| `num_constraints` | size of the instance, |C| |
| `greedy_weight` / `exact_weight` | total weight of selected correction subset |
| `approximation_ratio` | greedy_weight / exact_weight |
| `greedy_runtime_sec` / `exact_runtime_sec` | wall-clock time |
| `num_discovered_muses` | how many MUSes the lazy greedy loop visited |
| `greedy_iterations` | greedy outer-loop iterations |
| `overlap_measure` | mean pairwise Jaccard overlap of discovered MUSes |
| `weight_variance` | variance of constraint weights in the instance |

Across 300 instances, `approximation_ratio == 1.0` everywhere; the runtime
and MUS-discovery columns are where the structural differences between
generator families show up.

---

## Code design notes

- **`solver.py` uses Robinson unification, not Z3.** Satisfiability is the
  hot path inside the greedy loop, so I use a direct unification
  implementation; Z3 is reserved for the exact baseline and unsat-core
  extraction.

- **`exact.py` encodes constraints with a fresh Z3 datatype per call.** Each
  invocation builds a small `TypeExpr` algebraic datatype (Base / List / Fun)
  rather than reusing a global sort, which avoids cross-call interference
  between independent instances.

- **`greedy.py` is lazy.** It does not pre-enumerate MUSes; instead, each
  outer iteration extracts one MUS via Z3 unsat-core followed by
  deletion-based shrinking, then runs weighted greedy hitting-set over the
  MUSes discovered so far. This trades a tighter approximation bound for a
  cheaper inner loop.

- **`metrics.py` is the boundary between algorithms and experiments.** It
  takes an `Instance`, runs greedy and exact, and returns a flat
  `EvaluationResult` dataclass — easy to convert to a `pandas` DataFrame row.

---

## Background and report

The full report is in `paper/main.pdf`. It is structured around the seven
sections required by Assignment 2.2: Title, Abstract, Motivation,
Formulation, Methodology, Results, and Conclusion and Discussion.

The Methodology and Results sections in the report mirror the directory
layout: `src/generator.py` corresponds to Section 3.1, the algorithms in
`src/greedy.py` and `src/exact.py` to Section 3.3 and 3.4, and the figures
in `figures/` to the visualisations in Section 4.

---

## License

MIT. See `LICENSE`.

---

## Acknowledgement

This project was prepared for COMP4600/8460 Advanced Algorithms at the
Australian National University. Anthropic's Claude was used as a drafting
and code-review assistant during preparation of the report and the
implementation; all design decisions, claims, and final wording reflect the
author's own judgment.

# Pilot E-v1: "two kinds of compression" (frozen pre-registered pilot)

Implementation of the frozen spec `pilot_v1/SPEC.md` (sha256 in `pilot_v1/SPEC.sha256`, frozen
2026-09-29T12:49:39Z). It tests kill conditions K1 (code dependence) and K2 (degeneracy / collapse
of "constraint compression" into macro compression) for candidate E.

## Regenerate every number

```
cd research/pilot_e/pilot_v1
python3 run_all.py
```

Python 3.11, sympy 1.14, numpy, scipy. It uses 4 worker processes and re-executes itself with
`PYTHONHASHSEED=0`; global seed 20260929. The full run took 692 s on 4 cores (about 190 s of mandatory work plus 505 s
for the optional miner-on Shapley at D1). Ground truth is cached in `pilot_v1/truth.pkl` and
`pilot_v1/truth_scaling.pkl`. To rebuild it, delete both files; `corpus.build_truth()` and
`scaling.truth()` recreate them in about 6 min.

## Outputs

| File | Contents |
|---|---|
| `results.md` | Human-readable tables: verdict, unit tests, per-law × code values, K1, K2-deg-G, Track R / K2-deg-R, K2-collapse labels, 7.3, hypothesis (iii), prior predictions, deviations |
| `results.json` | Everything in `results.md` plus the counts table, Track R cells and per-setting classifications |
| `pilot_v1/results/job_<code>-<level>-B<bits>.json` | Per job: L(S\|T) for all 512 theories, Φ, V^add, V^loo and V^mid (uncharged and EC-charged), per-system φ(f,k), per-system lengths L*_k, law statement costs, miner-on totals for the 20 theories, U3–U6 diagnostics |
| `pilot_v1/results/trackR.json` | Track R cells (codim, I for R1–R4), excluded cells, monomial-tying table, POS, Mach block |
| `pilot_v1/results/analysis.json` | Same as `results.json` |
| `pilot_v1/results/run.log` | Run log |
| `pilot_v1/DEVIATIONS.txt` | Deviations (bug fixes caught by unit tests) and every implementation choice the spec left open |

## Code map (`pilot_v1/`)

| Module | Role |
|---|---|
| `common.py` | Constants, Elias-gamma code |
| `corpus.py` | The 24 systems, ground truth (family G by sympy Euler–Lagrange + `sp.simplify`), `truth.pkl` |
| `trees.py` | Hash-consed trees; C1/C2/C4 profile costs; C3 `sstr` length; sympy conversion |
| `reps.py` | Formulation builders (Newtonian with N2/N3/PC/LC/HK/UG/DIST encodings, Euler–Lagrange with KE, D2 gradient forms, family-G explicit / implicit / residual). Pipeline: C4 lumping, then `sympy.cse(optimizations='basic')`, then greedy inlining, then code length |
| `laws.py` | Law statement schemas and the charged EC decoder (9 rewrite rules) |
| `semantics.py` | Decoders (Newton, Euler–Lagrange, implicit linear solve, Dq/Dt) and the U4/U5 numeric check |
| `lattice.py` | Per (code, level, B) job: fair min over formulations, all 512 theories, exact Shapley, per-system decomposition, C8, EC surplus |
| `miner.py` | Corpus-level greedy tree-grammar miner: exact repeats, Plotkin anti-unification (≤4 holes, nonlinear), C4 constant dictionary, round-trip expansion |
| `trackr.py` | Track R: linearized classes, law equations, codims, R1–R4, SNF lattice index, Bayesian evidence, POS Monte Carlo, Mach block, U7/U8 |
| `scaling.py` | Out-of-corpus scaling series, used by the conditional test 7.3 |
| `analysis.py` | K1, bootstrap, K2-deg-G, K2-deg-R, K2-collapse classification, hypothesis (iii) |
| `tests.py` | U1, U7, U8. U2–U6 are evaluated inside `run_all.py` on the job outputs |
| `report.py` | Writes `results.md` and `results.json` |

Scratch code under `scratch_*` was used only as a starting point; its results are not inputs.

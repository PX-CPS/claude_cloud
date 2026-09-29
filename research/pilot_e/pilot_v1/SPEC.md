# PILOT E-v1: "Two kinds of compression" — frozen pre-registered spec

Status: FROZEN. Nothing in sections 1–9 may change after any number from sections 5–8 has been computed on the real corpus. Before running anything, the implementer:
- saves this text verbatim to `/home/user/claude_cloud/research/pilot_e/pilot_v1/SPEC.md`;
- records its sha256 in `pilot_v1/SPEC.sha256`.

Deviations are allowed only for pure bugs caught by the unit tests in section 9. Each goes in `pilot_v1/DEVIATIONS.txt` with a reason and a timestamp, and must be fixed before any kill statistic is looked at. No git commit or push.

Code may be copied from `research/pilot_e/scratch_grammar-code/` (codes.py, systems.py, pilot.py) and `scratch_model-class/` (linear_codims.py, unknown_mass.py). It must then be extended as specified here. The scratch results are NOT inputs.

Global RNG seed: 20260929. Python 3.11, sympy 1.14, numpy, scipy.

---------------------------------------------------------------------------------------------------

## 0. What the pilot decides

Candidate E claims:
- (a) a law has value as constraint compression, V_c(f) = Σ_k [L(M_k) − L(M_k|f)] − L(f);
- (b) this value is distinct from abbreviation (macro) compression;
- (c) it is robust across codes;
- (d) it is not a trivial count.

The pilot tests kill conditions K1 and K2. K3 (prior work) is judged from the literature and is not decided by this pilot.

There are two readings of L(M|f), and both are implemented:
- Track G (generator / decoder semantics). The law supplies a decoder Φ_f, and L(M|f) = min_θ L(θ) with Φ_f(θ)=M. This is computable on symbolic equations of motion (EOM), and it is what a grammar implementation computes.
- Track R (restriction semantics). L(M|f) = L(M) + log2 π(H_f). This is the only reading under which claim (ii), the Bar-Hillel–Carnap identity, holds. It is implemented in parameter space on linear / linearized model classes.

Global decision rule (pre-registered):
- K1 fires if the K1 test in section 6 fires. It is evaluated on Track G, because Track R's code dependence is known analytically: only residuals beyond codim depend on the code.
- K2 fires if BOTH of these hold:
  - Track R is degenerate (K2-deg-R fires);
  - Track G fails, meaning K2-deg-G fires OR K2-collapse fires.
  In words: V_c must survive K2 under at least one semantics.
- Candidate E is ABANDONED if K1 fires or K2 fires.
- If E survives, it survives only in the semantics that passed, with only the laws classified constraint-type. Claim (ii) survives only if Track R passed.

Recorded prior predictions (made before any run; they do not affect the decision):
- K2-deg-R fires, with R² ≥ 0.99 in R1.
- In Track G, N2, N3, LC, HK, UG, DIST and KE are abbreviation-type.
- PC is mixed.
- EC is constraint-type at D0/D1 when its decoder is uncharged, and abbreviation-type at D2.
- K1 fires on the machine axis, and probably on C4 vs the symbolic codes.
- Judge's prior that E survives everything: about 0.05.

---------------------------------------------------------------------------------------------------

## 1. Definitions (common)

### 1.1 Corpus and models

- Corpus S = {s_1..s_24} (section 2).
- Each system has a ground-truth model M_k: its EOM, derived from full physics.
  - Family P (Cartesian particles, d = 1 or 2): a_i = g_i(x, v, t; params).
  - Family G (generalized coordinates): q'' = h(q, q'; params). The ground truth is obtained by sympy Euler–Lagrange from the listed T and V, or from the listed dissipation for G02, then solved and simplified with `sp.simplify`. It is stored once in `truth.pkl` and used only for semantic checks.

### 1.2 Theory, representation and corpus length (Track G)

- A theory T ⊆ LAW9 = {N2, N3, PC, LC, EC, HK, UG, DIST, KE}.
- A representation rep_φ(M_k) is an ordered list of items (name, tree). Each item is either an output or a named definition (temp).
- Scope rules:
  - Leaves may be state symbols, parameter symbols, earlier temps, or EARLIER OUTPUTS. Output-referencing is a generic base mechanism available to every theory, including ∅.
  - Calls may be base constructors or templates in the library.

Base (generic) mechanisms available to ALL theories, including ∅:
- (i) within-system CSE / DAG sharing, via `sympy.cse` with `optimizations='basic'`, followed by greedy inlining of any temp whose definition cost plus its references exceed its inlined cost;
- (ii) canonical pair-difference orientation: x_a − x_b with a<b, so sign variants of a pair term are shared by CSE through a factor −1;
- (iii) output referencing;
- (iv) the vector constructor V(·,·).

Corpus code length:

  L_c(S|T) = Σ_{f∈T} L_c(f) + [dec_c(T)] + Σ_k L*_c(M_k|T)

  L*_c(M_k|T) = min over formulations φ licensed by T and the machine level of [ L_c(rep_φ(M_k)) ] + log2(#formulations licensed for s_k)

- The fair coder may always fall back to any formulation still licensed.
- dec_c(T) is the charged decoder cost (section 1.5); it is used only in the "charged" variant.
- A model pays Elias-gamma(#items+1) to announce its items.
- Every representation must decode to M_k. Unit test U5 checks this.

### 1.3 Value measures (Track G)

- Shapley value (headline):

  Φ_c(f) = Σ_{U ⊆ LAW9\{f}} |U|!(8−|U|)!/9! · [L_c(S|U) − L_c(S|U∪{f})]

  It is computed exactly over all 512 theories.
- V^add_c(f) = L_c(S|∅) − L_c(S|{f}). This is the proposal's V_c, with the generic mechanisms made explicit.
- V^loo_c(f) = L_c(S|LAW9\{f}) − L_c(S|LAW9).
- V^mid_c(f) = ½(V^add + V^loo). This is the statistic used for classification (section 8).
- Per-system decomposition. Without the miner, L_c(S|T) is additive over systems plus law statements. Therefore:
  - Φ_c(f) = Σ_k φ_c(f,k) − L_c(f), where φ_c(f,k) is the Shapley value of f computed on the per-system lengths L*_c(M_k|·);
  - sanity check: Σ_f Φ_c(f) = L_c(S|∅) − L_c(S|LAW9) (efficiency, unit test U3).

### 1.4 Machine levels (decoder primitives granted to ALL theories)

- D0 (substitution only): the family-G baseline is the explicit solved EOM.
- D1 (CANONICAL): adds a linear-solve primitive for everyone. The family-G baseline may use the implicit form M(q) q'' = f(q,q'), coded as matrix entries plus a vector. The fair min is taken over explicit vs implicit.
- D2: D1 plus differentiation primitives available to everyone as node kinds Func: Dq(e, v) (partial derivative) and Dt(e) (total time derivative along the state).
  - The baseline may define scalar temps and write outputs as, e.g., −Dq(Vtmp, x_i) or Dt(Dq(Ltmp, v_i)) − Dq(Ltmp, q_i).
  - The miner (section 1.6) may therefore rediscover the Euler–Lagrange template as an ordinary substitution macro.

EC licenses the Euler–Lagrange decoder at every level. At D2 that decoder is expressible in the base grammar.

### 1.5 Law statement cost L_c(f) and charged decoder cost

L_c(f) is the statement schema coded in code c, with scope = the formal arguments. Schemas (sympy):

| Law | Schema |
|---|---|
| N2 | F/m |
| N3 | −F |
| PC | Sm(m*a) |
| LC | Sm(Cr(q,F)) |
| EC | Dt(Dv(L)) − Dq(L) and L = T − V |
| HK | template bodies actually used (Fh1, Fh2, Vh1, Vh2 as in scratch `law_items`) |
| UG | Fg, Vg |
| DIST | sqrt(a**2 + b**2) |
| KE | m*w/2 |
| POS | the string "m>0, k>0", coded as 2 relation nodes |

Charged variant (reported for EC at D0 and D1): EC additionally pays dec_c(EC). This is the sum of code lengths of the following 9 rewrite rules. Each rule is coded as a pair of trees in the same code, with pattern variables as leaves in a scope of size 3:
1. d(a+b) → da+db
2. d(a*b) → da*b + a*db
3. d(a**n) → n*a**(n−1)*da
4. d(sin a) → cos a * da
5. d(cos a) → −sin a * da
6. d(v) → 1
7. d(w) → 0
8. d(c) → 0
9. Dt(e) → Σ_q (Dq(e,q)*q' + Dq(e,q')*q'')

No decoder charge applies at D2, where the primitives are base.

### 1.6 Generic corpus-level macro miner (the K2-collapse instrument; "miner-on" codes)

The miner is a greedy tree-grammar compressor, run corpus-wide (cross-system) on top of any theory T.

Input: all items of all systems under T, after CSE.

Candidate generation:
- (a) Every exact subtree of 3–30 nodes occurring ≥2 times corpus-wide (0-hole pattern; cross-system CSE).
- (b) Anti-unification patterns.
  - Group all subtrees of 3–30 nodes by (head kind, arity).
  - For each group, draw up to 50,000 random pairs (seeded; all pairs if fewer).
  - Compute the Plotkin least general generalization. The same differing pair of subtrees maps to the same hole, so nonlinear holes are allowed.
  - Keep patterns with ≥2 non-hole nodes and ≤4 holes.
- (c) In C4 only: a corpus-level constant dictionary. A lumped constant that is symbolically identical in ≥2 systems may be stored once.

Scoring:

  savings(π) = Σ_matches [ L_c(match) − L_c(call_π) − Σ_h L_c(arg_h) ] − L_c(π_def) − Δlib

- Matches: greedy, non-overlapping, preorder, across all trees.
- Matching is ORDERED (sympy canonical arg order), not AC-matching. This limitation is pre-registered.
- Repeated holes must match equal subtrees.
- L_c(call_π) = kind cost + log2(|lib|+1).
- Δlib = (#existing calls) · [log2(|lib|+1) − log2|lib|].

Greedy loop:
- Rank candidates by the heuristic count×size.
- Score the top 200 exactly.
- Add the best one if its savings > 0 and rewrite all matches.
- Templates may be used inside later templates.
- Stop when there is no positive candidate or after 64 iterations.

The miner is deterministic given the seed. L^+_c(S|T) denotes the corpus length with the miner on. V^{+,add}, V^{+,loo} and V^{+,mid} are defined as in section 1.3 with L^+.

Miner runs:
- For each primary code (C1–C4) and each machine level (D0, D1, D2): theories ∅, each {f} (9), LAW9, and each LAW9\{f} (9). That is 20 theories.
- Plus a per-system miner, the miner restricted to one system, for the EC scaling test in 7.3.
- Shapley with the miner on is OPTIONAL and secondary. It is computed only if all miner runs for the other 492 theories finish within 6 h wall time.

### 1.7 Track R definitions (restriction semantics, parameter space)

Model class per eligible system (section 2, column R): the linear / linearized affine second-order class

  q'' = A q + C q' + e (+ drive column u(t))

- Masses are in the known-mass variant, except in the Mach block.
- A law f defines a subset H_f:
  - a linear subspace (equality laws), or
  - a semialgebraic open set (POS).
- The per-system information is I_c(f,k) = −log2 π_c(H_f ∩ ε-tube).
- The corpus value is:

  V^R_c(f) = Σ_{k eligible} I_c(f,k) − L_c(f)

  L_c(f) here is the statement code length under C1 (the same bits for all R codes).

Laws in Track R and their linear equations (implemented from `scratch_model-class/linear_codims.py`):
- EC: M A symmetric, C=0, u=0.
- PC: m^T A = 0 per spatial component; free / isolated systems only.
- LC (planar): the symmetric part of (M⊗J^T)A = 0.
- N3 (1D, pairwise): K = −MA is a weighted graph Laplacian.
- N2, Mach block, corpus-level: bodies {A, B} × springs {kA, kC}. The lumped coefficient matrix c_bs = k_s/m_b is rank 1. Codim = (B−1)(S−1) = 1 on the joint space.
- POS: all masses > 0 and K positive-definite. Codim 0; the value is the volume fraction, estimated by Monte Carlo with 10^5 samples.
- HK, UG, DIST, KE: vacuous in this class (I=0), so V^R = −L(f). They are reported but excluded from the regressions.

---------------------------------------------------------------------------------------------------

## 2. Corpus (24 systems; frozen)

- Shared symbols across systems (body masses mA, mB; spring constants kA, kB, kC) denote the SAME physical body or spring.
- In C4 these are paid once per corpus ONLY when a law licenses cross-system identity:
  - masses are body-attached under N2;
  - spring constants are spring-attached under HK;
  - G is universal under UG;
  - ke and g are paid once per corpus in all theories.
- Otherwise each system pays its own lumped constants.

Legend for applicability: applicable laws are listed; "–" means not applicable. Family G systems admit only EC and KE (plus N2..UG = not applicable by formulation).

| id | name | fam | d | DOF | interactions | applicable laws | Track R |
|---|---|---|---|---|---|---|---|
| P01 | free particle | P | 1 | 1 | none (mA) | N2, EC, KE | – |
| P02 | SHO to wall | P | 1 | 1 | spring wall–A (kA) | N2, EC, HK, KE | yes |
| P03 | damped SHO | P | 1 | 1 | spring kA + damper b | N2, HK (NOT EC, KE) | yes |
| P04 | driven damped SHO | P | 1 | 1 (+t) | spring kA, damper b, drive F0 cos(ωt) | N2, HK (NOT EC) | yes |
| P05 | two masses, three springs, walls | P | 1 | 2 | wall–A kA, A–B kC, B–wall kB | N2, N3, EC, HK, KE | yes |
| P06 | free chain of 4 | P | 1 | 4 | 3 springs, shared k, free ends (m1..m4) | N2, N3, PC, EC, HK, KE | yes |
| P07 | projectile | P | 2 | 2 | uniform field g | N2, EC, KE | – (affine, A=0; reported only) |
| P08 | Kepler, fixed centre | P | 2 | 2 | fixgrav (M0, G) | N2, LC, EC, UG, DIST, KE | – |
| P09 | two-body gravity | P | 2 | 4 | grav A–B | N2, N3, PC, LC, EC, UG, DIST, KE | – |
| P10 | three-body gravity | P | 2 | 6 | grav ×3 (mA, mB, mC) | N2, N3, PC, LC, EC, UG, DIST, KE | – |
| P11 | five-body gravity | P | 2 | 10 | grav ×10 (m1..m5) | as P10 | – |
| P12 | spring triangle | P | 2 | 6 | 3 × 2D springs k1..k3, rest length l | N2, N3, PC, LC, EC, HK, DIST, KE | yes (linearized at equilateral eq.) |
| P13 | elastic pendulum | P | 2 | 2 | 2D spring kA, l to fixed pivot + uniform g | N2, LC (spring term), EC, HK, DIST, KE | yes (linearized at hanging eq.) |
| P14 | cyclotron | P | 2 | 2 | Lorentz force in uniform Bz (charge q) | N2 only (NOT EC: no scalar-potential L; NOT LC) | yes (q'' = C q') |
| P15 | three charges | P | 2 | 6 | Coulomb ×3 (qA, qB, qC, ke) | N2, N3, PC, LC, EC, DIST, KE (not HK/UG: tests template specificity) | – |
| P16 | Mach block I | P | 1 | 1 | body B on spring kA to wall | N2, EC, HK, KE | yes |
| P17 | Mach block II | P | 1 | 1 | body A on spring kC to wall | N2, EC, HK, KE | yes |
| G01 | simple pendulum | G | – | 1 | m, l, g | EC, KE | yes |
| G02 | damped pendulum | G | – | 1 | + linear damping b | (NOT EC, KE) | yes |
| G03 | double pendulum | G | – | 2 | m1, m2, l1, l2, g | EC, KE | yes |
| G04 | triple pendulum | G | – | 3 | m1..m3, l1..l3, g | EC, KE | yes |
| G05 | Atwood machine | G | – | 1 | m1, m2, g | EC, KE | yes (A=0) |
| G06 | bead on rotating hoop | G | – | 1 | m, R, Ω, g (time-independent L) | EC, KE | yes (stable eq.) |
| G07 | cart–pendulum | G | – | 2 | cart M, bob m, l, g | EC, KE | yes |

Additional notes:
- EC-inapplicable systems: P03, P04, P14, G02.
- PC-applicable (isolated): P06, P09, P10, P11, P12, P15.
- DOF range: 1–10.
- Mach block: bodies {A,B} × springs {kA,kC} appear across P02, P05, P16, P17, so the rank-1 structure is testable.

Scaling series (out-of-corpus; used only in 7.3 and reported):
- 2D N-body gravity, N = 2..6;
- free 1D spring chain, n = 2..10 (shared k);
- N-link pendulum, N = 1..4.

---------------------------------------------------------------------------------------------------

## 3. Laws (lattice LAW9 plus one outside-lattice control)

| Law | Role |
|---|---|
| N2 | physical law |
| N3 | physical law |
| PC | physical law |
| LC | physical law |
| EC | physical law |
| HK | physical law |
| UG | physical law |
| DIST | abbreviation control |
| KE | abbreviation control (non-creative definition) |
| POS | constraint control (codim-0 inequality) |

- POS is evaluated outside the lattice, in C4 and Track R only. Its value is additive: 1 bit saved per sign-known lumped constant, minus L(POS).
- Encodings "with" and "without" are given in the `laws` field. They follow the scratch `systems.py` builders, extended to the new systems.

---------------------------------------------------------------------------------------------------

## 4. Codes

Primary codes for K1 (Track G, at D1, CSE on, miner off unless stated):

- **C1, uniform node code.**
  - Node kind uniform over {Add, Mul, Pow, Func, Call, Sym, Int, Rat}: 3 bits.
  - n-ary Add/Mul arity: Elias-gamma(n−1).
  - Leaf: log2|system scope| (state ∪ params ∪ temps ∪ earlier outputs).
  - Func: log2|builtins| (Dq and Dt count as builtins at D2).
  - Call: log2|library|.
  - Int n: gamma(|n|+1)+1.
  - Rat p/q: gamma(|p|+1)+1+gamma(q).
- **C2, static frequency-weighted code.**
  - Kind, arity and leaf-role (state / param / temp / output) probabilities are Laplace-fitted on the all-explicit ∅-theory representations at the same machine level.
  - Leaf = −log2 p(role) + log2|role set|.
- **C3, plain-text code.**
  - Characters of `sp.sstr` with spaces removed, times log2(95).
  - Temps and templates get fixed 2-character names.
  - Each definition pays name + '='.
  - Law statements pay their text + 3 characters.
- **C4, numeric-lumped code.**
  - Maximal state-free parameter subtrees become one real constant each (identified symbolically after `sp.cancel`).
  - B = 16 bits per distinct constant per system, except law-licensed corpus-level constants (section 2).
  - The rest of the tree is coded as C1.

Secondary codes (reported; they enter decisions only where stated):
- C4-B8, C4-B32 (constant precision).
- C5 = C1 with CSE off. This is a probe of baseline strength and is excluded from K1.
- C8 = lzma (preset 9 | EXTREME) over the canonical `srepr` of the whole corpus representation, in bits, with law statements concatenated. This is a model-agnostic Kolmogorov proxy.
- Miner-on versions C1+ to C4+ (section 1.6).

Machine axis: D0 / D1 / D2, plus the charged-decoder variant for EC (section 1.4–1.5).

Track R codes:
- **R1, isotropic Gaussian.** θ ~ N(0, σ²I), cell ε, b = log2(σ/ε) ∈ {4, 8, 16}; b = 8 is primary. The exact formula for linear subspaces is I = r·b + (r/2) log2(2π).
- **R2, anisotropic Gaussian.** Per-slot σ equals the product of the physical scales of the parameters in that slot. Scales are drawn log-uniform on [0.1, 10] (seed). I = r log2(1/ε) + ½ log2 det(2π Σ_⊥).
- **R3, bounded-integer uniform.** |c| ≤ 100. I = r log2(201) + log2(lattice index), computed exactly by Smith normal form.
- **R4, Bayesian evidence (Laplace / BIC).** Simulated trajectories: N ∈ {100, 1000} samples (primary N = 1000), noise σ = 1e−3. Parameters are drawn log-uniform on [0.5, 2] (seed). I = log2 BF.

---------------------------------------------------------------------------------------------------

## 5. Outputs to compute (all written to `pilot_v1/results/*.json`)

For each primary and secondary code, each machine level, and miner off/on:
- L(S|T) for all 512 theories (miner-on: the 20 theories only);
- Φ, V^add, V^loo, V^mid;
- the per-system φ(f,k);
- counts r_fk, u_fk, P_k, n_k (section 7.1).

Also:
- the EC charged-decoder variants;
- Track R: I_c(f,k) and V^R_c(f) for R1–R4, with the codim table and a monomial-tying check. For each equality law, report whether RREF gives each slot as a rational multiple of a single free parameter;
- bootstrap: 1000 resamples of systems with replacement, using the per-system decomposition (miner off). Recompute Φ for each resample.

---------------------------------------------------------------------------------------------------

## 6. K1 test (code dependence) — pre-registered

- Items ranked: the 9 LAW9 items by Φ_c (headline).
- The 7-law ranking is reported but is not used for the decision.

**K1-primary** (codes C1, C2, C3, C4 at D1, miner off). K1 fires if EITHER:
- (a) the minimum pairwise Spearman ρ over the 6 code pairs is < 0.7 (point estimate on the full corpus); OR
- (b) there is a top-law flip. For some pair (a, b), top_a = argmax Φ_a satisfies Φ_b(top_a) < 0.9·max_f Φ_b(f). This 10% tie margin is the only tolerance.

**K1-machine** (C1 at D0, D1, D2, uncharged). Same criteria (a) and (b) over the 3 pairs.

**K1-precision** (C4 at B = 8, 16, 32). Criterion (b) only.

Decision:
- K1 FIRES if K1-primary fires, OR K1-machine fires, OR K1-precision fires.
- Rationale: the reference machine and parameter precision are part of "a reasonable code". This is the Kolmogorov invariance problem, which is what K1 probes.

Reported but not decisive:
- C8 vs C1..C4;
- miner-on vs miner-off;
- the charged-decoder variant;
- bootstrap 95% CIs of each ρ and P(top flip).

If the point estimates pass but bootstrap P(K1-primary fires) > 0.5, the result is labelled "K1-fragile". This is not a kill.

---------------------------------------------------------------------------------------------------

## 7. K2-degeneracy tests — pre-registered

### 7.1 Counts (defined per law f and system k; frozen proxies)

- n_k = DOF.
- P_k = number of parameter symbols.
- K_f = number of corpus systems where f is applicable.

r_fk ("DOF removed"):

| Law | r_fk |
|---|---|
| N2 | 0 (vacuous per system, H1) |
| N3 | d · #pair interactions |
| PC | d if isolated, else 0 |
| LC | (d−1) · #central terms |
| EC | n_k(n_k−1)/2 + [family G: 0; family P: 0] (the symmetric-matrix codim) |
| HK | #springs |
| UG | 2 · #gravity terms |
| DIST | 0 |
| KE | 0 |

u_fk ("uses"):

| Law | u_fk |
|---|---|
| N2 | #force terms |
| N3 | #pair terms |
| PC | 1 if isolated |
| LC | #central terms |
| EC | 1 if applicable |
| HK | #springs |
| UG | #gravity terms |
| DIST | #distinct distances |
| KE | #kinetic terms |

Only cells where f is applicable to k enter the regressions.

### 7.2 K2-deg-G (Track G, per-cell, miner off, D1)

Model M1:

  φ_c(f,k) ~ law fixed effects + β_r·r_fk + β_u·u_fk + β_P·P_k + β_n·n_k   (OLS)

- Score: leave-one-SYSTEM-out cross-validated R² (CV-R²).
- Fires if CV-R²(M1) > 0.9 in ≥ 3 of the 4 primary codes.

Law-level fit (reported; decisive only if extreme):

  Φ_c(f) ~ Σ_k r_fk + K_f + Σ_{k applicable} P_k

- n = 9, leave-one-law-out CV.
- Fires if CV-R² > 0.9 in ≥ 3 of 4 codes.

K2-deg-G = M1 fires OR the law-level fit fires.

### 7.3 EC scaling degeneracy (only if EC is constraint-type in section 8)

- Data: per-system miner-on surplus s_k(EC) = L^+(M_k | T_EC−) − L^+(M_k | T_EC+).
  - T_EC+ = the theory with all laws.
  - T_EC− = the same theory without EC.
  - The miner is run on that system alone.
- Points: all EC-applicable corpus systems plus the scaling series.
- Model: regress on {n_k, #interaction pairs_k, P_k, family indicator}.
- EC counts as count-degenerate if leave-one-system-out CV-R² > 0.9 in ≥ 3 of 4 codes.
- If so, EC is removed from the constraint-type set for the purpose of K2-collapse.

### 7.4 K2-deg-R (Track R)

- Per-cell regression: I_c(f,k) ~ r_fk (single regressor, no intercept per law), over the equality laws EC, PC, LC, N3 on eligible systems.
- Law-level regression: V^R_c(f) ~ Σ_k r_fk.
- Fires if the per-cell R² > 0.9 in ≥ 3 of the 4 R codes (R1 at b=8, R2, R3, R4 at N=1000).
- Predicted analytically: R² = 1 exactly in R1 (Prop 2b), and ≥ 0.9 in the others.

Reported but not decisive:
- POS: codim 0, value = −log2(volume fraction);
- the Mach-block N2 value (B−1)(S−1)·b;
- the monomial-tying table.

---------------------------------------------------------------------------------------------------

## 8. K2-collapse test (constraint vs abbreviation) — pre-registered

Settings: code c ∈ {C1, C2, C3, C4}, machine D1, EC decoder CHARGED (primary). Also reported: uncharged, D0 and D2.

- V_mid(f) = miner-off V^mid.
- V^+_mid(f) = miner-on V^{+,mid}.
- Threshold τ_c = 2 × the median over LAW9 of L_c(f).

Per-law classification in code c:
- **no-value:** V_mid(f) ≤ 0.
- **abbreviation-type:** V^+_mid(f) < τ_c OR V^+_mid(f) < 0.2·V_mid(f).
- **constraint-type:** V^+_mid(f) ≥ τ_c AND V^+_mid(f) ≥ 0.5·V_mid(f).
- **mixed:** otherwise.

A law's label is the label it receives in ≥ 3 of the 4 codes. Otherwise it is "unstable".

POS is classified with C4 only (+ miner dictionary). Since the miner cannot express sign bits, it is predicted constraint-type.

Validity gate (the classifier must separate the controls). The gate passes if ALL of the following hold:
- DIST is abbreviation-type or no-value;
- KE is abbreviation-type or no-value;
- POS is constraint-type.

If the gate fails, K2-collapse is reported as "INDETERMINATE". The classifier may not be retuned in this round. The analytic results (formalization A, P1; formalization B, Props 4–5) then stand as the operative verdict for the substitution-decoder laws.

K2-collapse FIRES iff:
- the gate passes, AND
- after applying 7.3, NO physical law in {N2, N3, PC, LC, EC, HK, UG} is constraint-type at the primary setting (D1, charged).

Structural confirmation (reported, not decisive):
- EC's label at D2, where differentiation is base and Euler–Lagrange is minable as a template, is predicted to be abbreviation-type.
- If EC is constraint-type at D1 but abbreviation-type at D2, its "constraint" content is attributable to the reference machine lacking differentiation, not to the law.

Hypothesis (iii) check (reported):
- Spearman correlation between κ(f) = V^+_mid / V_mid and the hypothesized order: conservation laws (EC, PC, LC) > force laws (HK, UG, N2, N3) > definitions (DIST, KE).
- Hypothesis (iii) is REFUTED if PC or LC is abbreviation-type, or if a definition outranks any conservation law in κ.

---------------------------------------------------------------------------------------------------

## 9. Unit tests (must pass before any statistic is inspected)

- **U1.** gamma_len(1..8) = 1, 3, 3, 5, 5, 5, 5, 7. The C1 cost of `x0*k - m1` is hand-checked.
- **U2.** L_c(S|T) is deterministic across two runs.
- **U3.** Shapley efficiency: Σ_f Φ_c(f) = L_c(S|∅) − L_c(S|LAW9) within 1e−6.
- **U4.** EC decoder: Euler–Lagrange on every EC-applicable Lagrangian representation reproduces the ground-truth EOM.
- **U5.** Semantic check: EVERY representation under EVERY theory and machine level, including miner-rewritten ones, decodes to M_k. Test: numeric evaluation at 5 random state/parameter points (log-uniform [0.5, 2]); max relative error < 1e−9.
- **U6.** Miner round trip: expanding all templates gives back the input trees exactly.
- **U7.** The Track R codim table reproduces formalization B for n = 1..5, d = 1, 2:
  - E: N(N−1)/2
  - P: dN
  - N3 (1D): n(n+1)/2
  - L: N(N+1)/2
  - E∧P = E∧T
- **U8.** Monte Carlo POS reproduces 1.00 ± 0.05 bits for A = −M⁻¹K at n = 2, and 4.00 ± 0.1 bits with K > 0.

Runtime cap: 12 h total. If exceeded, drop the secondary items in this order, then report:
1. miner-on Shapley;
2. C8;
3. D0 miner runs;
4. bootstrap reduced to 300 resamples.

The decision statistics in sections 6–8 are never dropped.

---------------------------------------------------------------------------------------------------

## 10. Final report format (one JSON plus a printed table; no prose-only conclusions)

- `verdict`: {K1: fired / not, with the sub-tests that fired; K2-deg-R; K2-deg-G; K2-collapse (fired / not / INDETERMINATE); K2 overall; candidate: ABANDON / SURVIVES-NARROW(semantics, constraint-type laws)}.
- `hypothesis_iii`: refuted / not.
- `prior_predictions_hit`: list.
- All deviations.

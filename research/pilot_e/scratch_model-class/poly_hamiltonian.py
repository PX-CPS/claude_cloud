"""Polynomial vector fields of degree <= D on R^{2d} (d DOF) vs Hamiltonian ones (X = J grad H, deg H <= D+1).
Checks the closed form codim and the asymptotic fraction; also the integer-lattice residual for d=1.
"""
from math import comb, log2
import itertools
import sympy as sp


def P_gen(d, D):
    return 2 * d * comb(2 * d + D, D)


def P_ham(d, D):
    return comb(2 * d + D + 1, D + 1) - 1  # constant term of H irrelevant


def codim_by_rank(d, D):
    q = sp.symbols(f"q0:{d}"); p = sp.symbols(f"p0:{d}")
    z = list(q) + list(p)
    monsH = [sp.Mul(*[v ** e for v, e in zip(z, ex)]) for ex in itertools.product(range(D + 2), repeat=2 * d)
             if 0 < sum(ex) <= D + 1]
    monsX = [sp.Mul(*[v ** e for v, e in zip(z, ex)]) for ex in itertools.product(range(D + 1), repeat=2 * d)
             if sum(ex) <= D]
    idx = {mm: i for i, mm in enumerate(monsX)}
    cols = []
    for h in monsH:
        vec = [0] * (2 * d * len(monsX))
        comps = [sp.diff(h, pi) for pi in p] + [-sp.diff(h, qi) for qi in q]
        for c, expr in enumerate(comps):
            for term in sp.Add.make_args(sp.expand(expr)):
                if term == 0:
                    continue
                coeff, mon = term.as_coeff_Mul()
                vec[c * len(monsX) + idx[mon]] += coeff
        cols.append(vec)
    M = sp.Matrix(cols)
    return 2 * d * len(monsX) - M.rank()


print("d D  P_gen  P_ham  codim(formula)  codim(rank)  codim/P_gen")
for d in (1, 2):
    for D in (1, 2, 3):
        cf = P_gen(d, D) - P_ham(d, D)
        cr = codim_by_rank(d, D)
        print(d, D, P_gen(d, D), P_ham(d, D), cf, cr, round(cf / P_gen(d, D), 3))
print("asymptotics codim/P_gen:")
for d in (1, 3, 10):
    print(" d=", d, [round(1 - P_ham(d, D) / P_gen(d, D), 3) for D in (1, 2, 5, 20, 100)])
print("1 DOF closed form codim = D(D+1)/2:", [(D, P_gen(1, D) - P_ham(1, D), D * (D + 1) // 2) for D in range(1, 7)])

# integer lattice residual, d=1, D=2: coefficients of (f,g) integers in [-C,C].
# divergence-free <=> three independent pair constraints: f10+g01=0, 2f20+g11=0, f11+2g02=0
for C in (1, 3, 10, 100, 1000):
    tot = (2 * C + 1) ** 6
    law = (2 * C + 1) * (2 * (C // 2) + 1) ** 2
    I = log2(tot / law)
    print(f"C={C}: I = {I:.3f} bits; codim*log2(2C+1) = {3*log2(2*C+1):.3f}; residual = {I-3*log2(2*C+1):.3f}")

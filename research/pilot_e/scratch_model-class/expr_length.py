"""Symbolic-grammar code lengths: vector field X vs Hamiltonian H (decoder code), vs first-order CSE (macro code).
size = number of nodes in the expression tree (atoms and operators count 1; a CSE name costs 1 per use + def).
"""
import sympy as sp


def size(e):
    if e.is_Atom:
        return 1
    return 1 + sum(size(a) for a in e.args)


def cse_size(exprs):
    reps, red = sp.cse(exprs, optimizations=None)
    return sum(size(v) + 1 for _, v in reps) + sum(size(r) for r in red), len(reps)


def ham_field(H, qs, ps):
    return [sp.diff(H, p) for p in ps] + [-sp.diff(H, q) for q in qs]


def report(name, H, qs, ps):
    X = ham_field(H, qs, ps)
    X = [sp.simplify(x) for x in X] if len(qs) < 9 else X
    LX = sum(size(x) for x in X)
    LH = size(H)
    LC, nrep = cse_size(X)
    LCH, _ = cse_size([H])
    print(f"{name:24s} L(X)={LX:5d} L(H)={LH:5d} save_H={LX-LH:5d} | L_cse(X)={LC:5d} save_cse={LX-LC:5d} (#macros={nrep}) | L_cse(H)={LCH:4d}  H-over-CSE={LC-LCH:5d}")
    return LX, LH, LC


def vec(name, i):
    return sp.Matrix(sp.symbols(f"{name}{i}_x {name}{i}_y {name}{i}_z", real=True))


# 1D harmonic oscillator, pendulum
x, p, m, k, g, l = sp.symbols("x p m k g l", positive=True)
report("1D harmonic oscillator", p ** 2 / (2 * m) + k * x ** 2 / 2, [x], [p])
report("pendulum", p ** 2 / (2 * m * l ** 2) - m * g * l * sp.cos(x), [x], [p])

# 1D chain, nearest-neighbour springs, fixed ends
for n in (2, 4, 8):
    q = sp.symbols(f"q0:{n}"); P = sp.symbols(f"p0:{n}"); ms = sp.symbols(f"m0:{n}", positive=True)
    ks = sp.symbols(f"k0:{n+1}", positive=True)
    H = sum(P[i] ** 2 / (2 * ms[i]) for i in range(n))
    H += ks[0] * q[0] ** 2 / 2 + ks[n] * q[n - 1] ** 2 / 2 + sum(ks[i + 1] * (q[i + 1] - q[i]) ** 2 / 2 for i in range(n - 1))
    report(f"1D spring chain n={n}", H, q, P)

# 3D N-body gravity
G = sp.Symbol("G", positive=True)
res = []
for N in (2, 3, 4, 5):
    qs, ps, ms = [], [], sp.symbols(f"m0:{N}", positive=True)
    Q = [vec("q", i) for i in range(N)]; Pm = [vec("p", i) for i in range(N)]
    for i in range(N):
        qs += list(Q[i]); ps += list(Pm[i])
    H = sum((Pm[i].dot(Pm[i])) / (2 * ms[i]) for i in range(N))
    H -= sum(G * ms[i] * ms[j] / sp.sqrt((Q[i] - Q[j]).dot(Q[i] - Q[j])) for i in range(N) for j in range(i + 1, N))
    res.append((N,) + report(f"3D gravity N={N}", H, qs, ps))
print("gravity: save_H vs pairs N(N-1)/2:", [(r[0], r[1] - r[2], r[0] * (r[0] - 1) // 2) for r in res])

# abbreviation control: T = m v^2/2 as a definition inside a corpus of Lagrangians/energies
v = sp.symbols("v0:6"); ms6 = sp.symbols("M0:6", positive=True)
Ts = [ms6[i] * v[i] ** 2 / 2 for i in range(6)]
print("abbreviation control T=(1/2)m v^2: body size", size(ms6[0] * v[0] ** 2 / 2), "citation T(m,v) size", 3)

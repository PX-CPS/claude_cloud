"""K2 probes: does EC (Lagrangian/Hamiltonian) savings scale like a count?
(a) N-link planar pendulum, N=1..3: explicit EOM vs implicit (M qdd = f) vs Lagrangian.
(b) N-body planar gravity, N=2..5: best Newtonian representation vs Lagrangian (C1 code)."""
import sys, time, itertools
import sympy as sp
import codes as C
import systems as Y
import pilot as P


def c1_items(items, nscope, nlib=3):
    return sum(C.c1_tree(e, nscope, nlib) for e in items)


def cse_c1(exprs, nscope, nlib=3):
    items = [(None, e) for e in exprs]
    f = lambda it: C.gamma_len(sum(1 for n, _ in it if n is not None) + 1) + \
        sum(C.c1_tree(e, nscope + sum(1 for n, _ in it if n is not None), nlib) for _, e in it)
    new = C.apply_cse(items, f)
    return f(new)


def npend(N, simplify=True):
    th = sp.symbols('th1:%d' % (N + 1))
    om = sp.symbols('om1:%d' % (N + 1))
    al = sp.symbols('al1:%d' % (N + 1))
    m = sp.symbols('m1:%d' % (N + 1))
    l = sp.symbols('l1:%d' % (N + 1))
    g = sp.Symbol('g')
    x = y = 0
    vx = vy = 0
    T = 0
    Vp = 0
    for i in range(N):
        x = x + l[i] * sp.sin(th[i]); y = y - l[i] * sp.cos(th[i])
        vx = vx + l[i] * om[i] * sp.cos(th[i]); vy = vy + l[i] * om[i] * sp.sin(th[i])
        T += m[i] * sp.expand(vx ** 2 + vy ** 2) / 2
        Vp += m[i] * g * y
    T = sp.trigsimp(sp.expand(T))
    Lg = T - Vp
    eqs = []
    for i in range(N):
        dLdv = sp.diff(Lg, om[i])
        dt = sum(sp.diff(dLdv, th[j]) * om[j] + sp.diff(dLdv, om[j]) * al[j] for j in range(N))
        eqs.append(sp.expand(dt - sp.diff(Lg, th[i])))
    Mm = sp.Matrix(N, N, lambda i, j: sp.trigsimp(eqs[i].coeff(al[j])))
    fv = sp.Matrix(N, 1, lambda i, _: sp.trigsimp(-(eqs[i] - sum(eqs[i].coeff(al[j]) * al[j] for j in range(N)))))
    nscope = 3 * N + 1
    L_lag = cse_c1([Lg], nscope)
    implicit = [e for e in list(Mm) if e != 0] + list(fv)
    L_imp = cse_c1([sp.factor(e) for e in implicit], nscope)
    t0 = time.time()
    sol = Mm.LUsolve(fv)
    sol = [sp.factor(sp.trigsimp(sp.cancel(e))) if simplify else e for e in sol]
    L_exp = cse_c1(sol, nscope)
    return dict(N=N, L_lagrangian=L_lag, L_implicit=L_imp, L_explicit=L_exp, t=time.time() - t0)


def nbody(N):
    inter = [('grav', i, j) for i, j in itertools.combinations(range(1, N + 1), 2)]
    s = Y.PSys('nb%d' % N, 2, N, inter)
    newt = frozenset(['N2', 'N3', 'PC', 'LC', 'UG', 'DIST'])
    ec = frozenset(['EC', 'UG', 'DIST', 'KE'])
    newt_noUG = frozenset(['N2', 'N3', 'PC', 'LC', 'DIST'])
    ec_noUG = frozenset(['EC', 'DIST', 'KE'])
    out = dict(N=N)
    for code in ('C1', 'C3'):
        out[code] = dict(newton=P.rep_cost(s, s.relevant(newt), code)[0],
                         lagr=P.rep_cost(s, s.relevant(ec), code)[0],
                         newton_noUG=P.rep_cost(s, s.relevant(newt_noUG), code)[0],
                         lagr_noUG=P.rep_cost(s, s.relevant(ec_noUG), code)[0],
                         bare=P.rep_cost(s, frozenset(), code)[0])
    return out


if __name__ == '__main__':
    P.fit_freq(Y.corpus(True))
    for N in (2, 3, 4, 5):
        r = nbody(N)
        print('nbody', r, flush=True)
    for N in (1, 2, 3):
        print('npend', npend(N), flush=True)

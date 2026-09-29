"""Mini-pilot: V_c(f) under five codes; leave-one-out, add-one, Shapley; K1/K2 probes."""
import itertools, math, sys as _sys, json, time
import sympy as sp
from sympy.core.function import AppliedUndef
import codes as C
import systems as Y

CODES = ['C1', 'C2', 'C3', 'C4', 'C5']


def libnames(S):
    n = {'V'}
    if 'LC' in S: n.add('Cn')
    if 'DIST' in S: n.add('D')
    if 'KE' in S: n.add('K')
    if 'HK' in S: n |= {'Fh1', 'Fh2', 'Vh1', 'Vh2'}
    if 'UG' in S: n |= {'Fg', 'Vg'}
    return n


def roles_for(sysm, items, extra_consts=()):
    r = {}
    for s in sysm.state: r[s] = 'state'
    for p in sysm.params: r[p] = 'param'
    for c in extra_consts: r[c] = 'param'
    for n, _ in items:
        if n is not None: r[n] = 'temp'
    return r


def tree_cost(code, e, roles, nlib):
    sizes = {}
    for v in roles.values(): sizes[v] = sizes.get(v, 0) + 1
    nscope = len(roles)
    if code in ('C1', 'C5', 'C4'):
        return C.c1_tree(e, nscope, nlib)
    if code == 'C2':
        return C.FREQ.tree(e, roles, sizes, nlib)
    if code == 'C3':
        return C.c3_tree(e)


def items_cost(code, items, sysm, nlib, extra_consts=()):
    roles = roles_for(sysm, items, extra_consts)
    ndefs = sum(1 for n, _ in items if n is not None)
    tot = C.gamma_len(ndefs + 1)
    for n, e in items:
        tot += tree_cost(code, e, roles, nlib)
        if n is not None and code == 'C3':
            tot += (len(str(n)) + 1) * C.LOG95
    return tot


def lump_items(sysm, items, S):
    statelike = set(sysm.state) | {n for n, _ in items if n is not None}
    table = {}
    new = [(n, C.lump(e, statelike, table)) for n, e in items]
    consts = dict(table)  # expr -> sym
    if 'N2' in S and sysm.family == 'P' and not ('EC' in S and sysm.conservative):
        for i in sysm.bodies:
            if Y.M(i) not in consts:
                consts[Y.M(i)] = sp.Symbol('cm%d' % i)
    return new, consts


def rep_cost(sysm, S, code):
    nlib = len(libnames(S))

    if code == 'C4':
        def cost_term(e):
            it, consts = lump_items(sysm, [(None, e)], set())
            return items_cost('C1', it, sysm, nlib, consts.values()) + C.B_BITS * len(
                [k for k in consts if k not in C.UNIVERSAL])
    else:
        def cost_term(e):
            return items_cost(code, [(None, e)], sysm, nlib)

    items = Y.build(sysm, S, cost_term)
    if code == 'C4':
        items, consts = lump_items(sysm, items, S)
        univ = {k for k in consts if k in C.UNIVERSAL}
        loc = [k for k in consts if k not in C.UNIVERSAL]

        def cf(it):
            return items_cost('C1', it, sysm, nlib, consts.values())
        items = C.apply_cse(items, cf)
        bits = cf(items) + C.B_BITS * len(loc)
    elif code == 'C5':
        bits = items_cost('C1', items, sysm, nlib)
        univ = set()
    else:
        cf = lambda it: items_cost(code, it, sysm, nlib)
        items = C.apply_cse(items, cf)
        bits = cf(items)
        univ = set()
    used = set()
    for _, e in items:
        for a in e.atoms(AppliedUndef):
            used.add(a.func.__name__)
    return bits, used, univ


CACHE = {}


def _rc(sysm, R, code):
    key = (sysm.name, R, code)
    if key not in CACHE:
        CACHE[key] = rep_cost(sysm, R, code)
    return CACHE[key]


def sys_cost(sysm, S, code):
    # fair coder: when EC is in the theory it may still use the Newtonian formulation
    # (min over formulations permitted by S, plus 1 bit to say which one).
    R = sysm.relevant(S)
    best = _rc(sysm, R, code)
    if 'EC' in R:
        alt = _rc(sysm, sysm.relevant(frozenset(S) - {'EC'}), code)
        best = min(best, alt, key=lambda x: x[0])
        best = (best[0] + 1.0,) + tuple(best[1:])
    return best


def law_cost(f, S, used, code):
    items, formals = Y.law_items(f, S, used)
    nlib = len(libnames(S)) + 5
    roles = {s: 'param' for s in formals}
    tot = 0.0
    for _, e in items:
        if code == 'C3':
            tot += C.c3_tree(e) + 3 * C.LOG95
        elif code == 'C2':
            sizes = {'param': len(formals)}
            tot += C.FREQ.tree(e, roles, sizes, nlib)
        else:
            tot += C.c1_tree(e, len(formals), nlib)
    return tot


def total(corp, S, code, detail=False):
    tot = 0.0
    used, univ = set(), set()
    per = {}
    for sysm in corp:
        b, u, v = sys_cost(sysm, S, code)
        per[sysm.name] = b
        tot += b
        used |= u
        univ |= v
    for f in S:
        tot += law_cost(f, S, used, code)
    tot += C.B_BITS * len(univ)
    return (tot, per) if detail else tot


def fit_freq(corp):
    trees = []
    for sysm in corp:
        items = Y.build(sysm, set(), lambda e: 0)
        r = roles_for(sysm, items)
        trees += [(e, r) for _, e in items]
    C.FREQ.fit(trees)


def spearman(a, b):
    import scipy.stats as st
    return st.spearmanr(a, b).correlation


def run(include_G=True, laws=Y.LAWS):
    corp = Y.corpus(include_G)
    fit_freq(Y.corpus(True))
    n = len(laws)
    res = {}
    for code in CODES:
        t0 = time.time()
        Lt = {}
        for r in range(n + 1):
            for comb in itertools.combinations(laws, r):
                Lt[frozenset(comb)] = total(corp, frozenset(comb), code)
        full = frozenset(laws)
        loo = {f: Lt[full - {f}] - Lt[full] for f in laws}
        add = {f: Lt[frozenset()] - Lt[frozenset([f])] for f in laws}
        shap = {}
        for f in laws:
            others = [g for g in laws if g != f]
            s = 0.0
            for r in range(n):
                w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                for comb in itertools.combinations(others, r):
                    S = frozenset(comb)
                    s += w * (Lt[S] - Lt[S | {f}])
            shap[f] = s
        res[code] = dict(loo=loo, add=add, shap=shap, L0=Lt[frozenset()], Lfull=Lt[full])
        print(code, 'done in %.1fs' % (time.time() - t0), 'L(empty)=%.0f L(full)=%.0f' % (Lt[frozenset()], Lt[full]), flush=True)
    return res, corp


if __name__ == '__main__':
    incG = '--noG' not in _sys.argv
    res, corp = run(incG)
    tag = 'withG' if incG else 'noG'
    json.dump({c: {k: (v if not isinstance(v, dict) else v) for k, v in d.items()} for c, d in res.items()},
              open('results_%s.json' % tag, 'w'), indent=1, default=float)
    for meas in ('loo', 'add', 'shap'):
        print('\n==', meas, '(bits saved; positive = law helps)')
        print('%-6s' % 'law' + ''.join('%9s' % c for c in CODES))
        for f in Y.LAWS:
            print('%-6s' % f + ''.join('%9.1f' % res[c][meas][f] for c in CODES))
        print('Spearman rho between codes:')
        for a, b in itertools.combinations(CODES, 2):
            va = [res[a][meas][f] for f in Y.LAWS]
            vb = [res[b][meas][f] for f in Y.LAWS]
            print('  %s-%s %.2f' % (a, b, spearman(va, vb)), end='')
        print()
        for c in CODES:
            top = max(Y.LAWS, key=lambda f: res[c][meas][f])
            print('  top[%s]=%s' % (c, top), end='')
        print()

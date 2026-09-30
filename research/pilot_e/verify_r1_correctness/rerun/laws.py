"""Law statement costs L_c(f) (spec 1.5) and the charged EC decoder cost dec_c(EC).

Implementation choices (not fixed by the spec text):
 - each statement item pays Elias-gamma(#formals+1) for its arity (decodability), then its tree
   coded with leaves = formal arguments (holes) of scope #formals;
 - schema operators (Sm, Cr, Dt, Dv, Dq, d) are Calls with library size |lib(T)|+5 (as scratch);
 - C4 statements are coded as C1; C3 statements pay their sstr text + 3 characters per item.
"""
import math
import sympy as sp
from common import gamma_len, LOG95, builtins_for, CFG
import trees as TR
import reps as R

Sm, Cr, Dt_, Dv, Dq_, dd = [sp.Function(n) for n in ('Sm', 'Cr', 'Dt', 'Dv', 'Dq', 'd')]
V, D = TR.V, TR.D


def law_items(f, dist_on, used_templates):
    """Return list of (expr, formals)."""
    a, b, m, F, q, Lg, T, Vv, w, k, l, n = sp.symbols('a b m F q L T Vv w k l n')
    xi, yi, xj, yj, mi, mj, GG = sp.symbols('xi yi xj yj mi mj G')

    def Rd():
        return D(xi - xj, yi - yj) if dist_on else sp.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    if f == 'N2':
        return [(F / m, [F, m])]
    if f == 'N3':
        return [(-F, [F])]
    if f == 'PC':
        return [(Sm(m * a), [m, a])]
    if f == 'LC':
        return [(Sm(Cr(q, F)), [q, F])]
    if f == 'EC':
        return [(Dt_(Dv(Lg)) - Dq_(Lg), [Lg]), (T - Vv, [T, Vv])]
    if f == 'DIST':
        return [(sp.sqrt(a ** 2 + b ** 2), [a, b])]
    if f == 'KE':
        return [(m * w / 2, [m, w])]
    if f == 'POS':
        return [(sp.StrictGreaterThan(m, 0, evaluate=False), [m]), (sp.StrictGreaterThan(k, 0, evaluate=False), [k])]
    if f == 'HK':
        R_ = Rd()
        cand = [('Fh1', -k * (a - b), [k, a, b]),
                ('Fh2', V(-k * (R_ - l) * (xi - xj) / R_, -k * (R_ - l) * (yi - yj) / R_), [k, l, xi, yi, xj, yj]),
                ('Vh1', k * (a - b) ** 2 / 2, [k, a, b]),
                ('Vh2', k * (R_ - l) ** 2 / 2, [k, l, xi, yi, xj, yj])]
        used = [c for c in cand if c[0] in used_templates] or [cand[0]]
        return [(e, fm) for _, e, fm in used]
    if f == 'UG':
        R_ = Rd()
        return [(V(-GG * mi * mj * (xi - xj) / R_ ** 3, -GG * mi * mj * (yi - yj) / R_ ** 3), [GG, mi, mj, xi, yi, xj, yj]),
                (-GG * mi * mj / R_, [GG, mi, mj, xi, yi, xj, yj])]
    raise ValueError(f)


def item_cost_plain(expr, formals, code, nlib, nbuiltin, text_extra=3):
    """Original (v1) statement item cost: no CSE."""
    holes = {s: j for j, s in enumerate(formals)}
    tid = TR.from_sympy(expr, 'LAW', law_mode=True, holes=holes)
    if code == 'C3':
        return TR.c3_len(tid) * LOG95 + text_extra * LOG95
    if code == 'C6':
        return (TR.latex_len(tid) + text_extra) * TR.LOG128
    tc = 'C1' if code in ('C4', 'C5') else code
    sizes = {'hole': len(formals)}
    return gamma_len(len(formals) + 1) + TR.tree_cost(tid, tc, sizes, nlib, nbuiltin)


def _stmt_items_cost(items, nformals, code, nlib, nbuiltin, text_extra):
    """Cost of a CSE'd statement item: (label, tree) list, temps first, main item last (label None).
    Mirrors reps.items_cost: temps see the formals and the earlier temps; the main item sees all;
    the number of temps is announced with Elias-gamma(#temps+1) (C3/C6: each definition pays its
    name + '=' instead)."""
    ntemps = sum(1 for l, _ in items if l is not None)
    tc = 'C1' if code in ('C4', 'C5') else code
    if code in ('C3', 'C6'):
        unit = LOG95 if code == 'C3' else TR.LOG128
        ln = TR.c3_len if code == 'C3' else TR.latex_len
        tot = 0.0
        for lab, tid in items:
            tot += (ln(tid) + (3 if lab is not None else text_extra)) * unit
        return tot
    tot = gamma_len(nformals + 1) + gamma_len(ntemps + 1)
    ti = 0
    for lab, tid in items:
        if lab is not None:
            sizes = {'hole': nformals, 'temp': ti}
            ti += 1
        else:
            sizes = {'hole': nformals, 'temp': ntemps}
        tot += TR.tree_cost(tid, tc, sizes, nlib, nbuiltin)
    return tot


def item_cost(expr, formals, code, nlib, nbuiltin, text_extra=3):
    """R4: statements are coded with the same base mechanisms as models: sympy CSE ('basic'),
    then greedy inlining of any temp whose definition plus references cost more than inlining."""
    if not CFG['stmt_cse']:
        return item_cost_plain(expr, formals, code, nlib, nbuiltin, text_extra)
    holes = {s: j for j, s in enumerate(formals)}
    if isinstance(expr, sp.Rel):
        repl, red = [], [expr]
    else:
        repl, red = sp.cse([expr], symbols=sp.numbered_symbols('_x'), optimizations='basic')
    items = [(TR.local_label_of(n, 'LAW'), TR.from_sympy(e, 'LAW', law_mode=True, holes=holes)) for n, e in repl]
    items.append((None, TR.from_sympy(red[0], 'LAW', law_mode=True, holes=holes)))
    costfn = lambda its: _stmt_items_cost(its, len(formals), code, nlib, nbuiltin, text_extra)
    items, c = R.greedy_inline(items, costfn)
    return c


def law_cost(f, T, used_templates, code, level):
    nlib = len(R.lib_names(T)) + 5
    nb = len(builtins_for(level))
    return sum(item_cost(e, fm, code, nlib, nb) for e, fm in law_items(f, 'DIST' in T, used_templates))


def dec_rules():
    a, b, n, v, w, c, e, q, qd, qdd = sp.symbols('a b n v w c e q qd qdd')
    return [
        (dd(a + b), dd(a) + dd(b)),
        (dd(a * b), dd(a) * b + a * dd(b)),
        (dd(a ** n), n * a ** (n - 1) * dd(a)),
        (dd(sp.sin(a)), sp.cos(a) * dd(a)),
        (dd(sp.cos(a)), -sp.sin(a) * dd(a)),
        (dd(v), sp.Integer(1)),
        (dd(w), sp.Integer(0)),
        (dd(c), sp.Integer(0)),
        (Dt_(e), Sm(Dq_(e, q) * qd + Dq_(e, qd) * qdd)),
    ]


def dec_rules_cse_check():
    """R4: sympy CSE finds no shared subterm in any rewrite-rule side (so CSE leaves dec_c unchanged)."""
    n = 0
    for lhs, rhs in dec_rules():
        for side in (lhs, rhs):
            repl, _ = sp.cse([side], optimizations='basic')
            n += len(repl)
    return n


def dec_cost(code, level):
    """Charged decoder cost of EC: 9 differentiation rewrite rules, pattern variables in scope 3."""
    nb = len(builtins_for(level))
    tot = 0.0
    for lhs, rhs in dec_rules():
        for side in (lhs, rhs):
            syms = sorted(side.free_symbols, key=str)
            holes = {s: j for j, s in enumerate(syms)}
            tid = TR.from_sympy(side, 'DEC', law_mode=True, holes=holes)
            if code == 'C3':
                tot += TR.c3_len(tid) * LOG95
            elif code == 'C6':
                tot += TR.latex_len(tid) * TR.LOG128
            else:
                tc = 'C1' if code in ('C4', 'C5') else code
                tot += TR.tree_cost(tid, tc, {'hole': 3}, 4, nb)
        if code == 'C3':
            tot += 2 * LOG95   # '->'
        if code == 'C6':
            tot += 2 * TR.LOG128
    return tot

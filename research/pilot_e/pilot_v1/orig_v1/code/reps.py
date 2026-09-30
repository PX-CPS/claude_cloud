"""Representation builders (formulations) and the realization pipeline
(C4 lumping -> sympy CSE('basic') -> trees -> greedy inlining -> code length).

Formulations
  family P: 'newton' (always; at D2 the min also includes the gradient form -Dq(Vtmp,x)/m,
            and for isolated systems without PC the output-referencing form of the last body),
            'el' (EC in theory).
  family G: 'explicit' (always), 'implicit' (D1, D2: residual M z - f; at D2 also the residual
            Dt(Dq(Ltmp,v))-Dq(Ltmp,q)), 'el' (EC in theory).
"""
import itertools, math
import sympy as sp
from sympy.core.function import AppliedUndef
from common import gamma_len, LOG95, builtins_for
import corpus as CP
import trees as TR
from trees import V, Cn, D, K, Fh1, Fh2, Vh1, Vh2, Fg, Vg, Dq, Dt

NEWTON_LAWS = frozenset({'N2', 'N3', 'PC', 'LC', 'HK', 'UG', 'DIST'})
EL_LAWS = frozenset({'HK', 'UG', 'DIST', 'KE'})


def lib_names(S):
    n = {'V'}
    if 'LC' in S:
        n.add('Cn')
    if 'DIST' in S:
        n.add('D')
    if 'KE' in S:
        n.add('K')
    if 'HK' in S:
        n |= {'Fh1', 'Fh2', 'Vh1', 'Vh2'}
    if 'UG' in S:
        n |= {'Fg', 'Vg'}
    return n


def formulations(sysm, level):
    if sysm.fam == 'P':
        return ['newton'] + (['el'] if 'EC' in sysm.applicable else [])
    return ['explicit'] + (['implicit'] if level != 'D0' else []) + (['el'] if 'EC' in sysm.applicable else [])


def form_subset(sysm, form, T):
    """Laws of theory T that the formulation actually uses for this system."""
    A = sysm.applicable & frozenset(T)
    if form == 'newton':
        return A & NEWTON_LAWS
    if form == 'el':
        return A & EL_LAWS
    return frozenset()


def licensed_symbols(sysm, S):
    """C4 corpus-level constants licensed by laws in S (spec section 2)."""
    lic = {CP.g, CP.ke}
    if sysm.fam == 'P':
        if 'N2' in S:
            lic |= set(sysm.masses)
        if 'HK' in S:
            for it in sysm.inter:
                if it[0].startswith('spring'):
                    lic.add(it[3])
        if 'UG' in S:
            lic.add(CP.G)
    return lic


# ---------------------------------------------------------------- family P builders
def _R(dx, dy, S):
    return D(dx, dy) if 'DIST' in S else sp.sqrt(dx ** 2 + dy ** 2)


def explicit_force(sysm, it, i, S):
    """Explicit force comps on body i (canonical pair orientation), and central triple (s,dx,dy) or None."""
    d = sysm.d
    kind = it[0]
    if kind == 'spring1':
        _, a, b, kk = it
        xb = CP.X(b) if b is not None else sp.Integer(0)
        f = -kk * (CP.X(a) - xb)
        return [f if i == a else -f], None
    if kind in ('spring2', 'grav', 'coul', 'fixgrav'):
        a = it[1]
        bb = None if kind == 'fixgrav' else it[2]
        pa, pb = sysm.pos(a), sysm.pos(bb)
        dx, dy = pa[0] - pb[0], pa[1] - pb[1]
        R = _R(dx, dy, S)
        if kind == 'spring2':
            s = -it[3] * (R - it[4]) / R
        elif kind == 'grav':
            s = -CP.G * sysm.M(a) * sysm.M(bb) / R ** 3
        elif kind == 'fixgrav':
            s = -CP.G * it[2] * sysm.M(a) / R ** 3
        else:
            s = CP.ke * it[3] * it[4] / R ** 3
        if i != a:
            s = -s
        return [s * dx, s * dy], (s, dx, dy)
    if kind == 'unif':
        return [sp.Integer(0), -sysm.M(it[1]) * CP.g], None
    if kind == 'damp':
        return [-it[2] * CP.U(it[1])], None
    if kind == 'drive':
        return [it[2] * sp.cos(it[3] * CP.t)], None
    if kind == 'mag':
        _, a, qq, BB = it
        return [qq * BB * CP.W(a), -qq * BB * CP.U(a)], None
    raise ValueError(kind)


def template_force(sysm, it, i, S):
    kind = it[0]
    d = sysm.d
    if kind == 'spring1' and 'HK' in S:
        _, a, b, kk = it
        j = b if i == a else a
        return Fh1(kk, CP.X(i), CP.X(j) if j is not None else sp.Integer(0))
    if kind == 'spring2' and 'HK' in S:
        a, b, kk, l = it[1:]
        j = b if i == a else a
        return Fh2(kk, l, *sysm.pos(i), *sysm.pos(j))
    if kind == 'grav' and 'UG' in S:
        a, b = it[1], it[2]
        j = b if i == a else a
        return Fg(CP.G, sysm.M(i), sysm.M(j), *sysm.pos(i), *sysm.pos(j))
    if kind == 'fixgrav' and 'UG' in S:
        return Fg(CP.G, sysm.M(i), it[2], *sysm.pos(i), *sysm.pos(None))
    return None


def as_vec(comps, d):
    return comps[0] if d == 1 else V(*comps)


def build_newton(sysm, S, cost_term, outref=False):
    d = sysm.d
    N2 = 'N2' in S
    items = []
    terms = {i: [] for i in sysm.bodies}
    expl = {i: [sp.Integer(0)] * d for i in sysm.bodies}
    npair = 0
    for it in sysm.inter:
        bl = sysm.bodies_of(it)
        first = bl[0]
        Pref = None
        for i in bl:
            mi = sysm.M(i)
            if 'N3' in S and CP.is_pair(it) and i != first and Pref is not None:
                terms[i].append(-Pref if N2 else -sysm.M(first) * Pref / mi)
                continue
            comps, cen = explicit_force(sysm, it, i, S)
            if not N2:
                comps = [c / mi for c in comps]
            cands = [('expl', as_vec(comps, d))]
            if 'LC' in S and cen is not None:
                s, dx, dy = cen
                cands.append(('call', Cn(s if N2 else s / mi, dx, dy)))
            tm = template_force(sysm, it, i, S)
            if tm is not None:
                cands.append(('call', tm if N2 else tm / mi))
            best = min(cands, key=lambda c: cost_term(c[1]))
            if 'N3' in S and CP.is_pair(it):
                npair += 1
                Pref = sp.Symbol('_P%d' % npair)
                items.append((Pref, best[1]))
                terms[i].append(Pref)
            elif best[0] == 'expl':
                expl[i] = [expl[i][c] + comps[c] for c in range(d)]
            else:
                terms[i].append(best[1])
    outs = []
    for i in sysm.bodies:
        parts = list(terms[i])
        if any(c != 0 for c in expl[i]):
            parts = [as_vec(expl[i], d)] + parts
        outs.append(sp.Add(*parts) if parts else sp.Integer(0))
    pc = 'PC' in S and sysm.isolated
    if pc:
        outs = outs[:-1]
    elif outref and sysm.isolated:
        n = sysm.n
        O = [sp.Symbol('_O%d' % j) for j in range(n - 1)]
        if N2:
            outs[-1] = -sp.Add(*O)
        else:
            outs[-1] = -sp.Add(*[sysm.M(j + 1) * O[j] for j in range(n - 1)]) / sysm.M(n)
    items += [(None, o) for o in outs]
    return items, dict(form='newton', n2=N2, pc=pc, variant='outref' if outref else 'std')


def potential_items(sysm, S, cost_term):
    """Potential terms with per-term template min (HK: Vh1/Vh2, UG: Vg)."""
    d = sysm.d
    pots = []
    for it in sysm.inter:
        kind = it[0]
        if kind == 'spring1':
            _, a, b, kk = it
            xb = CP.X(b) if b is not None else sp.Integer(0)
            ex = kk * (CP.X(a) - xb) ** 2 / 2
            tm = Vh1(kk, CP.X(a), xb) if 'HK' in S else None
        elif kind in ('spring2', 'grav', 'coul', 'fixgrav'):
            a = it[1]
            bb = None if kind == 'fixgrav' else it[2]
            pa, pb = sysm.pos(a), sysm.pos(bb)
            R = _R(pa[0] - pb[0], pa[1] - pb[1], S)
            tm = None
            if kind == 'spring2':
                ex = it[3] * (R - it[4]) ** 2 / 2
                if 'HK' in S:
                    tm = Vh2(it[3], it[4], *pa, *pb)
            elif kind == 'grav':
                ex = -CP.G * sysm.M(a) * sysm.M(bb) / R
                if 'UG' in S:
                    tm = Vg(CP.G, sysm.M(a), sysm.M(bb), *pa, *pb)
            elif kind == 'fixgrav':
                ex = -CP.G * it[2] * sysm.M(a) / R
                if 'UG' in S:
                    tm = Vg(CP.G, sysm.M(a), it[2], *pa, *pb)
            else:
                ex = CP.ke * it[3] * it[4] / R
        elif kind == 'unif':
            ex = sysm.M(it[1]) * CP.g * CP.Y(it[1])
            tm = None
        else:
            continue
        if tm is not None and cost_term(tm) < cost_term(ex):
            ex = tm
        pots.append(ex)
    return pots


def build_el_P(sysm, S, cost_term):
    d = sysm.d
    kin = []
    for i in sysm.bodies:
        w2 = CP.U(i) ** 2 + (CP.W(i) ** 2 if d == 2 else 0)
        kin.append(K(sysm.M(i), w2) if 'KE' in S else sysm.M(i) * w2 / 2)
    pots = potential_items(sysm, S, cost_term)
    return [(None, sp.Add(*kin) - sp.Add(*pots))], dict(form='el', n2=False, pc=False, variant='std')


def build_gradient_P(sysm, S, cost_term):
    """D2 gradient form: Vtmp = sum of potentials; a_i = -Dq(Vtmp, x_i)/m_i + nonconservative."""
    d = sysm.d
    N2 = 'N2' in S
    pots = potential_items(sysm, S, cost_term)
    if not pots:
        return None
    Vt = sp.Symbol('_Vt')
    items = [(Vt, sp.Add(*pots))]
    outs = []
    for i in sysm.bodies:
        comps = [-Dq(Vt, c) for c in sysm.pos(i)]
        for it in sysm.inter:
            if it[0] in CP.CONS or i not in sysm.bodies_of(it):
                continue
            f, _ = explicit_force(sysm, it, i, S)
            comps = [comps[c] + f[c] for c in range(d)]
        if not N2:
            comps = [c / sysm.M(i) for c in comps]
        outs.append(as_vec(comps, d))
    pc = 'PC' in S and sysm.isolated
    if pc:
        outs = outs[:-1]
    items += [(None, o) for o in outs]
    return items, dict(form='newton', n2=N2, pc=pc, variant='gradient')


# ---------------------------------------------------------------- family G builders
def build_explicit_G(sysm, truth):
    return [(None, e) for e in truth[sysm.sid]['acc']], dict(form='explicit', variant='std')


def build_implicit_G(sysm, truth):
    Mm, f = truth[sysm.sid]['M'], truth[sysm.sid]['f']
    outs = []
    for i in range(sysm.n):
        outs.append(sp.Add(*[Mm[i][j] * sysm.z[j] for j in range(sysm.n)]) - f[i])
    return [(None, o) for o in outs], dict(form='implicit', variant='matrix')


def build_gradres_G(sysm):
    Lt = sp.Symbol('_Lt')
    T = sp.Add(*[m_ * w_ / 2 for m_, w_ in sysm.kin_terms])
    items = [(Lt, T - sysm.Vp)]
    for i in range(sysm.n):
        r = Dt(Dq(Lt, sysm.v[i])) - Dq(Lt, sysm.q[i])
        if sysm.diss is not None:
            r = r + sp.diff(sysm.diss, sysm.v[i])
        items.append((None, r))
    return items, dict(form='implicit', variant='gradres')


def build_el_G(sysm, S):
    if 'KE' in S:
        T = sp.Add(*[K(m_, w_) for m_, w_ in sysm.kin_terms])
    else:
        T = sp.Add(*[m_ * w_ / 2 for m_, w_ in sysm.kin_terms])
    return [(None, T - sysm.Vp)], dict(form='el', variant='std')


# ---------------------------------------------------------------- realization pipeline
class Rep:
    __slots__ = ('sid', 'form', 'S', 'level', 'code', 'B', 'meta', 'items', 'consts', 'sign_known',
                 'licensed_refs', 'cost', 'tree_cost', 'templates', 'sympy_items')

    def summary(self):
        return dict(sid=self.sid, form=self.form, S=sorted(self.S), variant=self.meta['variant'],
                    cost=self.cost, nconst=len(self.consts), licensed=sorted(map(str, self.licensed_refs)))


def lump_expr(e, statelike, licensed, table, sid):
    """C4: replace maximal state-free parameter subtrees (containing >=1 unlicensed parameter)
    by one constant; licensed corpus constants and numbers are kept."""
    def has_call(x):
        return x.has(AppliedUndef)

    def free(x):
        return not (x.free_symbols & statelike) and not has_call(x)

    def lumpable(x):
        return free(x) and any(s not in licensed for s in x.free_symbols)

    def const_of(x):
        key = sp.cancel(x)
        if key not in table:
            table[key] = sp.Symbol('_c%d' % len(table))
        return table[key]

    def rec(x):
        if x.is_Number or isinstance(x, sp.Symbol):
            return const_of(x) if lumpable(x) else x
        if lumpable(x):
            return const_of(x)
        if isinstance(x, (sp.Add, sp.Mul)):
            lum = [a for a in x.args if lumpable(a)]
            nums = [a for a in x.args if a.is_Number]
            if lum:
                grp = x.func(*(lum + nums))
                c = const_of(grp)
                rest = [a for a in x.args if a not in lum and a not in nums]
                return x.func(c, *[rec(a) for a in rest])
            return x.func(*[rec(a) for a in x.args])
        if not x.args:
            return x
        return x.func(*[rec(a) for a in x.args])
    return rec(e)


def topo_temps(temps):
    names = [n for n, _ in temps]
    nameset = set(names)
    deps = {n: (e.free_symbols & nameset) - {n} for n, e in temps}
    done, out = set(), []
    pending = list(temps)
    while pending:
        progressed = False
        for idx, (n, e) in enumerate(pending):
            if deps[n] <= done:
                out.append((n, e))
                done.add(n)
                pending.pop(idx)
                progressed = True
                break
        if not progressed:
            raise RuntimeError('cyclic temps')
    return out


class SysCtx:
    """Scope sizes for a (system, formulation) in a code."""
    def __init__(self, sysm, form, S, level, code):
        self.n_state = len(sysm.state) + (sysm.n if (sysm.fam == 'G' and form == 'implicit') else 0)
        self.n_param = len(sysm.params)
        self.nlib = len(lib_names(S))
        self.nbuiltin = len(builtins_for(level))
        self.code = code


def items_cost(items, code, ctx, n_param=None):
    """items: list of (label or None, tree id); temps precede outputs."""
    np_ = ctx.n_param if n_param is None else n_param
    ntemps = sum(1 for l, _ in items if l is not None)
    tot = gamma_len(len(items) + 1)
    ti = oi = 0
    tc = 'C1' if code in ('C4', 'C5') else code
    for lab, tid in items:
        if lab is not None:
            sizes = {'state': ctx.n_state, 'param': np_, 'temp': ti, 'out': 0}
            ti += 1
        else:
            sizes = {'state': ctx.n_state, 'param': np_, 'temp': ntemps, 'out': oi}
            oi += 1
        tot += TR.tree_cost(tid, tc, sizes, ctx.nlib, ctx.nbuiltin)
        if code == 'C3' and lab is not None:
            tot += 3 * LOG95
    return tot


def greedy_inline(items, costfn):
    base = costfn(items)
    changed = True
    while changed:
        changed = False
        for idx in range(len(items) - 1, -1, -1):
            lab, tid = items[idx]
            if lab is None:
                continue
            lid = TR.leaf(lab)
            mapping = {lid: tid}
            memo = {}
            trial = [(l2, TR.subst(e2, mapping, memo)) for j, (l2, e2) in enumerate(items) if j != idx]
            c = costfn(trial)
            if c < base - 1e-12:
                items, base, changed = trial, c, True
                break
    return items, base


def realize(sysm, sympy_items, meta, S, level, code, B):
    """Run the pipeline on one sympy representation; return Rep."""
    sid = sysm.sid
    form = meta['form']
    ctx = SysCtx(sysm, form, S, level, code)
    statelike = set(sysm.state) | (set(sysm.z) if sysm.fam == 'G' else set())
    statelike |= {n for n, _ in sympy_items if n is not None}
    statelike |= {s for _, e in sympy_items for s in e.free_symbols if s.name.startswith('_O')}
    consts, licensed = {}, set()
    items = list(sympy_items)
    if code == 'C4':
        lic = licensed_symbols(sysm, S)
        table = {}
        items = [(n, lump_expr(e, statelike, lic, table, sid)) for n, e in items]
        consts = {v: k for k, v in table.items()}       # const symbol -> value expr
    if code != 'C5':
        exprs = [e for _, e in items]
        repl, red = sp.cse(exprs, symbols=sp.numbered_symbols('_x'), optimizations='basic')
        temps = list(repl) + [(n, r) for (n, _), r in zip(items, red) if n is not None]
        outs = [r for (n, _), r in zip(items, red) if n is None]
    else:
        temps = [(n, e) for n, e in items if n is not None]
        outs = [e for n, e in items if n is None]
    temps = topo_temps(temps)
    titems = [(TR.local_label_of(n, sid), TR.from_sympy(e, sid)) for n, e in temps]
    titems += [(None, TR.from_sympy(e, sid)) for e in outs]

    def const_labels(its):
        labs = set()
        for _, tid in its:
            TR.leaves(tid, labs)
        return labs

    if code == 'C4':
        labs0 = const_labels(titems)
        nconst = sum(1 for l in labs0 if l[0] == 'c')
        lic0 = licensed_symbols(sysm, S)
        nlic = sum(1 for s in sysm.params if s in lic0)
        costfn = lambda its: items_cost(its, code, ctx, n_param=nconst + nlic)
    else:
        costfn = lambda its: items_cost(its, code, ctx)
    if code != 'C5':
        titems, _ = greedy_inline(titems, costfn)
    rep = Rep()
    rep.sid, rep.form, rep.S, rep.level, rep.code, rep.B, rep.meta = sid, form, S, level, code, B, meta
    rep.items = titems
    rep.sympy_items = sympy_items
    labs = const_labels(titems)
    if code == 'C4':
        present = {l for l in labs if l[0] == 'c'}
        rep.consts = {('c', sid, int(s.name[2:])): v for s, v in consts.items()
                      if ('c', sid, int(s.name[2:])) in present}
        lic = licensed_symbols(sysm, S)
        rep.licensed_refs = {l[1] for l in labs if l[0] == 's' and l[1] in lic}
        if form == 'newton' and meta.get('n2'):
            rep.licensed_refs |= set(sysm.masses)        # the N2 decoder reads m_i
        if meta.get('pc'):
            rep.licensed_refs |= {m for m in sysm.masses if m in lic}
        nlic_scope = sum(1 for s in sysm.params if s in lic)
        rep.tree_cost = items_cost(titems, code, ctx, n_param=len(rep.consts) + nlic_scope)
        # (PC decoder constants added below are not tree leaves and do not enter the scope)
        if meta.get('pc') and not all(m in lic for m in sysm.masses):
            # the PC decoder a_N = -sum m_i a_i / m_N needs the mass ratios as numeric constants
            for j, m in enumerate(sysm.masses[:-1]):
                rep.consts[('c', sid, 1000 + j)] = m / sysm.masses[-1]
        rep.cost = rep.tree_cost + B * len(rep.consts)
        rep.sign_known = {l for l, v in rep.consts.items() if sign_known(v)}
    else:
        rep.consts, rep.licensed_refs, rep.sign_known = {}, set(), set()
        rep.tree_cost = items_cost(titems, code, ctx)
        rep.cost = rep.tree_cost
    tpl = set()
    for _, tid in titems:
        TR.calls_in(tid, tpl)
    rep.templates = tpl
    return rep


POS_SYMS = None


def sign_known(v):
    """POS (m>0, k>0): is the sign of the lumped constant determined?"""
    global POS_SYMS
    if POS_SYMS is None:
        POS_SYMS = {}
        for s in CP.corpus():
            cand = list(s.params)
            for p in cand:
                n = p.name
                if n in ('m', 'M') or (n[0] == 'm' and (n[1:].isdigit() or n[1:] in ('A', 'B', 'C'))) or \
                        n == 'M0' or (n[0] == 'k' and n != 'ke'):
                    POS_SYMS[p] = sp.Symbol(p.name + '_pos', positive=True)
    e = v.xreplace(POS_SYMS)
    if e.free_symbols - set(POS_SYMS.values()):
        return False
    return bool(e.is_positive) or bool(e.is_negative)


def make_cost_term(sysm, form, S, level, code, B):
    """Standalone cost of one term (used for the per-term template-vs-explicit choice)."""
    ctx = SysCtx(sysm, form, S, level, code)
    statelike = set(sysm.state)
    lic = licensed_symbols(sysm, S) if code == 'C4' else set()
    cache = {}

    def cost_term(e):
        if e in cache:
            return cache[e]
        if code == 'C4':
            table = {}
            e2 = lump_expr(e, statelike, lic, table, sysm.sid)
            tid = TR.from_sympy(e2, sysm.sid)
            nl = sum(1 for s in sysm.params if s in lic)
            c = items_cost([(None, tid)], code, ctx, n_param=len(table) + nl) + B * len(table)
        else:
            tid = TR.from_sympy(e, sysm.sid)
            c = items_cost([(None, tid)], code, ctx)
        cache[e] = c
        return c
    return cost_term


def build_all(sysm, form, S, level, code, B, truth):
    """All alternative representations for (system, formulation, subset); returns list of Rep."""
    ct = make_cost_term(sysm, form, S, level, code, B)
    out = []
    if sysm.fam == 'P':
        if form == 'newton':
            out.append(realize(sysm, *build_newton(sysm, S, ct), S, level, code, B))
            if sysm.isolated and 'PC' not in S:
                out.append(realize(sysm, *build_newton(sysm, S, ct, outref=True), S, level, code, B))
            if level == 'D2':
                g = build_gradient_P(sysm, S, ct)
                if g is not None:
                    out.append(realize(sysm, *g, S, level, code, B))
        else:
            out.append(realize(sysm, *build_el_P(sysm, S, ct), S, level, code, B))
    else:
        if form == 'explicit':
            if truth[sysm.sid].get('acc') is None:     # scaling series SNL4 only (see DEVIATIONS.txt)
                return out
            out.append(realize(sysm, *build_explicit_G(sysm, truth), S, level, code, B))
        elif form == 'implicit':
            out.append(realize(sysm, *build_implicit_G(sysm, truth), S, level, code, B))
            if level == 'D2':
                out.append(realize(sysm, *build_gradres_G(sysm), S, level, code, B))
        else:
            out.append(realize(sysm, *build_el_G(sysm, S), S, level, code, B))
    return out

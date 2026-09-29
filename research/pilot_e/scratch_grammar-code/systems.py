"""Corpus of textbook mechanical systems + law-dependent representations."""
import sympy as sp
from functools import lru_cache

t = sp.Symbol('t')
G, g, ke = sp.symbols('G g ke')
V = sp.Function('V')      # vector constructor (base grammar)
Cn = sp.Function('Cn')    # central-force constructor s*(dx,dy) (base grammar, used only under LC)
D = sp.Function('D')      # DIST template
K = sp.Function('K')      # KE template  K(m,w) = m*w/2
Fh1, Fh2, Vh1, Vh2 = [sp.Function(n) for n in ('Fh1', 'Fh2', 'Vh1', 'Vh2')]
Fg, Vg = sp.Function('Fg'), sp.Function('Vg')

LAWS = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG', 'DIST', 'KE']
CONTROLS = {'DIST', 'KE'}

X = lambda i: sp.Symbol('x%d' % i)
Y = lambda i: sp.Symbol('y%d' % i)
U = lambda i: sp.Symbol('u%d' % i)   # vx
W = lambda i: sp.Symbol('w%d' % i)   # vy
M = lambda i: sp.Symbol('m%d' % i)
Rph = sp.Symbol('Rph')               # distance placeholder

PAIR = {'spring1', 'spring2', 'grav', 'coul'}
CONS = {'spring1', 'spring2', 'grav', 'fixgrav', 'coul', 'unif'}


class PSys:
    family = 'P'

    def __init__(self, name, d, n, inter):
        self.name, self.d, self.n, self.inter = name, d, n, inter
        self.bodies = list(range(1, n + 1))

    @property
    def state(self):
        s = []
        for i in self.bodies:
            s += [X(i), U(i)] + ([Y(i), W(i)] if self.d == 2 else [])
        return s + ([t] if any(it[0] == 'drive' for it in self.inter) else [])

    @property
    def params(self):
        p = set(M(i) for i in self.bodies)
        for it in self.inter:
            for a in it[1:]:
                if isinstance(a, sp.Symbol):
                    p.add(a)
            if it[0] in ('grav', 'fixgrav'):
                p.add(G)
            if it[0] == 'unif':
                p.add(g)
            if it[0] == 'coul':
                p.add(ke)
        return sorted(p, key=str)

    conservative = property(lambda s: all(it[0] in CONS for it in s.inter))
    isolated = property(lambda s: s.n >= 2 and all(it[0] in PAIR for it in s.inter))
    has_pair = property(lambda s: any(it[0] in PAIR for it in s.inter))
    has_R = property(lambda s: s.d == 2 and any(it[0] in ('spring2', 'grav', 'fixgrav', 'coul') for it in s.inter))

    def relevant(self, S):
        r = set()
        if 'EC' in S and self.conservative:
            r |= {'EC', 'KE'}
            if any(it[0].startswith('spring') for it in self.inter): r.add('HK')
            if any(it[0] in ('grav', 'fixgrav') for it in self.inter): r.add('UG')
            if self.has_R: r.add('DIST')
            return frozenset(r & S)
        r.add('N2')
        if self.has_pair: r.add('N3')
        if self.isolated: r.add('PC')
        if self.d == 2 and any(it[0] in ('spring2', 'grav', 'fixgrav', 'coul') for it in self.inter): r.add('LC')
        if any(it[0].startswith('spring') for it in self.inter): r.add('HK')
        if any(it[0] in ('grav', 'fixgrav') for it in self.inter): r.add('UG')
        if self.has_R: r.add('DIST')
        return frozenset(r & S)


def _R(dx, dy, S):
    return D(dx, dy) if 'DIST' in S else sp.sqrt(dx ** 2 + dy ** 2)


def pos(i, d):
    if i is None:
        return [sp.Integer(0)] * d
    return [X(i), Y(i)][:d]


def explicit_force(sys, it, i, S):
    """Explicit force vector on body i from interaction it, plus central scalar (or None)."""
    d = sys.d
    kind = it[0]
    if kind == 'spring1':
        _, a, b, k = it
        xb = X(b) if b is not None else 0
        return [(-k if i == a else k) * (X(a) - xb)], None
    if kind in ('spring2', 'grav', 'coul', 'fixgrav'):
        if kind == 'fixgrav':
            _, a, Mc = it
            j = None
        else:
            a, b = it[1], it[2]
            j = b if i == a else a
        # canonical orientation: difference always (first body) - (second body);
        # the second body gets the sign flip.  Keeps the no-N3 baseline DAG-shareable (fairness).
        first = it[1]
        sgn = 1 if i == first else -1
        pf, po = pos(first, d), pos(j if i == first else i, d)
        dx, dy = pf[0] - po[0], pf[1] - po[1]
        if kind == 'spring2':
            k, l = it[3], it[4]
            s = -k * (Rph - l) / Rph
        elif kind == 'grav':
            s = -G * M(i) * M(j) / Rph ** 3
        elif kind == 'fixgrav':
            s = -G * Mc * M(i) / Rph ** 3
        else:
            s = ke * it[3] * it[4] / Rph ** 3
        s = sgn * s
        R = _R(dx, dy, S)
        return [(s * dx).xreplace({Rph: R}), (s * dy).xreplace({Rph: R})], (s.xreplace({Rph: R}), dx, dy)
    if kind == 'unif':
        return [sp.Integer(0), -M(i) * g], None
    if kind == 'damp':
        return [-it[2] * U(i)], None
    if kind == 'drive':
        return [it[2] * sp.cos(it[3] * t)], None
    if kind == 'mag':
        _, a, qc, Bf = it
        return [qc * Bf * W(i), -qc * Bf * U(i)], None
    raise ValueError(kind)


def template_force(sys, it, i, S):
    kind = it[0]
    d = sys.d
    if kind == 'spring1' and 'HK' in S:
        _, a, b, k = it
        j = b if i == a else a
        return Fh1(k, X(i), X(j) if j is not None else sp.Integer(0))
    if kind == 'spring2' and 'HK' in S:
        a, b, k, l = it[1:]
        j = b if i == a else a
        return Fh2(k, l, *pos(i, d), *pos(j, d))
    if kind == 'grav' and 'UG' in S:
        a, b = it[1], it[2]
        j = b if i == a else a
        return Fg(G, M(i), M(j), *pos(i, d), *pos(j, d))
    if kind == 'fixgrav' and 'UG' in S:
        return Fg(G, M(i), it[2], *pos(i, d), *pos(None, d))
    return None


def bodies_of(it):
    if it[0] in PAIR:
        return [b for b in it[1:3] if b is not None]
    return [it[1]]


def as_vec(comps, d):
    return comps[0] if d == 1 else V(*comps)


def build_P(sys, S, cost_term):
    """Return list of (name|None, expr) for a Cartesian particle system under law set S.
    cost_term(expr) -> bits, used for the coder's local choice template-vs-explicit."""
    d = sys.d
    if 'EC' in S and sys.conservative:
        pot = []
        for it in sys.inter:
            kind = it[0]
            if kind == 'spring1':
                _, a, b, k = it
                xb = X(b) if b is not None else sp.Integer(0)
                e = Vh1(k, X(a), xb) if 'HK' in S else k * (X(a) - xb) ** 2 / 2
            elif kind in ('spring2', 'grav', 'coul', 'fixgrav'):
                if kind == 'fixgrav':
                    a, j = it[1], None
                else:
                    a, j = it[1], it[2]
                pa, pj = pos(a, d), pos(j, d)
                R = _R(pa[0] - pj[0], pa[1] - pj[1], S)
                if kind == 'spring2':
                    k, l = it[3], it[4]
                    e = Vh2(k, l, *pa, *pj) if 'HK' in S else k * (R - l) ** 2 / 2
                elif kind == 'grav':
                    e = Vg(G, M(a), M(j), *pa, *pj) if 'UG' in S else -G * M(a) * M(j) / R
                elif kind == 'fixgrav':
                    e = Vg(G, M(a), it[2], *pa, *pj) if 'UG' in S else -G * it[2] * M(a) / R
                else:
                    e = ke * it[3] * it[4] / R
            elif kind == 'unif':
                e = M(it[1]) * g * Y(it[1])
            pot.append(e)
        kin = []
        for i in sys.bodies:
            w2 = U(i) ** 2 + (W(i) ** 2 if d == 2 else 0)
            kin.append(K(M(i), w2) if 'KE' in S else M(i) * w2 / 2)
        Lg = sp.Add(*kin) - sp.Add(*pot)
        return [(None, Lg)]

    items = []
    N2 = 'N2' in S
    terms = {i: [] for i in sys.bodies}
    pcount = 0
    for it in sys.inter:
        bl = bodies_of(it)
        first = bl[0]
        Pref = None
        for i in bl:
            if 'N3' in S and it[0] in PAIR and i != first and Pref is not None:
                # force on second body = -(force on first); without N2 need mass ratio
                term = -Pref if N2 else -M(first) * Pref / M(i)
                terms[i].append(term)
                continue
            comps, cen = explicit_force(sys, it, i, S)
            if 'LC' in S and cen is not None:
                s, dx, dy = cen
                expl = Cn(s if N2 else s / M(i), dx, dy)
            else:
                expl = as_vec([c if N2 else c / M(i) for c in comps], d)
            tmpl = template_force(sys, it, i, S)
            if tmpl is not None and not N2:
                tmpl = tmpl / M(i)
            term = expl
            if tmpl is not None and cost_term(tmpl) < cost_term(expl):
                term = tmpl
            if 'N3' in S and it[0] in PAIR and len(bl) == 2:
                pcount += 1
                Pref = sp.Symbol('P%d' % pcount)
                items.append((Pref, term))
                term = Pref
            terms[i].append(term)
    outs = list(sys.bodies)
    if 'PC' in S and sys.isolated:
        outs = outs[:-1]
    for i in outs:
        items.append((None, sp.Add(*terms[i]) if terms[i] else sp.Integer(0)))
    return items


# ---------------------------------------------------------------- family G (generalized coordinates)
class GSys:
    family = 'P'  # placeholder, overwritten
    def __init__(self, name, q, qd, params, T, Vp, kin_terms):
        self.family = 'G'
        self.name, self.q, self.qd, self.T, self.Vp = name, q, qd, T, Vp
        self.kin_terms = kin_terms  # list of (mass, squared speed w)
        self._params = params
        self._eom = None

    state = property(lambda s: list(s.q) + list(s.qd))
    params = property(lambda s: s._params)
    conservative = True
    isolated = False
    has_pair = False

    def relevant(self, S):
        return frozenset({'EC', 'KE'} & S) if 'EC' in S else frozenset()

    def eom(self):
        if self._eom is None:
            qdd = [sp.Symbol('a_' + str(qq)) for qq in self.q]
            Lg = self.T - self.Vp
            eqs = []
            for qq, v, a in zip(self.q, self.qd, qdd):
                dLdv = sp.diff(Lg, v)
                # total time derivative
                dt = sum(sp.diff(dLdv, q2) * v2 for q2, v2 in zip(self.q, self.qd)) + \
                     sum(sp.diff(dLdv, v2) * a2 for v2, a2 in zip(self.qd, qdd))
                eqs.append(sp.expand(dt - sp.diff(Lg, qq)))
            sol = sp.solve(eqs, qdd, dict=True)[0]
            self._eom = [sp.simplify(sp.trigsimp(sol[a])) for a in qdd]
        return self._eom


def build_G(sys, S, cost_term):
    if 'EC' in S:
        if 'KE' in S:
            T = sp.Add(*[K(m, w) for m, w in sys.kin_terms])
        else:
            T = sys.T
        return [(None, T - sys.Vp)]
    return [(None, e) for e in sys.eom()]


def build(sys, S, cost_term):
    return build_G(sys, S, cost_term) if sys.family == 'G' else build_P(sys, S, cost_term)


# ---------------------------------------------------------------- corpus
def corpus(include_G=True):
    k, k1, k2, k3, l, l1, l2, l3, b, F0, w, qc, Bf, Mc, q1, q2 = sp.symbols('k k1 k2 k3 l l1 l2 l3 b F0 w qc Bf M q1 q2')
    P = [
        PSys('free1D', 1, 1, []),
        PSys('SHO', 1, 1, [('spring1', 1, None, k)]),
        PSys('2mass1D', 1, 2, [('spring1', 1, 2, k)]),
        PSys('3chain1D', 1, 3, [('spring1', 1, 2, k1), ('spring1', 2, 3, k2)]),
        PSys('damped', 1, 1, [('spring1', 1, None, k), ('damp', 1, b)]),
        PSys('driven', 1, 1, [('spring1', 1, None, k), ('drive', 1, F0, w)]),
        PSys('projectile', 2, 1, [('unif', 1)]),
        PSys('kepler', 2, 1, [('fixgrav', 1, Mc)]),
        PSys('2body', 2, 2, [('grav', 1, 2)]),
        PSys('3body', 2, 3, [('grav', 1, 2), ('grav', 1, 3), ('grav', 2, 3)]),
        PSys('springpend2D', 2, 1, [('spring2', 1, None, k, l), ('unif', 1)]),
        PSys('2mass2D', 2, 2, [('spring2', 1, 2, k, l)]),
        PSys('triangle', 2, 3, [('spring2', 1, 2, k1, l1), ('spring2', 2, 3, k2, l2), ('spring2', 1, 3, k3, l3)]),
        PSys('coulomb2', 2, 2, [('coul', 1, 2, q1, q2)]),
        PSys('cyclotron', 2, 1, [('mag', 1, qc, Bf)]),
    ]
    out = list(P)
    if include_G:
        out += gsystems()
    return out


def gsystems():
    th, th1, th2, ph, x = sp.symbols('th th1 th2 ph x')
    thd, thd1, thd2, phd, xd = sp.symbols('thd thd1 thd2 phd xd')
    m, m1, m2, l, l1, l2, R, Om = sp.symbols('m m1 m2 l l1 l2 R Om')
    gs = []
    w = l ** 2 * thd ** 2
    gs.append(GSys('pendulum', [th], [thd], [m, l, g], m * w / 2, -m * g * l * sp.cos(th), [(m, w)]))
    w1 = l1 ** 2 * thd1 ** 2
    w2 = l1 ** 2 * thd1 ** 2 + l2 ** 2 * thd2 ** 2 + 2 * l1 * l2 * thd1 * thd2 * sp.cos(th1 - th2)
    gs.append(GSys('dblpend', [th1, th2], [thd1, thd2], [m1, m2, l1, l2, g],
                   m1 * w1 / 2 + m2 * w2 / 2,
                   -(m1 + m2) * g * l1 * sp.cos(th1) - m2 * g * l2 * sp.cos(th2), [(m1, w1), (m2, w2)]))
    gs.append(GSys('atwood', [x], [xd], [m1, m2, g], (m1 + m2) * xd ** 2 / 2, -(m1 - m2) * g * x,
                   [(m1, xd ** 2), (m2, xd ** 2)]))
    wh = R ** 2 * (thd ** 2 + Om ** 2 * sp.sin(th) ** 2)
    gs.append(GSys('hoop', [th], [thd], [m, R, Om, g], m * wh / 2, -m * g * R * sp.cos(th), [(m, wh)]))
    ws = l ** 2 * (thd ** 2 + sp.sin(th) ** 2 * phd ** 2)
    gs.append(GSys('sphpend', [th, ph], [thd, phd], [m, l, g], m * ws / 2, -m * g * l * sp.cos(th), [(m, ws)]))
    return gs


# ---------------------------------------------------------------- law statements (templates / schemata)
def law_items(f, S, used_templates):
    a, b, m, F, s, q, Lg, T, Vv, wv = sp.symbols('a b m F s q L T Vv wv')
    xi, yi, xj, yj, k, l = sp.symbols('xi yi xj yj k l')
    mi, mj = sp.symbols('mi mj')
    Rdef = D(xi - xj, yi - yj) if 'DIST' in S else sp.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
    Sm, Cr, Dt, Dv, Dq = [sp.Function(n) for n in ('Sm', 'Cr', 'Dt', 'Dv', 'Dq')]
    if f == 'N2':
        return [(None, F / m)], [F, m]
    if f == 'N3':
        return [(None, -F)], [F]
    if f == 'PC':
        return [(None, Sm(m * a))], [m, a]
    if f == 'LC':
        return [(None, Sm(Cr(q, F)))], [q, F]
    if f == 'EC':
        return [(None, Dt(Dv(Lg)) - Dq(Lg)), (None, T - Vv)], [Lg, T, Vv]
    if f == 'DIST':
        return [(None, sp.sqrt(a ** 2 + b ** 2))], [a, b]
    if f == 'KE':
        return [(None, m * wv / 2)], [m, wv]
    defs = []
    if f == 'HK':
        cand = {'Fh1': -k * (a - b), 'Vh1': k * (a - b) ** 2 / 2,
                'Fh2': V(-k * (Rdef - l) * (xi - xj) / Rdef, -k * (Rdef - l) * (yi - yj) / Rdef),
                'Vh2': k * (Rdef - l) ** 2 / 2}
    elif f == 'UG':
        cand = {'Fg': V(-G * mi * mj * (xi - xj) / Rdef ** 3, -G * mi * mj * (yi - yj) / Rdef ** 3),
                'Vg': -G * mi * mj / Rdef}
    else:
        raise ValueError(f)
    used = [n for n in cand if n in used_templates] or [list(cand)[0]]
    for n in used:
        defs.append((None, cand[n]))
    return defs, [a, b, k, l, xi, yi, xj, yj, mi, mj, G]

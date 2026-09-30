"""Corpus of 24 textbook systems (spec section 2) + ground truth (truth.pkl).

Symbol conventions
  Family P (Cartesian): body i of a system has position x_i (, y_i) and velocity u_i (, w_i).
    Parameters are plain sympy Symbols; identical names denote the same physical entity
    across systems (spec section 2: mA, mB, kA, kB, kC, and literally m1..m5 shared by P06/P11).
  Family G (generalized coords): q_i, v_i (velocities), z_i (accelerations, only in implicit forms).
    All family-G symbols except g carry real=True, which makes them distinct sympy objects from
    family-P symbols of the same printed name (so a pendulum bob 'm1' is not body m1 of P06).
    g is shared by P07, P13 and family G (spec: g paid once per corpus).
"""
import os, pickle, itertools
import sympy as sp
from common import HERE, LAW9

t = sp.Symbol('t')
G, g, ke = sp.symbols('G g ke')

X = lambda i: sp.Symbol('x%d' % i)
Y = lambda i: sp.Symbol('y%d' % i)
U = lambda i: sp.Symbol('u%d' % i)
W = lambda i: sp.Symbol('w%d' % i)

mA, mB, mC = sp.symbols('mA mB mC')
m1, m2, m3, m4, m5 = sp.symbols('m1:6')
kA, kB, kC, k, k1, k2, k3 = sp.symbols('kA kB kC k k1 k2 k3')
lP = sp.Symbol('l')
bP, F0, wdr, M0, qch, Bz = sp.symbols('b F0 w M0 q Bz')
qA, qB, qC = sp.symbols('qA qB qC')

PAIR = {'spring1', 'spring2', 'grav', 'coul'}
CONS = {'spring1', 'spring2', 'grav', 'fixgrav', 'coul', 'unif'}
CENTRAL = {'spring2', 'grav', 'fixgrav', 'coul'}

STATE_NAMES_P = ('x', 'y', 'u', 'w')


def is_pair(it):
    """two-body interaction (a wall / pivot / fixed centre is external, not a pair)."""
    return it[0] in PAIR and it[2] is not None


class PSys:
    fam = 'P'

    def __init__(self, sid, name, d, masses, inter, applicable, trackR):
        self.sid, self.name, self.d = sid, name, d
        self.masses = masses            # list of mass symbols, body i -> masses[i-1]
        self.n = len(masses)
        self.bodies = list(range(1, self.n + 1))
        self.inter = inter
        self.applicable = frozenset(applicable)
        self.trackR = trackR

    def M(self, i):
        return self.masses[i - 1]

    @property
    def dof(self):
        return self.n * self.d

    @property
    def coords(self):
        c = []
        for i in self.bodies:
            c += [X(i)] + ([Y(i)] if self.d == 2 else [])
        return c

    @property
    def vels(self):
        c = []
        for i in self.bodies:
            c += [U(i)] + ([W(i)] if self.d == 2 else [])
        return c

    @property
    def driven(self):
        return any(it[0] == 'drive' for it in self.inter)

    @property
    def state(self):
        s = []
        for i in self.bodies:
            s += [X(i), U(i)] + ([Y(i), W(i)] if self.d == 2 else [])
        return s + ([t] if self.driven else [])

    @property
    def params(self):
        p = list(self.masses)
        for it in self.inter:
            kind = it[0]
            if kind == 'spring1':
                p.append(it[3])
            elif kind == 'spring2':
                p += [it[3], it[4]]
            elif kind == 'grav':
                p.append(G)
            elif kind == 'fixgrav':
                p += [it[2], G]
            elif kind == 'coul':
                p += [it[3], it[4], ke]
            elif kind == 'unif':
                p.append(g)
            elif kind == 'damp':
                p.append(it[2])
            elif kind == 'drive':
                p += [it[2], it[3]]
            elif kind == 'mag':
                p += [it[2], it[3]]
        out = []
        for s in p:
            if s not in out:
                out.append(s)
        return sorted(out, key=str)

    conservative = property(lambda s: all(it[0] in CONS for it in s.inter))
    isolated = property(lambda s: s.n >= 2 and len(s.inter) > 0 and all(is_pair(it) for it in s.inter))

    # ---------------------------------------------------------------- physics
    def pos(self, i):
        if i is None:
            return [sp.Integer(0)] * self.d
        return [X(i), Y(i)][:self.d]

    def force(self, it, i):
        """Physical force vector (list of d comps) on body i from interaction it."""
        d = self.d
        kind = it[0]
        if kind == 'spring1':
            _, a, b, kk = it
            xb = X(b) if b is not None else 0
            f = -kk * (X(a) - xb)
            return [f if i == a else -f]
        if kind in ('spring2', 'grav', 'coul'):
            a, b = it[1], it[2]
            pa, pb = self.pos(a), self.pos(b)
            r = [pa[c] - pb[c] for c in range(d)]
            R = sp.sqrt(sum(rc ** 2 for rc in r))
            if kind == 'spring2':
                s = -it[3] * (R - it[4]) / R
            elif kind == 'grav':
                s = -G * self.M(a) * self.M(b) / R ** 3
            else:
                s = ke * it[3] * it[4] / R ** 3
            f = [s * rc for rc in r]
            return f if i == a else [-fc for fc in f]
        if kind == 'fixgrav':
            a, Mc = it[1], it[2]
            r = self.pos(a)
            R = sp.sqrt(sum(rc ** 2 for rc in r))
            return [-G * Mc * self.M(a) * rc / R ** 3 for rc in r]
        if kind == 'unif':
            return [sp.Integer(0), -self.M(it[1]) * g]
        if kind == 'damp':
            return [-it[2] * U(it[1])]
        if kind == 'drive':
            return [it[2] * sp.cos(it[3] * t)]
        if kind == 'mag':
            _, a, qq, BB = it
            return [qq * BB * W(a), -qq * BB * U(a)]
        raise ValueError(kind)

    def bodies_of(self, it):
        if it[0] in PAIR:
            return [b for b in it[1:3] if b is not None]
        return [it[1]]

    def truth(self):
        """Accelerations in coordinate order (x1,(y1),x2,...)."""
        acc = []
        for i in self.bodies:
            F = [sp.Integer(0)] * self.d
            for it in self.inter:
                if i in self.bodies_of(it):
                    f = self.force(it, i)
                    F = [F[c] + f[c] for c in range(self.d)]
            acc += [sp.expand(Fc / self.M(i)) for Fc in F]
        return acc

    def potential_terms(self):
        """List of (interaction, explicit potential expr) for conservative interactions."""
        out = []
        d = self.d
        for it in self.inter:
            kind = it[0]
            if kind == 'spring1':
                _, a, b, kk = it
                xb = X(b) if b is not None else 0
                out.append((it, kk * (X(a) - xb) ** 2 / 2))
            elif kind in ('spring2', 'grav', 'coul', 'fixgrav'):
                a = it[1]
                bb = None if kind == 'fixgrav' else it[2]
                pa, pb = self.pos(a), self.pos(bb)
                R = sp.sqrt(sum((pa[c] - pb[c]) ** 2 for c in range(d)))
                if kind == 'spring2':
                    out.append((it, it[3] * (R - it[4]) ** 2 / 2))
                elif kind == 'grav':
                    out.append((it, -G * self.M(a) * self.M(bb) / R))
                elif kind == 'fixgrav':
                    out.append((it, -G * it[2] * self.M(a) / R))
                else:
                    out.append((it, ke * it[3] * it[4] / R))
            elif kind == 'unif':
                out.append((it, self.M(it[1]) * g * Y(it[1])))
        return out

    def lagrangian(self):
        T = sum(self.M(i) * (U(i) ** 2 + (W(i) ** 2 if self.d == 2 else 0)) / 2 for i in self.bodies)
        return T - sum(v for _, v in self.potential_terms())

    # counts (spec 7.1)
    def n_pairs(self):
        return sum(1 for it in self.inter if is_pair(it))

    def n_central(self):
        return sum(1 for it in self.inter if it[0] in CENTRAL)

    def n_springs(self):
        return sum(1 for it in self.inter if it[0].startswith('spring'))

    def n_grav(self):
        return sum(1 for it in self.inter if it[0] in ('grav', 'fixgrav'))

    def n_force_terms(self):
        return sum(len(self.bodies_of(it)) for it in self.inter)


class GSys:
    fam = 'G'

    def __init__(self, sid, name, n, params, Tfun, Vfun, kin_fun, applicable, trackR, diss=None):
        self.sid, self.name = sid, name
        self.q = list(sp.symbols('q1:%d' % (n + 1), real=True))
        self.v = list(sp.symbols('v1:%d' % (n + 1), real=True))
        self.z = list(sp.symbols('z1:%d' % (n + 1), real=True))
        self.n = n
        self._params = params
        self.T = Tfun(self.q, self.v)
        self.Vp = Vfun(self.q)
        self.kin_terms = kin_fun(self.q, self.v)      # list of (mass, w) with T = sum m*w/2
        assert sp.simplify(self.T - sum(m_ * w_ / 2 for m_, w_ in self.kin_terms)) == 0, sid
        self.diss = diss                               # Rayleigh dissipation function (G02)
        self.applicable = frozenset(applicable)
        self.trackR = trackR
        self.d = None

    dof = property(lambda s: s.n)
    coords = property(lambda s: s.q)
    vels = property(lambda s: s.v)
    state = property(lambda s: s.q + s.v)
    params = property(lambda s: sorted(s._params, key=str))
    conservative = property(lambda s: s.diss is None)
    isolated = False

    def el_equations(self):
        """EL residuals r_i = d/dt dL/dv_i - dL/dq_i + dR/dv_i  (affine in z)."""
        Lg = self.T - self.Vp
        eqs = []
        for i in range(self.n):
            dLdv = sp.diff(Lg, self.v[i])
            dt = sum(sp.diff(dLdv, self.q[j]) * self.v[j] + sp.diff(dLdv, self.v[j]) * self.z[j]
                     for j in range(self.n))
            r = dt - sp.diff(Lg, self.q[i])
            if self.diss is not None:
                r += sp.diff(self.diss, self.v[i])
            eqs.append(sp.expand(r))
        return eqs

    def implicit(self):
        """M (n x n) and f (n) with M z = f, entries simplified."""
        eqs = self.el_equations()
        Mm = [[sp.simplify(eqs[i].coeff(self.z[j])) for j in range(self.n)] for i in range(self.n)]
        f = [sp.simplify(-(eqs[i] - sum(eqs[i].coeff(self.z[j]) * self.z[j] for j in range(self.n))).expand())
             for i in range(self.n)]
        return Mm, f

    def truth(self):
        Mm, f = self.implicit()
        sol = sp.Matrix(Mm).LUsolve(sp.Matrix(f))
        return [sp.simplify(e) for e in sol]

    def n_pairs(self):
        return 0

    def n_force_terms(self):
        return 0


# ---------------------------------------------------------------- corpus definition
def _gsystems():
    R = lambda *names: sp.symbols(names, real=True)
    out = []
    m, l = sp.symbols('m l', real=True)
    # G01 simple pendulum
    out.append(GSys('G01', 'simple pendulum', 1, [m, l, g],
                    lambda q, v: m * l ** 2 * v[0] ** 2 / 2,
                    lambda q: -m * g * l * sp.cos(q[0]),
                    lambda q, v: [(m, l ** 2 * v[0] ** 2)], {'EC', 'KE'}, True))
    # G02 damped pendulum (Rayleigh dissipation b*v^2/2)
    b = sp.Symbol('b', real=True)
    gd = GSys('G02', 'damped pendulum', 1, [m, l, g, b],
              lambda q, v: m * l ** 2 * v[0] ** 2 / 2,
              lambda q: -m * g * l * sp.cos(q[0]),
              lambda q, v: [(m, l ** 2 * v[0] ** 2)], set(), True)
    gd.diss = b * gd.v[0] ** 2 / 2
    out.append(gd)
    # G03 double pendulum
    m1_, m2_, l1_, l2_ = R('m1', 'm2', 'l1', 'l2')

    def w2(q, v):
        return l1_ ** 2 * v[0] ** 2 + l2_ ** 2 * v[1] ** 2 + 2 * l1_ * l2_ * v[0] * v[1] * sp.cos(q[0] - q[1])
    out.append(GSys('G03', 'double pendulum', 2, [m1_, m2_, l1_, l2_, g],
                    lambda q, v: m1_ * l1_ ** 2 * v[0] ** 2 / 2 + m2_ * w2(q, v) / 2,
                    lambda q: -(m1_ + m2_) * g * l1_ * sp.cos(q[0]) - m2_ * g * l2_ * sp.cos(q[1]),
                    lambda q, v: [(m1_, l1_ ** 2 * v[0] ** 2), (m2_, w2(q, v))], {'EC', 'KE'}, True))
    # G04 triple pendulum
    m3_, l3_ = R('m3', 'l3')
    ms, ls = [m1_, m2_, m3_], [l1_, l2_, l3_]

    def wN(q, v, N):
        e = 0
        for i in range(N):
            e += ls[i] ** 2 * v[i] ** 2
        for i in range(N):
            for j in range(i + 1, N):
                e += 2 * ls[i] * ls[j] * v[i] * v[j] * sp.cos(q[i] - q[j])
        return e

    def V3(q):
        yy, V_ = 0, 0
        for i in range(3):
            yy = yy - ls[i] * sp.cos(q[i])
            V_ += ms[i] * g * yy
        return V_
    out.append(GSys('G04', 'triple pendulum', 3, [m1_, m2_, m3_, l1_, l2_, l3_, g],
                    lambda q, v: sum(ms[i] * wN(q, v, i + 1) / 2 for i in range(3)),
                    V3,
                    lambda q, v: [(ms[i], wN(q, v, i + 1)) for i in range(3)], {'EC', 'KE'}, True))
    # G05 Atwood (x = descent of m1)
    out.append(GSys('G05', 'Atwood machine', 1, [m1_, m2_, g],
                    lambda q, v: (m1_ + m2_) * v[0] ** 2 / 2,
                    lambda q: -m1_ * g * q[0] + m2_ * g * q[0],
                    lambda q, v: [(m1_, v[0] ** 2), (m2_, v[0] ** 2)], {'EC', 'KE'}, True))
    # G06 bead on rotating hoop
    Rh, Om = R('R', 'Om')
    out.append(GSys('G06', 'bead on rotating hoop', 1, [m, Rh, Om, g],
                    lambda q, v: m * Rh ** 2 * (v[0] ** 2 + Om ** 2 * sp.sin(q[0]) ** 2) / 2,
                    lambda q: -m * g * Rh * sp.cos(q[0]),
                    lambda q, v: [(m, Rh ** 2 * (v[0] ** 2 + Om ** 2 * sp.sin(q[0]) ** 2))], {'EC', 'KE'}, True))
    # G07 cart-pendulum (q1 = cart x, q2 = angle)
    Mc = sp.Symbol('M', real=True)

    def wb(q, v):
        return v[0] ** 2 + 2 * l * v[0] * v[1] * sp.cos(q[1]) + l ** 2 * v[1] ** 2
    out.append(GSys('G07', 'cart-pendulum', 2, [Mc, m, l, g],
                    lambda q, v: Mc * v[0] ** 2 / 2 + m * wb(q, v) / 2,
                    lambda q: -m * g * l * sp.cos(q[1]),
                    lambda q, v: [(Mc, v[0] ** 2), (m, wb(q, v))], {'EC', 'KE'}, True))
    return out


def _psystems():
    P = []
    A = lambda *s: set(s)
    P.append(PSys('P01', 'free particle', 1, [mA], [], A('N2', 'EC', 'KE'), False))
    P.append(PSys('P02', 'SHO to wall', 1, [mA], [('spring1', 1, None, kA)], A('N2', 'EC', 'HK', 'KE'), True))
    P.append(PSys('P03', 'damped SHO', 1, [mA], [('spring1', 1, None, kA), ('damp', 1, bP)], A('N2', 'HK'), True))
    P.append(PSys('P04', 'driven damped SHO', 1, [mA],
                  [('spring1', 1, None, kA), ('damp', 1, bP), ('drive', 1, F0, wdr)], A('N2', 'HK'), True))
    P.append(PSys('P05', 'two masses three springs walls', 1, [mA, mB],
                  [('spring1', 1, None, kA), ('spring1', 1, 2, kC), ('spring1', 2, None, kB)],
                  A('N2', 'N3', 'EC', 'HK', 'KE'), True))
    P.append(PSys('P06', 'free chain of 4', 1, [m1, m2, m3, m4],
                  [('spring1', 1, 2, k), ('spring1', 2, 3, k), ('spring1', 3, 4, k)],
                  A('N2', 'N3', 'PC', 'EC', 'HK', 'KE'), True))
    P.append(PSys('P07', 'projectile', 2, [mA], [('unif', 1)], A('N2', 'EC', 'KE'), False))
    P.append(PSys('P08', 'Kepler fixed centre', 2, [mA], [('fixgrav', 1, M0)],
                  A('N2', 'LC', 'EC', 'UG', 'DIST', 'KE'), False))
    nb = A('N2', 'N3', 'PC', 'LC', 'EC', 'UG', 'DIST', 'KE')
    P.append(PSys('P09', 'two-body gravity', 2, [mA, mB], [('grav', 1, 2)], nb, False))
    P.append(PSys('P10', 'three-body gravity', 2, [mA, mB, mC],
                  [('grav', i, j) for i, j in itertools.combinations(range(1, 4), 2)], nb, False))
    P.append(PSys('P11', 'five-body gravity', 2, [m1, m2, m3, m4, m5],
                  [('grav', i, j) for i, j in itertools.combinations(range(1, 6), 2)], nb, False))
    P.append(PSys('P12', 'spring triangle', 2, [mA, mB, mC],
                  [('spring2', 1, 2, k1, lP), ('spring2', 2, 3, k2, lP), ('spring2', 1, 3, k3, lP)],
                  A('N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'DIST', 'KE'), True))
    P.append(PSys('P13', 'elastic pendulum', 2, [mA], [('spring2', 1, None, kA, lP), ('unif', 1)],
                  A('N2', 'LC', 'EC', 'HK', 'DIST', 'KE'), True))
    P.append(PSys('P14', 'cyclotron', 2, [mA], [('mag', 1, qch, Bz)], A('N2'), True))
    P.append(PSys('P15', 'three charges', 2, [mA, mB, mC],
                  [('coul', 1, 2, qA, qB), ('coul', 1, 3, qA, qC), ('coul', 2, 3, qB, qC)],
                  A('N2', 'N3', 'PC', 'LC', 'EC', 'DIST', 'KE'), False))
    P.append(PSys('P16', 'Mach block I', 1, [mB], [('spring1', 1, None, kA)], A('N2', 'EC', 'HK', 'KE'), True))
    P.append(PSys('P17', 'Mach block II', 1, [mA], [('spring1', 1, None, kC)], A('N2', 'EC', 'HK', 'KE'), True))
    return P


_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _psystems() + _gsystems()
        for s in _CORPUS:
            s.truth_expr = None
    return _CORPUS


def by_id():
    return {s.sid: s for s in corpus()}


# ---------------------------------------------------------------- global symbol roles
def is_state_symbol(sym):
    n = sym.name
    if n == 't':
        return True
    if sym.is_real and n[0] in 'qvz' and n[1:].isdigit():
        return True
    return n[0] in 'xyuw' and n[1:].isdigit() and not sym.is_real


TRUTH_PKL = os.path.join(HERE, 'truth.pkl')


def build_truth(force=False):
    """Ground truth EOM for every system; G via EL + sp.simplify.  Also family-G implicit (M,f).
    For family P the hand-coded Newtonian truth is cross-checked against Euler-Lagrange for
    conservative systems (assertion)."""
    if os.path.exists(TRUTH_PKL) and not force:
        with open(TRUTH_PKL, 'rb') as fh:
            return pickle.load(fh)
    out = {}
    for s in corpus():
        if s.fam == 'P':
            acc = s.truth()
            if s.conservative:
                Lg = s.lagrangian()
                qs, vs = s.coords, s.vels
                import random
                rnd = random.Random(1)
                syms = sorted((Lg.free_symbols | set().union(*[a.free_symbols for a in acc])), key=str)
                for _ in range(3):
                    pt = {sy: rnd.uniform(0.5, 2.0) for sy in syms}
                    for ci, (qq, vv) in enumerate(zip(qs, vs)):
                        # Cartesian: m a = dL/dq  (numeric cross-check of hand-coded Newtonian truth)
                        mi = sp.diff(Lg, vv, 2)
                        el = float((sp.diff(Lg, qq) / mi).subs(pt))
                        tv = float(acc[ci].subs(pt))
                        assert abs(el - tv) <= 1e-10 * max(1.0, abs(tv)), (s.sid, ci, el, tv)
            out[s.sid] = dict(acc=acc)
        else:
            Mm, f = s.implicit()
            acc = s.truth()
            out[s.sid] = dict(acc=acc, M=Mm, f=f)
        print('truth', s.sid, 'ok', flush=True)
    with open(TRUTH_PKL, 'wb') as fh:
        pickle.dump(out, fh)
    return out


if __name__ == '__main__':
    tr = build_truth(force=True)
    for s in corpus():
        print(s.sid, s.name, s.fam, 'dof', s.dof, 'params', s.params, 'applicable', sorted(s.applicable))
        print('   acc', tr[s.sid]['acc'][:2])

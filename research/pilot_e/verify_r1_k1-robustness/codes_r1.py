"""K1-robustness verification, round 1: codes NOT used by the pilot (C1-C9) nor by verifier r0.

Nothing in pilot_v1/ is modified.  We wrap trees.tree_cost (and reps.items_cost for the per-definition
charge of the complexity codes) so that new code names are dispatched here; all representation choices
(formulation min, CSE, greedy inlining, per-term template-vs-explicit choice, statement CSE) are re-run
by the pilot's own machinery under the new code, exactly as for C1/C2.

New codes
  TBX  Context-conditioned entropy code fitted on an INDEPENDENT corpus of ~100 textbook equations written
       below (mechanics, E&M, thermo, waves, QM, relativity).  Each node's symbol is coded given its
       parent context (root / Add-arg / Mul-arg / Pow-base / Pow-exp / Func-arg / Call-arg).  Symbols:
       Add, Mul, Pow, Call, Sym, each builtin (sin, cos, ..., Dq, Dt, Map), each specific small integer /
       rational, and an escape Int / Rat that then pays the C1 payload.  Add/Mul arity from the textbook
       arity histogram.  Leaf = symbol bits + log2|scope|; Call = symbol bits + log2|lib|.  Add-1/2
       (KT) smoothing.
  EQW  Eureqa/Formulize-style weighted complexity (a standard symbolic-regression complexity measure):
       variable 1, constant 1, add/sub 1, mul 1, div 2, neg 1, sqrt 4, pow 3, sin/cos 3, tan/cot/exp/log 4,
       Dq/Dt 4, Map 2, library call 1 (+args), each named definition 1 (assignment).  Scaled to bits by
       U = 4 bits per unit (so the pilot's gamma headers and log2(#formulations) flags keep their meaning).
  NCT  Plain tree-size (node count incl. leaves; n-ary Add/Mul counted once; the gplearn/'length' measure)
       x U = 4 bits; each named definition 1 unit.
  C2F  C2 re-fitted (Laplace, same features) on the representations that the FULL theory LAW9 actually
       uses (C1 choice at the same level), instead of on the all-explicit empty-theory representations.
       The pilot's choice of fitting corpus for C2 is arbitrary; this is the other end of it.
  C4-B64 / C4-B4 (run via the pilot's own C4 with B=64: IEEE double; B=4: order-of-magnitude constants).
"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), 'pilot_v1')
sys.path.insert(0, PILOT)
import sympy as sp
from common import gamma_len, int_cost, rat_cost
import trees as TR
import reps as R

NEW = ('TBX', 'EQW', 'NCT', 'C2F')
UNIT = float(os.environ.get('R1_UNIT', '4'))


# =============================================================== textbook corpus (independent)
def textbook_corpus():
    S = sp.symbols
    m, M, k, g, G, l, r, R_, t, w, b, F0, q, Q, E, B_, v, c, h, hb = S('m M k g G l r R t omega b F0 q Q E B v c h hbar')
    x, y, z, vx, vy, th, ph, p, px, py = S('x y z v_x v_y theta phi p p_x p_y')
    m1, m2, x1, x2, y1, y2, k1, k2, L_, I_, T_, V_, U_ = S('m1 m2 x1 x2 y1 y2 k1 k2 L I T V U')
    kB, n_, P_, Vol, N_, mu, eps0, rho, A, lam, f, a, d, e0, alpha, s = S(
        'k_B n P Vol N mu epsilon0 rho A lambda f a d e alpha s')
    Vf, Ff, Lf, Hf, Uf, psi, Vec = [sp.Function(n) for n in ('Vf', 'Ff', 'Lf', 'Hf', 'Uf', 'psi', 'Vec')]
    Dq = TR.Dq; Dt = TR.Dt
    eq = []
    add = eq.append
    # --- kinematics / Newtonian mechanics
    add(v * t + a * t ** 2 / 2); add(sp.sqrt(v ** 2 + 2 * a * x)); add(F0 / m); add(-k * x / m); add(-g)
    add(-G * M * m / r ** 2); add(-G * M / r ** 2); add(-G * M * m / r); add(m * v ** 2 / 2); add(m * g * y)
    add(k * x ** 2 / 2); add(m * v); add(m * v ** 2 / r); add(v ** 2 / r); add(2 * sp.pi / w); add(sp.sqrt(k / m))
    add(sp.sqrt(g / l)); add(2 * sp.pi * sp.sqrt(l / g)); add(-g / l * sp.sin(th)); add(-b * v / m - k * x / m)
    add(-b / m * v - k / m * x + F0 / m * sp.cos(w * t)); add(F0 * sp.cos(w * t)); add(sp.exp(-b * t / (2 * m)))
    add(-k1 * (x1 - x2)); add(-k * (x1 - x2) / m1); add(k * (x1 - x2) / m2); add(m1 * x1 + m2 * x2)
    add((m1 * x1 + m2 * x2) / (m1 + m2)); add(m1 * m2 / (m1 + m2)); add(x * py - y * px)
    add(sp.sqrt(x ** 2 + y ** 2)); add(sp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)); add(-G * M * x / (x ** 2 + y ** 2) ** sp.Rational(3, 2))
    add(-G * m2 * (x1 - x2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** sp.Rational(3, 2))
    add(m * l ** 2 * w ** 2 / 2 + m * g * l * sp.cos(th)); add(m * l ** 2 / 3); add(M * R_ ** 2 / 2); add(I_ * w ** 2 / 2)
    add(I_ * w); add(r * F0 * sp.sin(th)); add(v * sp.cos(th)); add(v * sp.sin(th) - g * t)
    add(v ** 2 * sp.sin(2 * th) / g); add(R_ * w ** 2 * sp.sin(th) * sp.cos(th) - g / R_ * sp.sin(th))
    add((m1 - m2) * g / (m1 + m2)); add(sp.atan(y / x) if False else y / x)
    # --- Lagrangian / Hamiltonian (named functions and derivative operators)
    add(T_ - V_); add(Dt(Dq(Lf(q, v), v)) - Dq(Lf(q, v), q)); add(p ** 2 / (2 * m) + Vf(x)); add(-Dq(Vf(x), x))
    add(Dq(Hf(q, p), p)); add(-Dq(Hf(q, p), q)); add(Ff(x) / m); add(-Dq(Uf(r), r)); add(Vf(r) + L_ ** 2 / (2 * m * r ** 2))
    add(m * (vx ** 2 + vy ** 2) / 2); add(Vec(-k * x, -k * y)); add(Vec(vx, vy - g * t)); add(Vec(x1 - x2, y1 - y2))
    add(Vf(sp.sqrt(x ** 2 + y ** 2))); add(Ff(r) * x / r)
    # --- E&M
    add(Q * q / (4 * sp.pi * eps0 * r ** 2)); add(Q / (4 * sp.pi * eps0 * r)); add(q * E); add(q * v * B_)
    add(q * B_ / m); add(m * v / (q * B_)); add(q * E / m + q * B_ * vy / m); add(-q * B_ * vx / m)
    add(mu * n_ * I_); add(mu * I_ / (2 * sp.pi * r)); add(Q ** 2 / (2 * c)); add(E ** 2 * eps0 / 2)
    add(I_ ** 2 * R_); add(Q / (R_ * c) * sp.exp(-t / (R_ * c))); add(1 / sp.sqrt(L_ * c))
    # --- thermo / stat mech
    add(n_ * R_ * T_ / Vol); add(N_ * kB * T_); add(3 * kB * T_ / 2); add(sp.exp(-E / (kB * T_)))
    add(sp.sqrt(3 * kB * T_ / m)); add(kB * sp.log(Q)); add(1 - T_ / M); add(P_ * Vol); add(-P_ * Vol + T_ * s)
    # --- waves / optics / QM / relativity
    add(A * sp.sin(k * x - w * t)); add(w / k); add(c / lam); add(h * f); add(h / p); add(hb * k)
    add(-hb ** 2 / (2 * m) * Dq(Dq(psi(x), x), x) + Vf(x) * psi(x)); add(hb ** 2 * k ** 2 / (2 * m))
    add(n_ ** 2 * hb ** 2 * sp.pi ** 2 / (2 * m * l ** 2)); add(-e0 ** 2 / (4 * sp.pi * eps0 * r))
    add(1 / sp.sqrt(1 - v ** 2 / c ** 2)); add(m * c ** 2); add(sp.sqrt(p ** 2 * c ** 2 + m ** 2 * c ** 4))
    add(2 * d * sp.sin(th)); add(lam * l / d); add(sp.sin(th) / sp.sin(ph)); add(rho * g * y); add(rho * v ** 2 / 2)
    add(-alpha * x); add(sp.exp(-alpha * t) * sp.cos(w * t)); add(a * sp.cosh(x) if False else a * sp.exp(x))
    return eq


CTXS = ('root', 'add', 'mul', 'pbase', 'pexp', 'farg', 'carg')
SMALL_INTS = list(range(-4, 5))
SMALL_RATS = [(1, 2), (-1, 2), (3, 2), (-3, 2), (1, 3), (1, 4)]


def _sym_of(kind, label):
    if kind in ('Add', 'Mul', 'Pow', 'Call', 'Sym'):
        return kind
    if kind == 'Func':
        return 'F:' + label
    if kind == 'Int':
        return 'I:%d' % label if label in SMALL_INTS else 'Iesc'
    if kind == 'Rat':
        return 'R:%d/%d' % label if tuple(label) in SMALL_RATS else 'Resc'
    if kind == 'Rel':
        return 'Call'
    raise ValueError(kind)


def _child_ctx(kind, j):
    return {'Add': 'add', 'Mul': 'mul', 'Func': 'farg', 'Call': 'carg', 'Rel': 'carg'}.get(kind) or ('pbase' if j == 0 else 'pexp')


ALLSYMS = (['Add', 'Mul', 'Pow', 'Call', 'Sym', 'Iesc', 'Resc'] + ['I:%d' % v for v in SMALL_INTS]
           + ['R:%d/%d' % pq for pq in SMALL_RATS] + ['F:' + n for n in ('sin', 'cos', 'tan', 'cot', 'exp', 'log', 'Dq', 'Dt', 'Map')])


class TBModel:
    def __init__(self, alpha=0.5):
        cnt = {cx: {s: alpha for s in ALLSYMS} for cx in CTXS}
        ar = {n: alpha for n in range(2, 17)}
        TR_ = TR
        for e in textbook_corpus():
            tid = TR_.from_sympy(e.xreplace({sp.pi: sp.Symbol('pi')}), 'TXT')
            self._walk(tid, 'root', cnt, ar)
        self.bits = {cx: {} for cx in CTXS}
        for cx in CTXS:
            tot = sum(cnt[cx].values())
            for s in ALLSYMS:
                self.bits[cx][s] = -math.log2(cnt[cx][s] / tot)
        tot = sum(ar.values())
        self.arbits = {n: -math.log2(v / tot) for n, v in ar.items()}
        self.counts = cnt

    def _walk(self, i, cx, cnt, ar):
        kind, label, ch = TR.NODES[i]
        cnt[cx][_sym_of(kind, label)] += 1
        if kind in ('Add', 'Mul'):
            ar[min(len(ch), 16)] += 1
        for j, c in enumerate(ch):
            self._walk(c, _child_ctx(kind, j), cnt, ar)


_TB = [None]
_TBC = {}


def tb_prof(i, cx='root'):
    """(base bits, #leaves, #funcs, #calls) under TBX; leaves/calls get scope/lib bits added later."""
    key = (i, cx)
    r = _TBC.get(key)
    if r is not None:
        return r
    if _TB[0] is None:
        _TB[0] = TBModel()
    M = _TB[0]
    kind, label, ch = TR.NODES[i]
    s = _sym_of(kind, label)
    b = M.bits[cx][s]
    nl = nc = 0
    if kind in ('Add', 'Mul'):
        b += M.arbits[min(len(ch), 16)]
    elif kind == 'Sym':
        nl = 1
    elif kind == 'Int' and s == 'Iesc':
        b += int_cost(label)
    elif kind == 'Rat' and s == 'Resc':
        b += rat_cost(*label)
    elif kind == 'Call':
        nc = 1
    for j, c in enumerate(ch):
        p = tb_prof(c, _child_ctx(kind, j))
        b += p[0]; nl += p[1]; nc += p[2]
    r = (b, nl, nc)
    _TBC[key] = r
    return r


# =============================================================== Eureqa-style weighted complexity / node count
W = dict(var=1, const=1, add=1, mul=1, div=2, neg=1, sqrt=4, pow=3, sin=3, cos=3, tan=4, cot=4, exp=4, log=4,
         Dq=4, Dt=4, Map=2, call=1)
_WSET = os.environ.get('R1_WSET', 'eureqa')
if _WSET == 'pysr':           # PySR default: every operator, variable and constant has complexity 1
    W = {k: 1 for k in W}
elif _WSET == 'eureqa_pow5':  # heavier power / sqrt
    W = dict(W, pow=5, sqrt=5)
_EQ = {}


def _neg_coef(i):
    kind, label, ch = TR.NODES[i]
    if kind == 'Int':
        return label < 0
    if kind == 'Rat':
        return label[0] < 0
    if kind == 'Mul':
        k0, l0, _ = TR.NODES[ch[0]]
        return (k0 == 'Int' and l0 < 0) or (k0 == 'Rat' and l0[0] < 0)
    return False


def eq_cx(i, negate=False):
    """Weighted complexity in binary-operator form.  negate=True: sign absorbed by an enclosing subtraction."""
    key = (i, negate)
    r = _EQ.get(key)
    if r is not None:
        return r
    kind, label, ch = TR.NODES[i]
    if kind == 'Sym':
        r = W['var']
    elif kind in ('Int', 'Rat'):
        r = W['const']
    elif kind == 'Add':
        r = sum(eq_cx(c, negate=_neg_coef(c)) for c in ch) + W['add'] * (len(ch) - 1)
        if all(_neg_coef(c) for c in ch):
            r += W['neg']
    elif kind == 'Mul':
        args = list(ch)
        k0, l0, _ = TR.NODES[args[0]]
        neg = (k0 == 'Int' and l0 < 0) or (k0 == 'Rat' and l0[0] < 0)
        if k0 == 'Int' and l0 == -1:
            args = args[1:]
        num, den = [], []
        for c in args:
            kc, lc, cc = TR.NODES[c]
            if kc == 'Pow':
                ke, le, _ = TR.NODES[cc[1]]
                if ke == 'Int' and le < 0:
                    den.append(c)
                    continue
                if ke == 'Rat' and le[0] < 0:
                    den.append(c)
                    continue
            if kc == 'Rat' and abs(lc[0]) == 1 and lc[1] != 1:
                den.append(('RD', lc[1]))
                continue
            num.append(c)
        r = 0
        for c in num:
            kc, lc, _ = TR.NODES[c]
            r += eq_cx(c)
        if num:
            r += W['mul'] * (len(num) - 1)
        else:
            r += W['const']
        for d_ in den:
            if isinstance(d_, tuple):
                r += W['const'] + W['div']
            else:
                kc, lc, cc = TR.NODES[d_]
                ke, le, _ = TR.NODES[cc[1]]
                inv = sp.Rational(*le) if ke == 'Rat' else sp.Integer(le)
                inv = -inv
                r += W['div'] + eq_cx(cc[0])
                if inv == sp.Rational(1, 2):
                    r += W['sqrt']
                elif inv != 1:
                    r += W['pow'] + W['const']
        if neg and not negate:
            r += W['neg']
    elif kind == 'Pow':
        b_, e_ = ch
        ke, le, _ = TR.NODES[e_]
        if ke == 'Rat' and le == (1, 2):
            r = eq_cx(b_) + W['sqrt']
        elif ke == 'Rat' and le == (-1, 2):
            r = W['const'] + W['div'] + W['sqrt'] + eq_cx(b_)
        elif ke == 'Int' and le == -1:
            r = W['const'] + W['div'] + eq_cx(b_)
        elif ke == 'Int' and le < 0:
            r = W['const'] + W['div'] + W['pow'] + W['const'] + eq_cx(b_)
        else:
            r = eq_cx(b_) + eq_cx(e_) + W['pow']
        if negate:
            r += W['neg']
    elif kind == 'Func':
        r = W.get(label, 3) + sum(eq_cx(c) for c in ch)
        if negate:
            r += W['neg']
    elif kind in ('Call', 'Rel'):
        r = W['call'] + sum(eq_cx(c) for c in ch)
        if negate:
            r += W['neg']
    else:
        raise ValueError(kind)
    _EQ[key] = r
    return r


def node_count(i):
    return TR.SIZE[i]


# =============================================================== dispatch
_orig_tree_cost = TR.tree_cost


def tree_cost(i, code, sizes, nlib, nbuiltin):
    if code == 'TBX':
        b, nl, nc = tb_prof(i)
        tot = sum(sizes.get(rl, 0) for rl in TR.ROLES)
        return b + nl * TR.lg(tot) + nc * TR.lg(nlib)
    if code == 'EQW':
        return UNIT * eq_cx(i)
    if code == 'NCT':
        return UNIT * node_count(i)
    if code == 'C2F':
        return _orig_tree_cost(i, 'C2', sizes, nlib, nbuiltin)
    return _orig_tree_cost(i, code, sizes, nlib, nbuiltin)


TR.tree_cost = tree_cost

_orig_items_cost = R.items_cost


def items_cost(items, code, ctx, n_param=None):
    c = _orig_items_cost(items, code, ctx, n_param)
    if code in ('EQW', 'NCT'):
        c += UNIT * sum(1 for l, _ in items if l is not None)      # assignment per named definition
    return c


R.items_cost = items_cost

import laws as LW  # noqa: E402  (laws binds R.items_cost lazily through R.greedy_inline -> costfn)
_orig_stmt = LW._stmt_items_cost


def _stmt_items_cost(items, nformals, code, nlib, nbuiltin, text_extra):
    c = _orig_stmt(items, nformals, code, nlib, nbuiltin, text_extra)
    if code in ('EQW', 'NCT'):
        c += UNIT * (1 + sum(1 for l, _ in items if l is not None))  # statement name '=' + each temp
    return c


LW._stmt_items_cost = _stmt_items_cost

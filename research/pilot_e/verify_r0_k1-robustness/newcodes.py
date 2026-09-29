"""K1-robustness verification: additional reasonable codes plugged into the pilot's machinery.

Nothing in pilot_v1/ is modified; we monkeypatch trees.tree_cost, reps.items_cost, laws.item_cost
and laws.dec_cost at import time so that new code names are dispatched here.  All representation
choices (formulation min, CSE, greedy inlining, per-term template choice) are re-run under the
new code exactly as the pilot does for C1/C2/C3.

New codes
  C6  LaTeX-token code: tokens of sp.latex(tree) (\\left/\\right dropped, braces dropped), 7 bits/token
      (vocabulary 128).  Definitions pay name + '=' (3 tokens); law statements pay text + 3 tokens.
  C7  Symbolic-regression prefix code (PySR/DSR style): binary operators + - * / ^, unary neg, sqrt,
      builtins, library calls; every token drawn uniformly from one alphabet
      A = 7 ops + builtins + library + scope + 1 const-token; ints/rats pay C1 payload after the const token.
  C9  C1 with a GLOBAL symbol dictionary: a leaf costs log2(#distinct corpus symbols + temps + outs)
      instead of log2(|system scope|) (the pilot never pays to announce a per-system scope).
  C10 Entropy code fitted on an independent corpus of ~70 textbook equations (written below):
      kind, arity, builtin-function-name and integer/rational distributions (Laplace); leaf =
      kind bits + log2|scope|; call = kind bits + log2|lib|.
  C11 C1 with the SAME within-item CSE applied to law statements (and decoder rules) as to models
      (the pilot codes statements without CSE; DEVIATIONS C7).
"""
import os, sys, math, re
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), 'pilot_v1')
sys.path.insert(0, PILOT)
import sympy as sp
from common import gamma_len, int_cost, rat_cost, LOG95, KINDS, builtins_for
import corpus as CP
import trees as TR
import reps as R
import laws as LW

NEW = ('C6', 'C7', 'C7b', 'C9', 'C10', 'C11')
LOG128 = 7.0

# ------------------------------------------------------------------ C6 LaTeX tokens
_TOKRE = re.compile(r'\\[a-zA-Z]+|[A-Za-z0-9]|[^\sA-Za-z0-9{}]')
_LTX = {}


def latex_tokens_expr(e):
    s = sp.latex(e)
    s = s.replace('\\left', '').replace('\\right', '')
    return len(_TOKRE.findall(s))


def latex_len(i):
    r = _LTX.get(i)
    if r is None:
        r = latex_tokens_expr(TR.to_sympy_print(i))
        _LTX[i] = r
    return r


# ------------------------------------------------------------------ C7 SR prefix code
_SR = {}


def _is_neg(i):
    kind, label, ch = TR.NODES[i]
    if kind == 'Int':
        return label < 0
    if kind == 'Rat':
        return label[0] < 0
    if kind == 'Mul':
        k0, l0, _ = TR.NODES[ch[0]]
        return (k0 == 'Int' and l0 < 0) or (k0 == 'Rat' and l0[0] < 0)
    return False


def sr_prof(i, negate=False):
    """(n_tokens, payload_bits, n_func, n_call) of the binary-operator prefix form of tree i;
    negate=True: count the tree as -tree (its sign absorbed by an enclosing subtraction)."""
    key = (i, negate)
    r = _SR.get(key)
    if r is not None:
        return r
    kind, label, ch = TR.NODES[i]
    n = pay = nf = nc = 0

    def add(p):
        nonlocal n, pay, nf, nc
        n += p[0]; pay += p[1]; nf += p[2]; nc += p[3]
    if kind == 'Sym':
        n = 1
    elif kind == 'Int':
        v = abs(label) if negate else label
        n = 1
        pay = int_cost(v)
        if v == -1:
            pass
    elif kind == 'Rat':
        p_, q_ = label
        if negate:
            p_ = abs(p_)
        n = 1
        pay = rat_cost(p_, q_)
    elif kind == 'Add':
        pos = [c for c in ch if not _is_neg(c)]
        neg = [c for c in ch if _is_neg(c)]
        for c in pos:
            add(sr_prof(c))
        for c in neg:
            add(sr_prof(c, negate=True))
        n += len(ch) - 1            # binary + / - nodes
        if not pos:
            n += 1                  # unary neg
    elif kind == 'Mul':
        args = list(ch)
        k0, l0, _ = TR.NODES[args[0]]
        sign_neg = False
        if k0 == 'Int' and l0 < 0:
            if l0 == -1:
                args = args[1:]
            sign_neg = True
        elif k0 == 'Rat' and l0[0] < 0:
            sign_neg = True
        num, den = [], []
        for idx, c in enumerate(args):
            kc, lc, cc = TR.NODES[c]
            if kc == 'Pow':
                ke_, le_, _ = TR.NODES[cc[1]]
                if ke_ == 'Int' and le_ < 0:
                    den.append((cc[0], -le_))
                    continue
            if kc == 'Rat' and lc[0] != 0 and abs(lc[0]) == 1:
                den.append(('RATDEN', lc[1]))       # 1/q or -1/q -> / q
                continue
            num.append(c)
        # numerator
        if num:
            for idx, c in enumerate(num):
                kc, lc, _ = TR.NODES[c]
                if sign_neg and idx == 0 and kc in ('Int', 'Rat'):
                    add(sr_prof(c, negate=True))
                else:
                    add(sr_prof(c))
            n += len(num) - 1
        else:
            n += 1; pay += int_cost(1)
        if den:
            for b, e in den:
                if b == 'RATDEN':
                    n += 1; pay += int_cost(e)
                else:
                    add(sr_prof(b))
                    if e != 1:
                        n += 2; pay += int_cost(e)   # ^ and exponent
            n += len(den) - 1 + 1                     # * among den, one /
        if sign_neg and not negate:
            n += 1                                     # unary neg
    elif kind == 'Pow':
        b, e = ch
        ke_, le_, _ = TR.NODES[e]
        if ke_ == 'Rat' and le_ == (1, 2):
            add(sr_prof(b)); n += 1                    # sqrt
        elif ke_ == 'Rat' and le_ == (-1, 2):
            add(sr_prof(b)); n += 3; pay += int_cost(1)  # 1 / sqrt(b)
        elif ke_ == 'Int' and le_ < 0:
            add(sr_prof(b)); n += 2; pay += int_cost(1)  # 1 / b
            if le_ != -1:
                n += 2; pay += int_cost(-le_)
        else:
            add(sr_prof(b)); add(sr_prof(e)); n += 1
        if negate:
            n += 1
    elif kind in ('Func', 'Call', 'Rel'):
        for c in ch:
            add(sr_prof(c))
        n += 1
        if kind == 'Func':
            nf += 1
        elif kind == 'Call':
            nc += 1
        if negate:
            n += 1
    else:
        raise ValueError(kind)
    r = (n, pay, nf, nc)
    _SR[key] = r
    return r


_SRL = {}


def sr_leaves(i):
    r = _SRL.get(i)
    if r is None:
        kind, label, ch = TR.NODES[i]
        r = 1 if kind in ('Sym', 'Int', 'Rat') else sum(sr_leaves(c) for c in ch)
        _SRL[i] = r
    return r


# ------------------------------------------------------------------ C9 global dictionary
_NGLOBAL = None


def nglobal():
    global _NGLOBAL
    if _NGLOBAL is None:
        syms = set()
        for s in CP.corpus():
            syms |= set(s.state) | set(s.params)
            if s.fam == 'G':
                syms |= set(s.z)
        syms.add(CP.t)
        _NGLOBAL = len(syms)
    return _NGLOBAL


# ------------------------------------------------------------------ C10 textbook-fitted prior
def textbook_corpus():
    S = sp.symbols
    F, m, a, G, M, r, E, c, v, x, x0, v0, t, d, T, l, g, k, w, A, phi, p, th, I, P, W, U = S(
        'F m a G M r E c v x x0 v0 t d T l g k omega A phi p theta I P W U')
    q, q1, q2, B, V, C, eps0, sig, Rr, L, n1, n2, f, do, di, vo, vs, n, Rg, Q, Tc, Th, kB, h, lam = S(
        'q q1 q2 B V C epsilon0 sigma R L n1 n2 f d_o d_i v_o v_s n R_g Q T_c T_h k_B h lambda')
    rho, eta, zeta, mu0, hbar, N, m1, m2, x1, x2, v1, v2, ve, m0, mf, y, kk, H, Om, tau, e, R0 = S(
        'rho eta zeta mu0 hbar N m1 m2 x1 x2 v1 v2 v_e m0 m_f y kappa H Omega tau e R0')
    psi, Vf, ff, Ef, Uf = [sp.Function(nm) for nm in ('psi', 'V', 'f', 'E', 'U')]
    pi = sp.pi
    eqs = [
        m * a, G * m1 * m2 / r ** 2, m * c ** 2, m * v ** 2 / 2, x0 + v0 * t + a * t ** 2 / 2, v0 + a * t,
        v0 ** 2 + 2 * a * d, 2 * pi * sp.sqrt(l / g), sp.sqrt(k / m), A * sp.cos(w * t + phi), m * v,
        r * F * sp.sin(th), m1 * r ** 2 + m2 * x ** 2, I * w ** 2 / 2, F * v, F * d * sp.cos(th),
        -G * M * m / r, k * x ** 2 / 2, -k * x, q * E, q * v * B, kk * q1 * q2 / r ** 2, kk * q / r,
        sig / eps0, eps0 * A / d, C * V ** 2 / 2, V / Rr, I ** 2 * Rr, Rr * C, 1 / sp.sqrt(L * C),
        n1 * sp.sin(th) - n2 * sp.sin(phi), 1 / do + 1 / di - 1 / f, f * (v + vo) / (v - vs), n * Rg * T / V,
        Q - W, 1 - Tc / Th, sp.sqrt(3 * kB * T / m), h * f, h / p, -E / n ** 2, 1 / sp.sqrt(1 - v ** 2 / c ** 2),
        (t - v * x / c ** 2) / sp.sqrt(1 - v ** 2 / c ** 2), P + rho * v ** 2 / 2 + rho * g * h,
        6 * pi * eta * r * v, -2 * zeta * w * v - w ** 2 * x, v ** 2 * sp.sin(2 * th) / g, v ** 2 / r,
        4 * pi ** 2 * a ** 3 / (G * M), sp.sqrt(2 * G * M / r), -g * sp.sin(th) / l, A * sp.sin(k * x - w * t),
        mu0 * I / (2 * pi * r), n ** 2 * pi ** 2 * hbar ** 2 / (2 * m * L ** 2), sig * A * T ** 4,
        m1 * m2 / (m1 + m2), (m1 * x1 + m2 * x2) / (m1 + m2), (m1 - m2) * v1 / (m1 + m2) + 2 * m2 * v2 / (m1 + m2),
        ve * sp.log(m0 / mf), m * v ** 2 / 2 + k * x ** 2 / 2, m * l ** 2 * w ** 2 / 2 + m * g * l * sp.cos(th),
        p ** 2 / (2 * m) + m * w ** 2 * x ** 2 / 2, q * (E + v * B), -k * (x1 - x2), -G * m1 * m2 * (x1 - x2) / r ** 3,
        sp.sqrt((x1 - x2) ** 2 + (y - x0) ** 2), R0 * sp.exp(-t / tau), -hbar ** 2 / (2 * m) + Vf(x),
        psi(x, t) * sp.exp(-sp.I * E * t / hbar) if False else psi(x) * sp.exp(-E * t / hbar),
        Ef(v) + Uf(x), -Vf(r) / r, ff(x) * ff(y), m * g * y, -m * g, 2 * pi / T, 2 * pi * r / T,
        -e ** 2 / (8 * pi * eps0 * r), Om ** 2 * r * sp.sin(th) * sp.cos(th) - g * sp.sin(th) / r,
    ]
    return eqs


class TextbookPrior:
    def __init__(self):
        kc = {kk: 1.0 for kk in KINDS}
        ac = {nn: 1.0 for nn in range(2, 17)}
        fc = {nm: 1.0 for nm in builtins_for('D2')}
        ic = {}
        self.nint = 0
        eqs = textbook_corpus()
        self.neq = len(eqs)

        def walk(x):
            if isinstance(x, sp.Symbol) or x in (sp.pi, sp.E):
                kc['Sym'] += 1
                return
            if isinstance(x, sp.Integer):
                kc['Int'] += 1; ic[('i', int(x))] = ic.get(('i', int(x)), 0) + 1; return
            if isinstance(x, sp.Rational):
                kc['Rat'] += 1; ic[('r', int(x.p), int(x.q))] = ic.get(('r', int(x.p), int(x.q)), 0) + 1; return
            if isinstance(x, sp.Add):
                kc['Add'] += 1; ac[min(len(x.args), 16)] += 1
            elif isinstance(x, sp.Mul):
                kc['Mul'] += 1; ac[min(len(x.args), 16)] += 1
            elif isinstance(x, sp.Pow):
                kc['Pow'] += 1
            elif isinstance(x, sp.core.function.AppliedUndef):
                kc['Call'] += 1
            elif isinstance(x, sp.Function):
                kc['Func'] += 1
                nm = x.func.__name__
                fc[nm] = fc.get(nm, 1.0) + 1
            for y_ in x.args:
                walk(y_)
        for e in eqs:
            walk(e)
        s = sum(kc.values()); self.kind = {kk: v / s for kk, v in kc.items()}
        s = sum(ac.values()); self.arity = {kk: v / s for kk, v in ac.items()}
        s = sum(fc.values()); self.func = {kk: v / s for kk, v in fc.items()}
        # integers/rationals: seen values with counts, plus escape (count 1) -> C1 payload
        tot = sum(ic.values()) + 1.0
        self.num = {kk: v / tot for kk, v in ic.items()}
        self.esc = 1.0 / tot
        self.counts = dict(kind=kc, arity=ac, func=fc, num={str(k_): v for k_, v in ic.items()})

    def kind_bits(self, kind):
        return -math.log2(self.kind['Func' if kind == 'Rel' else kind])

    def arity_bits(self, n):
        return -math.log2(self.arity[min(n, 16)])

    def func_bits(self, name):
        return -math.log2(self.func.get(name, min(self.func.values())))

    def int_bits(self, v):
        p = self.num.get(('i', v))
        return -math.log2(p) if p else -math.log2(self.esc) + int_cost(v)

    def rat_bits(self, pq):
        p = self.num.get(('r',) + tuple(pq))
        return -math.log2(p) if p else -math.log2(self.esc) + rat_cost(*pq)


TBP = None
_C10 = {}


def c10_prof(i):
    """(base bits excluding leaf-scope and call-name bits, n_leaves, n_call)."""
    r = _C10.get(i)
    if r is not None:
        return r
    kind, label, ch = TR.NODES[i]
    b = TBP.kind_bits(kind)
    nl = nc = 0
    if kind in ('Add', 'Mul'):
        b += TBP.arity_bits(len(ch))
    elif kind == 'Sym':
        nl = 1
    elif kind == 'Int':
        b += TBP.int_bits(label)
    elif kind == 'Rat':
        b += TBP.rat_bits(label)
    elif kind == 'Func':
        b += TBP.func_bits(label)
    elif kind == 'Call':
        nc = 1
    for c in ch:
        p = c10_prof(c)
        b += p[0]; nl += p[1]; nc += p[2]
    r = (b, nl, nc)
    _C10[i] = r
    return r


# ------------------------------------------------------------------ dispatch
_orig_tree_cost = TR.tree_cost


def tree_cost(i, code, sizes, nlib, nbuiltin):
    global TBP
    if code not in NEW:
        return _orig_tree_cost(i, code, sizes, nlib, nbuiltin)
    tot = sum(sizes.get(r, 0) for r in TR.ROLES)
    if code == 'C6':
        return latex_len(i) * LOG128
    if code == 'C7':
        n, pay, nf, nc = sr_prof(i)
        alpha = 7 + nbuiltin + nlib + tot + 1
        return n * math.log2(alpha) + pay
    if code == 'C7b':
        # two-part token code: 1 flag bit (operator vs leaf) + uniform choice within the class
        n, pay, nf, nc = sr_prof(i)
        nleaf = sr_leaves(i)
        nop = n - nleaf
        return n * 1.0 + nop * math.log2(7 + nbuiltin + nlib) + nleaf * TR.lg(tot + 1) + pay
    if code == 'C9':
        if 'hole' in sizes:        # law statements / decoder rules: keep local formal scope
            return _orig_tree_cost(i, 'C1', sizes, nlib, nbuiltin)
        s2 = dict(sizes)
        s2['state'] = nglobal()
        s2['param'] = 0
        return _orig_tree_cost(i, 'C1', s2, nlib, nbuiltin)
    if code == 'C10':
        if TBP is None:
            TBP = TextbookPrior()
        b, nl, nc = c10_prof(i)
        return b + nl * TR.lg(tot) + nc * TR.lg(nlib)
    if code == 'C11':
        return _orig_tree_cost(i, 'C1', sizes, nlib, nbuiltin)
    raise ValueError(code)


TR.tree_cost = tree_cost

_orig_items_cost = R.items_cost


def items_cost(items, code, ctx, n_param=None):
    tot = _orig_items_cost(items, code, ctx, n_param)
    if code == 'C6':
        tot += sum(3 * LOG128 for l, _ in items if l is not None)
    return tot


R.items_cost = items_cost

_orig_item_cost = LW.item_cost


def item_cost(expr, formals, code, nlib, nbuiltin, text_extra=3):
    if code == 'C6':
        holes = {s: j for j, s in enumerate(formals)}
        tid = TR.from_sympy(expr, 'LAW', law_mode=True, holes=holes)
        return (latex_len(tid) + text_extra) * LOG128
    if code == 'C11':
        # statement CSE'd exactly like model items: temps see formals + earlier temps
        if isinstance(expr, sp.Rel):
            return _orig_item_cost(expr, formals, 'C1', nlib, nbuiltin, text_extra)
        repl, red = sp.cse([expr], symbols=sp.numbered_symbols('_s'), optimizations='basic')
        tot = gamma_len(len(formals) + 1) + gamma_len(len(repl) + 2)
        scope = list(formals)
        for sym, e in repl:
            holes = {s: j for j, s in enumerate(scope)}
            tid = TR.from_sympy(e, 'LAW', law_mode=True, holes=holes)
            tot += TR.tree_cost(tid, 'C1', {'hole': len(scope)}, nlib, nbuiltin)
            scope.append(sym)
        holes = {s: j for j, s in enumerate(scope)}
        tid = TR.from_sympy(red[0], 'LAW', law_mode=True, holes=holes)
        tot += TR.tree_cost(tid, 'C1', {'hole': len(scope)}, nlib, nbuiltin)
        return min(tot, _orig_item_cost(expr, formals, 'C1', nlib, nbuiltin, text_extra))
    return _orig_item_cost(expr, formals, code, nlib, nbuiltin, text_extra)


LW.item_cost = item_cost

_orig_dec_cost = LW.dec_cost


def dec_cost(code, level):
    if code == 'C6':
        tot = 0.0
        for lhs, rhs in LW.dec_rules():
            for side in (lhs, rhs):
                syms = sorted(side.free_symbols, key=str)
                holes = {s: j for j, s in enumerate(syms)}
                tid = TR.from_sympy(side, 'DEC', law_mode=True, holes=holes)
                tot += latex_len(tid) * LOG128
            tot += 2 * LOG128
        return tot
    if code == 'C11':
        return _orig_dec_cost('C1', level)
    return _orig_dec_cost(code, level)


LW.dec_cost = dec_cost

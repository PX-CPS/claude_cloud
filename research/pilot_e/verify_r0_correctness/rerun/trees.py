"""Hash-consed expression trees, conversion from/to sympy, and the codes C1-C4 (+C5).

A tree is an int id into NODES; NODES[id] = (kind, label, children_ids).
Kinds: Add Mul Pow Func Call Sym Int Rat (+ 'Rel' only in the POS statement).
Leaf labels (kind 'Sym'):
  ('s', Symbol)        global symbol (state or parameter; role from corpus.is_state_symbol)
  ('t', sid, i)        temp (system-local)
  ('o', sid, i)        earlier output reference (system-local)
  ('c', sid, i)        C4 lumped constant (system-local)
  ('d', i)             C4 corpus dictionary constant (global; miner-on C4 only)
  ('h', i)             hole / formal argument (templates, law statements)
"""
import math, zlib
import sympy as sp
from sympy.core.function import AppliedUndef
from common import KINDS, LOG95, gamma_len, int_cost, rat_cost
import corpus as CP

NODES = []
IDX = {}
SIZE = []


def mk(kind, label=None, ch=()):
    key = (kind, label, ch)
    i = IDX.get(key)
    if i is None:
        i = len(NODES)
        NODES.append(key)
        IDX[key] = i
        SIZE.append(1 + sum(SIZE[c] for c in ch))
    return i


def leaf(label):
    return mk('Sym', label)


# ---------------------------------------------------------------- sympy function heads
FUNC_BUILTIN = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'cot': sp.cot, 'exp': sp.exp, 'log': sp.log}
Dq = sp.Function('Dq')
Dt = sp.Function('Dt')
V = sp.Function('V')
Cn = sp.Function('Cn')
D = sp.Function('D')
K = sp.Function('K')
Fh1, Fh2, Vh1, Vh2, Fg, Vg = [sp.Function(n) for n in ('Fh1', 'Fh2', 'Vh1', 'Vh2', 'Fg', 'Vg')]
LAW_TEMPLATES = {'Fh1', 'Fh2', 'Vh1', 'Vh2', 'Fg', 'Vg'}
C3NAME = {'V': 'V', 'Cn': 'Cn', 'D': 'D', 'K': 'K', 'Fh1': 'F1', 'Fh2': 'F2', 'Vh1': 'H1', 'Vh2': 'H2',
          'Fg': 'Fg', 'Vg': 'Vg', 'Dq': 'Dq', 'Dt': 'Dt'}


def local_label_of(sym, sid):
    """Map builder/CSE/lump symbol names (prefixed '_') to local labels."""
    n = sym.name
    if not n.startswith('_'):
        return None
    tag, rest = n[1], n[2:]
    if tag in ('P', 'x', 'V', 'L', 'R'):   # temps: N3 pair temps, CSE temps, Vtmp, Ltmp
        return ('t', sid, n[1:])
    if tag == 'O':
        return ('o', sid, int(rest))
    if tag == 'c':
        return ('c', sid, int(rest))
    raise ValueError(n)


def from_sympy(e, sid, level_builtins=None, law_mode=False, holes=None):
    """Convert a sympy expression into a tree id.
    law_mode: Dq/Dt/Dv/Sm/Cr/d are Calls (statement language); holes maps formal symbols -> hole index."""
    memo = {}

    def rec(x):
        if x in memo:
            return memo[x]
        if isinstance(x, sp.Symbol):
            if holes is not None and x in holes:
                r = leaf(('h', holes[x]))
            else:
                ll = local_label_of(x, sid)
                r = leaf(ll if ll is not None else ('s', x))
        elif isinstance(x, sp.Integer):
            r = mk('Int', int(x))
        elif isinstance(x, sp.Rational):
            r = mk('Rat', (int(x.p), int(x.q)))
        elif isinstance(x, sp.Float):
            raise ValueError('float in tree %r' % x)
        elif isinstance(x, sp.Add):
            r = mk('Add', None, tuple(rec(a) for a in x.args))
        elif isinstance(x, sp.Mul):
            r = mk('Mul', None, tuple(rec(a) for a in x.args))
        elif isinstance(x, sp.Pow):
            r = mk('Pow', None, (rec(x.base), rec(x.exp)))
        elif isinstance(x, AppliedUndef):
            name = x.func.__name__
            if name in ('Dq', 'Dt') and not law_mode:
                r = mk('Func', name, tuple(rec(a) for a in x.args))
            else:
                r = mk('Call', name, tuple(rec(a) for a in x.args))
        elif isinstance(x, sp.Function) and x.func.__name__ in FUNC_BUILTIN:
            r = mk('Func', x.func.__name__, tuple(rec(a) for a in x.args))
        elif isinstance(x, (sp.StrictGreaterThan,)):
            r = mk('Rel', '>', (rec(x.lhs), rec(x.rhs)))
        else:
            raise ValueError('unknown node %r (%s)' % (x, type(x)))
        memo[x] = r
        return r
    return rec(e)


# ---------------------------------------------------------------- role bookkeeping
def leaf_role(label):
    tag = label[0]
    if tag == 's':
        return 'state' if CP.is_state_symbol(label[1]) else 'param'
    return {'t': 'temp', 'o': 'out', 'c': 'param', 'd': 'param', 'h': 'hole'}[tag]


ROLES = ('state', 'param', 'temp', 'out', 'hole')
_PROFILE = {}


def profile(i):
    """C1/C2 decomposition of a tree: (c1_base, c2_base, leafcounts[5], nfunc, ncall).
    c1_base: 3 bits per node + gamma arities + integer payloads (+ Rel kind = 3 bits).
    c2_base: -log2 p(kind) + -log2 p(arity) + integer payloads (FREQ must be fitted)."""
    r = _PROFILE.get(i)
    if r is not None and r[5] == FREQ.version:
        return r
    kind, label, ch = NODES[i]
    lc = [0, 0, 0, 0, 0]
    nf = nc = 0
    b1 = 3.0
    b2 = FREQ.kind_bits(kind)
    if kind in ('Add', 'Mul'):
        b1 += gamma_len(len(ch) - 1)
        b2 += FREQ.arity_bits(len(ch))
    elif kind == 'Sym':
        lc[ROLES.index(leaf_role(label))] += 1
    elif kind == 'Int':
        b1 += int_cost(label)
        b2 += int_cost(label)
    elif kind == 'Rat':
        b1 += rat_cost(*label)
        b2 += rat_cost(*label)
    elif kind == 'Func':
        nf += 1
    elif kind == 'Call':
        nc += 1
    for c in ch:
        p = profile(c)
        b1 += p[0]
        b2 += p[1]
        for j in range(5):
            lc[j] += p[2][j]
        nf += p[3]
        nc += p[4]
    r = (b1, b2, tuple(lc), nf, nc, FREQ.version)
    _PROFILE[i] = r
    return r


class FreqModel:
    """C2 static frequency model (Laplace-smoothed)."""
    def __init__(self):
        self.version = 0
        self.kind = {k: 1.0 / len(KINDS) for k in KINDS}
        self.arity = {n: 1.0 / 15 for n in range(2, 17)}
        self.role = {r: 0.25 for r in ('state', 'param', 'temp', 'out')}

    def fit(self, trees):
        kc = {k: 1.0 for k in KINDS}
        ac = {n: 1.0 for n in range(2, 17)}
        rc = {r: 1.0 for r in ('state', 'param', 'temp', 'out')}
        seen = []

        def walk(i):
            kind, label, ch = NODES[i]
            kc[kind if kind in kc else 'Func'] += 1
            if kind in ('Add', 'Mul'):
                ac[min(len(ch), 16)] += 1
            if kind == 'Sym':
                r = leaf_role(label)
                if r in rc:
                    rc[r] += 1
            for c in ch:
                walk(c)
        for t in trees:
            walk(t)
        s = sum(kc.values())
        self.kind = {k: v / s for k, v in kc.items()}
        s = sum(ac.values())
        self.arity = {k: v / s for k, v in ac.items()}
        s = sum(rc.values())
        self.role = {k: v / s for k, v in rc.items()}
        self.version += 1

    def kind_bits(self, kind):
        if kind == 'Rel':
            kind = 'Func'
        return -math.log2(self.kind[kind])

    def arity_bits(self, n):
        return -math.log2(self.arity[min(n, 16)])

    def role_bits(self, role):
        if role == 'hole':
            role = 'param'
        return -math.log2(self.role[role])

    def to_json(self):
        return dict(kind=self.kind, arity={str(k): v for k, v in self.arity.items()}, role=self.role)

    def from_json(self, d):
        self.kind = dict(d['kind'])
        self.arity = {int(k): v for k, v in d['arity'].items()}
        self.role = dict(d['role'])
        self.version += 1


FREQ = FreqModel()


def lg(n):
    return math.log2(n) if n > 1 else 0.0


def tree_cost(i, code, sizes, nlib, nbuiltin):
    """sizes: dict role -> size of that role's set in scope (role 'hole' = #holes or #formals).
    C1/C4/C5: leaf = log2(sum of all sizes)."""
    if code == 'C3':
        return c3_len(i) * LOG95
    p = profile(i)
    fb = lg(nbuiltin) * p[3]
    cb = lg(nlib) * p[4]
    if code in ('C1', 'C4', 'C5'):
        tot = sum(sizes.get(r, 0) for r in ROLES)
        return p[0] + sum(p[2]) * lg(tot) + fb + cb
    if code == 'C2':
        b = p[1] + fb + cb
        for j, r in enumerate(ROLES):
            if p[2][j]:
                b += p[2][j] * (FREQ.role_bits(r) + lg(sizes.get(r, 0)))
        return b
    raise ValueError(code)


# ---------------------------------------------------------------- C3 printing
_C3 = {}
_PRINT_SYM = {}


def print_symbol(label):
    s = _PRINT_SYM.get(label)
    if s is not None:
        return s
    tag = label[0]
    if tag == 's':
        s = label[1]
    else:
        # fixed 2-character names for temps / outputs / constants / holes
        idx = zlib.crc32(repr(label).encode()) % 62
        ch = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[idx]
        s = sp.Symbol({'t': 'T', 'o': 'O', 'c': 'C', 'd': 'Q', 'h': 'h'}[tag] + ch)
    _PRINT_SYM[label] = s
    return s


def call_print_name(label):
    if isinstance(label, tuple):          # miner template ('M', j)
        return 'Z' + '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[label[1] % 62]
    return C3NAME.get(label, label)


def to_sympy_print(i):
    kind, label, ch = NODES[i]
    if kind == 'Sym':
        return print_symbol(label)
    if kind == 'Int':
        return sp.Integer(label)
    if kind == 'Rat':
        return sp.Rational(*label)
    args = [to_sympy_print(c) for c in ch]
    if kind == 'Add':
        return sp.Add(*args, evaluate=False)
    if kind == 'Mul':
        return sp.Mul(*args, evaluate=False)
    if kind == 'Pow':
        return sp.Pow(args[0], args[1], evaluate=False)
    if kind == 'Func':
        if label in FUNC_BUILTIN:
            return FUNC_BUILTIN[label](*args, evaluate=False)
        return sp.Function(C3NAME.get(label, label))(*args)
    if kind == 'Call':
        return sp.Function(call_print_name(label))(*args)
    if kind == 'Rel':
        return sp.StrictGreaterThan(args[0], args[1], evaluate=False)
    raise ValueError(kind)


def c3_len(i):
    r = _C3.get(i)
    if r is None:
        r = len(sp.sstr(to_sympy_print(i)).replace(' ', ''))
        _C3[i] = r
    return r


# ---------------------------------------------------------------- substitution / traversal
def subst(i, mapping, memo=None):
    """Replace leaf ids per mapping {leaf_id: tree_id}."""
    if memo is None:
        memo = {}
    if i in mapping:
        return mapping[i]
    r = memo.get(i)
    if r is not None:
        return r
    kind, label, ch = NODES[i]
    if not ch:
        r = i
    else:
        nch = tuple(subst(c, mapping, memo) for c in ch)
        r = i if nch == ch else mk(kind, label, nch)
    memo[i] = r
    return r


def leaves(i, acc=None):
    if acc is None:
        acc = set()
    stack = [i]
    seen = set()
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        kind, label, ch = NODES[j]
        if kind == 'Sym':
            acc.add(label)
        stack.extend(ch)
    return acc


def calls_in(i, acc=None):
    if acc is None:
        acc = set()
    stack = [i]
    seen = set()
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        kind, label, ch = NODES[j]
        if kind == 'Call':
            acc.add(label)
        stack.extend(ch)
    return acc


def count_label(i, lab_id, memo=None):
    """number of occurrences of leaf id lab_id in tree i (with multiplicity)."""
    if memo is None:
        memo = {}
    if i == lab_id:
        return 1
    r = memo.get(i)
    if r is not None:
        return r
    r = sum(count_label(c, lab_id, memo) for c in NODES[i][2])
    memo[i] = r
    return r

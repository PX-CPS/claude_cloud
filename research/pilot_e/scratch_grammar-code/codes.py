"""Prefix codes over sympy expression trees (scratch for candidate E).

A representation of one system is a list of items (name, expr):
  name is None  -> an output expression (coded as-is)
  name is Symbol -> a definition (temp); the name joins the leaf scope
Leaf scope for a system = state symbols + parameter symbols + temps.

Codes
  C1  uniform: node kind uniform over KINDS, n-ary arity Elias-gamma, leaf log2|scope|,
      call name log2|library|, integers Elias-gamma.  CSE on.
  C2  static frequency: kind / arity / leaf-role probabilities fitted on a reference
      corpus (the all-explicit baseline), Laplace-smoothed.  CSE on.
  C3  plain-text characters (sympy sstr, no spaces) * log2(95).  CSE on.
  C4  numeric-lumped: maximal state-free parameter subtrees -> one real constant,
      B bits per distinct constant per system, universal constants (G) once per corpus,
      tree otherwise coded as C1.  CSE on.
  C5  C1 with CSE off (probe of baseline-mechanism dependence).
"""
import math
import sympy as sp
from sympy.core.function import AppliedUndef

KINDS = ['Add', 'Mul', 'Pow', 'Func', 'Call', 'Sym', 'Int', 'Rat']
BUILTIN = {'sin', 'cos', 'tan', 'cot', 'exp', 'log'}
LOG95 = math.log2(95)
B_BITS = 16.0


def gamma_len(n):
    n = int(n)
    assert n >= 1
    return 2 * int(math.floor(math.log2(n))) + 1


def int_cost(v):
    v = int(v)
    return gamma_len(abs(v) + 1) + 1


def walk(e):
    """Preorder token list: tuples (kind, info)."""
    out = []

    def rec(x):
        if isinstance(x, sp.Add):
            out.append(('Add', len(x.args)))
            for a in x.args:
                rec(a)
        elif isinstance(x, sp.Mul):
            out.append(('Mul', len(x.args)))
            for a in x.args:
                rec(a)
        elif isinstance(x, sp.Pow):
            out.append(('Pow', None))
            rec(x.base)
            rec(x.exp)
        elif isinstance(x, sp.Symbol):
            out.append(('Sym', x))
        elif isinstance(x, sp.Integer):
            out.append(('Int', int(x)))
        elif isinstance(x, sp.Rational):
            out.append(('Rat', (int(x.p), int(x.q))))
        elif isinstance(x, sp.Float):
            r = sp.Rational(str(x))
            out.append(('Rat', (int(r.p), int(r.q))))
        elif isinstance(x, AppliedUndef):
            out.append(('Call', (x.func.__name__, len(x.args))))
            for a in x.args:
                rec(a)
        elif isinstance(x, sp.Function) and x.func.__name__ in BUILTIN:
            out.append(('Func', x.func.__name__))
            for a in x.args:
                rec(a)
        else:
            raise ValueError('unknown node %r (%s)' % (x, type(x)))
    rec(e)
    return out


# ---------------------------------------------------------------- C1
def c1_tree(e, nscope, nlib):
    bits = 0.0
    lk = math.log2(len(KINDS))
    for kind, info in walk(e):
        bits += lk
        if kind in ('Add', 'Mul'):
            bits += gamma_len(info - 1)
        elif kind == 'Sym':
            bits += math.log2(max(nscope, 1))
        elif kind == 'Int':
            bits += int_cost(info)
        elif kind == 'Rat':
            bits += int_cost(info[0]) + gamma_len(info[1])
        elif kind == 'Func':
            bits += math.log2(len(BUILTIN))
        elif kind == 'Call':
            bits += math.log2(max(nlib, 1))
    return bits


# ---------------------------------------------------------------- C2
class FreqModel:
    def __init__(self):
        self.kind = {k: 1.0 for k in KINDS}
        self.arity = {n: 1.0 for n in range(2, 12)}
        self.role = {'state': 1.0, 'param': 1.0, 'temp': 1.0}

    def fit(self, trees_with_roles):
        for e, roles in trees_with_roles:
            for kind, info in walk(e):
                self.kind[kind] += 1
                if kind in ('Add', 'Mul'):
                    self.arity[min(info, 11)] += 1
                if kind == 'Sym':
                    self.role[roles.get(info, 'param')] += 1
        self._norm()

    def _norm(self):
        for d in (self.kind, self.arity, self.role):
            s = sum(d.values())
            for k in d:
                d[k] = d[k] / s

    def tree(self, e, roles, role_sizes, nlib):
        bits = 0.0
        for kind, info in walk(e):
            bits += -math.log2(self.kind[kind])
            if kind in ('Add', 'Mul'):
                bits += -math.log2(self.arity[min(info, 11)])
            elif kind == 'Sym':
                r = roles.get(info, 'param')
                bits += -math.log2(self.role[r]) + math.log2(max(role_sizes.get(r, 1), 1))
            elif kind == 'Int':
                bits += int_cost(info)
            elif kind == 'Rat':
                bits += int_cost(info[0]) + gamma_len(info[1])
            elif kind == 'Func':
                bits += math.log2(len(BUILTIN))
            elif kind == 'Call':
                bits += math.log2(max(nlib, 1))
        return bits


FREQ = FreqModel()


# ---------------------------------------------------------------- C3
def c3_tree(e):
    return len(sp.sstr(e).replace(' ', '')) * LOG95


# ---------------------------------------------------------------- lumping for C4
UNIVERSAL = {sp.Symbol('G'), sp.Symbol('ke')}


def lump(e, statelike, table):
    """Replace maximal state-free parameter-bearing subtrees by constant symbols.
    table: dict expr -> Symbol (per system)."""
    def is_free(x):
        return not (x.free_symbols & statelike) and not x.has(AppliedUndef) and len(x.free_symbols) > 0

    def const_of(x):
        if x not in table:
            table[x] = sp.Symbol('c%d' % len(table))
        return table[x]

    def rec(x):
        if isinstance(x, (sp.Symbol,)) or x.is_Number:
            if is_free(x):
                return const_of(x)
            return x
        if is_free(x):
            return const_of(x)
        if isinstance(x, (sp.Add, sp.Mul)):
            free = [a for a in x.args if not (a.free_symbols & statelike) and not a.has(AppliedUndef)]
            rest = [a for a in x.args if a not in free]
            if any(len(a.free_symbols) > 0 for a in free):
                grp = x.func(*free)
                c = const_of(grp)
                return x.func(c, *[rec(a) for a in rest], evaluate=False)
            return x.func(*[rec(a) for a in x.args], evaluate=False)
        return x.func(*[rec(a) for a in x.args], evaluate=False)
    return rec(e)


# ---------------------------------------------------------------- CSE with fair inlining
def apply_cse(items, cost_fn):
    """items: list of (name|None, expr).  Run sympy CSE on all expressions, then inline
    any temp whose definition does not pay for itself (greedy, reverse order)."""
    names = [n for n, _ in items]
    exprs = [e for _, e in items]
    gen = sp.numbered_symbols('t', cls=sp.Symbol)
    repl, red = sp.cse(exprs, symbols=gen, optimizations=None)
    new = [(s, d) for s, d in repl] + list(zip(names, red))
    # greedy inline pass
    changed = True
    while changed:
        changed = False
        base = cost_fn(new)
        for idx in range(len(new) - 1, -1, -1):
            s, d = new[idx]
            if s is None:
                continue
            trial = [(n2, e2.xreplace({s: d})) for j, (n2, e2) in enumerate(new) if j != idx]
            c = cost_fn(trial)
            if c <= base:
                new = trial
                base = c
                changed = True
                break
    return new

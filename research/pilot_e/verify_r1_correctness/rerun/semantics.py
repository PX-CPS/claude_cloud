"""Decoders for every formulation + numeric semantic check (U4, U5).

Error metric (U5): normwise relative error  max_i |a_rep_i - a_true_i| / max(max_i |a_true_i|, 1e-12)
at 5 random points (all state and parameter symbols log-uniform on [0.5, 2]).
"""
import math, random, zlib
import numpy as np
import sympy as sp
import corpus as CP
import trees as TR
from common import SEED

NP_FUN = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'cot': lambda x: 1 / np.tan(x), 'exp': np.exp, 'log': np.log}
SP_FUN = {'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'cot': sp.cot, 'exp': sp.exp, 'log': sp.log}


def _R(a, b):
    return sp.sqrt(a ** 2 + b ** 2)


def _vecm(a, b):
    return sp.Matrix([a, b])


LAW_SYM = {
    'V': lambda a, b: _vecm(a, b),
    'Cn': lambda s, dx, dy: _vecm(s * dx, s * dy),
    'D': lambda a, b: _R(a, b),
    'K': lambda m, w: m * w / 2,
    'Fh1': lambda k, a, b: -k * (a - b),
    'Vh1': lambda k, a, b: k * (a - b) ** 2 / 2,
    'Fh2': lambda k, l, xi, yi, xj, yj: _vecm(-k * (_R(xi - xj, yi - yj) - l) * (xi - xj) / _R(xi - xj, yi - yj),
                                              -k * (_R(xi - xj, yi - yj) - l) * (yi - yj) / _R(xi - xj, yi - yj)),
    'Vh2': lambda k, l, xi, yi, xj, yj: k * (_R(xi - xj, yi - yj) - l) ** 2 / 2,
    'Fg': lambda G, mi, mj, xi, yi, xj, yj: _vecm(-G * mi * mj * (xi - xj) / _R(xi - xj, yi - yj) ** 3,
                                                  -G * mi * mj * (yi - yj) / _R(xi - xj, yi - yj) ** 3),
    'Vg': lambda G, mi, mj, xi, yi, xj, yj: -G * mi * mj / _R(xi - xj, yi - yj),
}


def _hyp(a, b):
    return math.sqrt(a * a + b * b)


LAW_NUM = {
    'V': lambda a, b: np.array([a, b], dtype=float),
    'Cn': lambda s, dx, dy: s * np.array([dx, dy], dtype=float),
    'D': lambda a, b: _hyp(a, b),
    'K': lambda m, w: m * w / 2,
    'Fh1': lambda k, a, b: -k * (a - b),
    'Vh1': lambda k, a, b: k * (a - b) ** 2 / 2,
    'Fh2': lambda k, l, xi, yi, xj, yj: -k * (_hyp(xi - xj, yi - yj) - l) / _hyp(xi - xj, yi - yj) * np.array([xi - xj, yi - yj]),
    'Vh2': lambda k, l, xi, yi, xj, yj: k * (_hyp(xi - xj, yi - yj) - l) ** 2 / 2,
    'Fg': lambda G, mi, mj, xi, yi, xj, yj: -G * mi * mj / _hyp(xi - xj, yi - yj) ** 3 * np.array([xi - xj, yi - yj]),
    'Vg': lambda G, mi, mj, xi, yi, xj, yj: -G * mi * mj / _hyp(xi - xj, yi - yj),
}


def has_diff(rep, templates):
    """True if the rep (or a template it uses, transitively) contains Dq/Dt."""
    seen = set()
    stack = [tid for _, tid in rep.items]
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        kind, label, ch = TR.NODES[j]
        if kind == 'Func' and label in ('Dq', 'Dt'):
            return True
        if kind == 'Call' and isinstance(label, tuple) and label in templates:
            stack.append(templates[label][1])
        stack.extend(ch)
    return False


class Decoder:
    def __init__(self, sysm, truth, npoints=5):
        self.sysm = sysm
        self.syms = list(sysm.state) + list(sysm.params)
        rng = random.Random(SEED + zlib.crc32(sysm.sid.encode()))
        self.points = [{s: math.exp(rng.uniform(math.log(0.5), math.log(2.0))) for s in self.syms}
                       for _ in range(npoints)]
        tr = truth[sysm.sid]
        if tr.get('acc') is not None:
            f = sp.lambdify(self.syms, tr['acc'], 'math')
            self.true = [np.array(f(*[p[s] for s in self.syms]), dtype=float) for p in self.points]
        else:
            # scaling-series fallback (no symbolic inverse): solve M(q) z = f(q, v) numerically
            fM = sp.lambdify(self.syms, tr['M'], 'math')
            ff = sp.lambdify(self.syms, tr['f'], 'math')
            self.true = [np.linalg.solve(np.array(fM(*[p[s] for s in self.syms]), dtype=float),
                                         np.array(ff(*[p[s] for s in self.syms]), dtype=float)) for p in self.points]
        self._lam = {}
        if sysm.fam == 'G':
            self.q, self.v, self.z = sysm.q, sysm.v, sysm.z
        else:
            self.q, self.v, self.z = sysm.coords, sysm.vels, []

    # ---------------------------------------------------------------- symbolic expansion
    def to_sym(self, i, symenv, templates):
        memo = {}
        alive = [symenv]

        def rec(j, env):
            key = (j, id(env))
            if key in memo:
                return memo[key]
            kind, label, ch = TR.NODES[j]
            if kind == 'Sym':
                if label in env:
                    r = env[label]
                elif label[0] == 's':
                    r = label[1]
                else:
                    raise KeyError(label)
            elif kind == 'Int':
                r = sp.Integer(label)
            elif kind == 'Rat':
                r = sp.Rational(*label)
            elif kind == 'Add':
                r = None
                for c in ch:
                    x = rec(c, env)
                    r = x if r is None else r + x
            elif kind == 'Mul':
                r = None
                for c in ch:
                    x = rec(c, env)
                    if r is None:
                        r = x
                    elif isinstance(r, sp.MatrixBase) and not isinstance(x, sp.MatrixBase):
                        r = x * r
                    else:
                        r = r * x
            elif kind == 'Pow':
                r = rec(ch[0], env) ** rec(ch[1], env)
            elif kind == 'Func' and label == 'Map':
                # R7: 'for each coordinate i' -- bound variables (q^, v^) -> (q_i, v_i); returns a list
                r = []
                for qi, vi in zip(self.q, self.v):
                    e2 = dict(env)
                    e2[('b', self.sysm.sid, 'q')] = qi
                    e2[('b', self.sysm.sid, 'v')] = vi
                    alive.append(e2)
                    r.append(rec(ch[0], e2))
            elif kind == 'Func':
                args = [rec(c, env) for c in ch]
                if label in SP_FUN:
                    r = SP_FUN[label](*args)
                elif label == 'Dq':
                    r = sp.diff(args[0], args[1])
                elif label == 'Dt':
                    e = args[0]
                    r = sum(sp.diff(e, q) * v for q, v in zip(self.q, self.v))
                    if self.z:
                        r += sum(sp.diff(e, v) * z for v, z in zip(self.v, self.z))
                    r += sp.diff(e, CP.t)
                else:
                    raise ValueError(label)
            elif kind == 'Call':
                args = [rec(c, env) for c in ch]
                if isinstance(label, tuple):
                    nh, body = templates[label]
                    henv = dict(symenv)
                    henv.update({('h', h): args[h] for h in range(nh)})
                    alive.append(henv)
                    r = rec(body, henv)
                else:
                    r = LAW_SYM[label](*args)
            else:
                raise ValueError(kind)
            memo[key] = r
            return r
        return rec(i, symenv)

    def sym_env(self, rep, templates):
        env = {}
        for lab, v in rep.consts.items():
            env[lab] = v
        oi = 0
        for lab, tid in rep.items:
            val = self.to_sym(tid, env, templates)
            if lab is not None:
                env[lab] = val
            else:
                env[('o', rep.sid, oi)] = val
                oi += 1
        return env

    # ---------------------------------------------------------------- numeric evaluation
    def _lambda(self, expr, extra=()):
        key = (expr, tuple(extra))
        f = self._lam.get(key)
        if f is None:
            f = sp.lambdify(self.syms + list(extra), expr, 'math')
            self._lam[key] = f
        return f

    def num_eval(self, i, env, templates, point, symenv_fn):
        memo = {}
        alive = [env]

        def rec(j, env_):
            key = (j, id(env_))
            if key in memo:
                return memo[key]
            kind, label, ch = TR.NODES[j]
            if kind == 'Sym':
                r = env_[label] if label in env_ else point[label[1]]
            elif kind == 'Int':
                r = float(label)
            elif kind == 'Rat':
                r = label[0] / label[1]
            elif kind == 'Add':
                r = 0.0
                for c in ch:
                    r = r + rec(c, env_)
            elif kind == 'Mul':
                r = 1.0
                for c in ch:
                    r = r * rec(c, env_)
            elif kind == 'Pow':
                r = rec(ch[0], env_) ** rec(ch[1], env_)
            elif kind == 'Func':
                if label in NP_FUN:
                    r = NP_FUN[label](rec(ch[0], env_))
                else:   # Dq / Dt: symbolic route
                    se = symenv_fn()
                    s = self.to_sym(j, se, templates)
                    r = float(self._lambda(s)(*[point[x] for x in self.syms]))
            elif kind == 'Call':
                args = [rec(c, env_) for c in ch]
                if isinstance(label, tuple):
                    nh, body = templates[label]
                    henv = dict(env)
                    henv.update({('h', h): args[h] for h in range(nh)})
                    alive.append(henv)
                    r = rec(body, henv)
                else:
                    r = LAW_NUM[label](*args)
            else:
                raise ValueError(kind)
            memo[key] = r
            return r
        return rec(i, env)

    # ---------------------------------------------------------------- decoders
    def decode(self, rep, templates, point):
        s = self.sysm
        form = rep.meta['form']
        symenv_cache = []

        def symenv_fn():
            if not symenv_cache:
                symenv_cache.append(self.sym_env(rep, templates))
            return symenv_cache[0]

        if form == 'el':
            L = self.to_sym(rep.items[-1][1], symenv_fn(), templates)
            return self._el(L, point)
        if form == 'implicit':
            se = symenv_fn()
            o0 = se[('o', rep.sid, 0)]
            if isinstance(o0, list):          # D2M Map output: one item gives all n residuals
                rs = [sp.sympify(x) for x in o0]
            else:
                rs = [se[('o', rep.sid, j)] for j in range(s.n)]
            return self._implicit(rs, point)
        if has_diff(rep, templates):
            se = symenv_fn()
            outs = []
            args = [point[x] for x in self.syms]
            for j in range(sum(1 for l, _ in rep.items if l is None)):
                e = se[('o', rep.sid, j)]
                if isinstance(e, sp.MatrixBase):
                    outs.append(np.array([float(self._lambda(c)(*args)) for c in e]))
                else:
                    outs.append(float(self._lambda(sp.sympify(e))(*args)))
        else:
            env = {}
            for lab, v in rep.consts.items():
                env[lab] = float(self._lambda(v)(*[point[x] for x in self.syms]))
            outs = []
            for lab, tid in rep.items:
                val = self.num_eval(tid, env, templates, point, symenv_fn)
                if lab is not None:
                    env[lab] = val
                else:
                    env[('o', rep.sid, len(outs))] = val
                    outs.append(val)
        if form == 'explicit':
            return np.array([float(o) for o in outs])
        # newton
        n2, pc = rep.meta['n2'], rep.meta['pc']
        acc = []
        for bi, o in enumerate(outs):
            m = point[s.masses[bi]]
            a = np.atleast_1d(np.asarray(o, dtype=float)) * np.ones(s.d)
            acc.append(a / m if n2 else a)
        if pc:
            ms = [point[m] for m in s.masses]
            last = -sum(ms[j] * acc[j] for j in range(s.n - 1)) / ms[-1]
            acc.append(last)
        return np.concatenate(acc)

    def _el(self, L, point):
        key = ('EL', L)
        f = self._lam.get(key)
        if f is None:
            n = len(self.q)
            Mm = [[sp.diff(L, self.v[i], self.v[j]) for j in range(n)] for i in range(n)]
            rhs = [sp.diff(L, self.q[i]) - sum(sp.diff(L, self.v[i], self.q[j]) * self.v[j] for j in range(n))
                   - sp.diff(L, self.v[i], CP.t) for i in range(n)]
            f = (sp.lambdify(self.syms, Mm, 'math'), sp.lambdify(self.syms, rhs, 'math'))
            self._lam[key] = f
        args = [point[x] for x in self.syms]
        return np.linalg.solve(np.array(f[0](*args), dtype=float), np.array(f[1](*args), dtype=float))

    def _implicit(self, rs, point):
        key = ('IMP', tuple(rs))
        f = self._lam.get(key)
        if f is None:
            n = len(self.q)
            zero = {z: 0 for z in self.z}
            Mm = [[sp.diff(rs[i], self.z[j]) for j in range(n)] for i in range(n)]
            fv = [-rs[i].subs(zero) for i in range(n)]
            f = (sp.lambdify(self.syms, Mm, 'math'), sp.lambdify(self.syms, fv, 'math'))
            self._lam[key] = f
        args = [point[x] for x in self.syms]
        return np.linalg.solve(np.array(f[0](*args), dtype=float), np.array(f[1](*args), dtype=float))

    def check(self, rep, templates=None):
        templates = templates or {}
        worst = 0.0
        for p, tv in zip(self.points, self.true):
            a = self.decode(rep, templates, p)
            err = float(np.max(np.abs(a - tv))) / max(float(np.max(np.abs(tv))), 1e-12)
            worst = max(worst, err)
        return worst

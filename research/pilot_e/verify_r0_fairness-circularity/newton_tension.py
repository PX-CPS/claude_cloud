"""Fairness probe F4: an EC-free textbook Newtonian formulation for family-G systems
(Cartesian bob kinematics in generalized coordinates + rod tensions as extra unknowns,
solved by the D1 linear-solve primitive that the pilot grants to everyone).
Uses only N2 (masses), N3 (equal/opposite rod tensions) and ideal rigid rods; no energy,
no Lagrangian, no differentiation primitive.  Costs computed with the pilot's own pipeline."""
import os, sys, math, json, random
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1)
os.chdir(PV1)
import numpy as np
import sympy as sp
import reps as R
import corpus as CP
import trees as TR
from trees import V
import lattice as LT

truth = CP.build_truth()
C = CP.by_id()


class Proxy:
    fam = 'G'

    def __init__(self, s, lam):
        self.s, self.lam = s, lam
        self.sid, self.n, self.z, self.q, self.v = s.sid, s.n, s.z, s.q, s.v
        self.params = s.params
        self.state = list(s.q) + list(s.v) + list(lam)   # lam = extra unknowns (scope +n_lam)
        self.applicable = s.applicable


def chain_items(s, ms, ls):
    n = s.n
    lam = list(sp.symbols('z%d:%d' % (n + 1, 2 * n + 1), real=True))
    e = [(sp.sin(q), -sp.cos(q)) for q in s.q]
    nn = [(sp.cos(q), sp.sin(q)) for q in s.q]
    outs = []
    ax = ay = 0
    for i in range(n):
        ax += ls[i] * (s.z[i] * nn[i][0] - s.v[i] ** 2 * e[i][0])
        ay += ls[i] * (s.z[i] * nn[i][1] - s.v[i] ** 2 * e[i][1])
        rx = ms[i] * ax + lam[i] * e[i][0]
        ry = ms[i] * ay + ms[i] * CP.g + lam[i] * e[i][1]
        if i + 1 < n:
            rx -= lam[i + 1] * e[i + 1][0]
            ry -= lam[i + 1] * e[i + 1][1]
        outs.append(V(rx, ry))
    return lam, outs


def chain_items_temps(s, ms, ls):
    n = s.n
    lam = list(sp.symbols('z%d:%d' % (n + 1, 2 * n + 1), real=True))
    e = [(sp.sin(q), -sp.cos(q)) for q in s.q]
    nn = [(sp.cos(q), sp.sin(q)) for q in s.q]
    temps, outs = [], []
    ax = ay = 0
    for i in range(n):
        tx, ty = sp.Symbol('_Rax%d' % i), sp.Symbol('_Ray%d' % i)
        temps += [(tx, ax + ls[i] * (s.z[i] * nn[i][0] - s.v[i] ** 2 * e[i][0])),
                  (ty, ay + ls[i] * (s.z[i] * nn[i][1] - s.v[i] ** 2 * e[i][1]))]
        ax, ay = tx, ty
        rx = ms[i] * ax + lam[i] * e[i][0]
        ry = ms[i] * ay + ms[i] * CP.g + lam[i] * e[i][1]
        if i + 1 < n:
            rx -= lam[i + 1] * e[i + 1][0]
            ry -= lam[i + 1] * e[i + 1][1]
        outs.append(V(rx, ry))
    return lam, temps, outs


def cart_items(s, M, m, l):
    # q1 = cart x, q2 = angle; bob at (x + l sin q2, -l cos q2); rod tension lam
    lam = [sp.Symbol('z3', real=True)]
    x2, th = s.q
    z1, z2 = s.z
    v2 = s.v[1]
    e = (sp.sin(th), -sp.cos(th))
    abx = z1 + l * (z2 * sp.cos(th) - v2 ** 2 * sp.sin(th))
    aby = l * (z2 * sp.sin(th) + v2 ** 2 * sp.cos(th))
    # cart (horizontal only; rail normal force eliminated): M z1 = lam*sin(th)  (rod pulls cart toward bob)
    r1 = M * z1 - lam[0] * e[0]
    rb = V(m * abx + lam[0] * e[0], m * aby + m * CP.g + lam[0] * e[1])
    return lam, [r1, rb]


def atwood_items(s, m1, m2):
    lam = [sp.Symbol('z2', real=True)]
    z = s.z[0]
    # q = descent of m1; m1 z = m1 g - T ; m2 z = T - m2 g
    return lam, [m1 * z - m1 * CP.g + lam[0], m2 * z + m2 * CP.g - lam[0]]


def pendulum1(s, m, l):
    return chain_items(s, [m], [l])


def numeric_check(s, lam, outs):
    # flatten residuals, solve linear system in (z, lam), compare with truth
    res = []
    for o in outs:
        if isinstance(o, sp.Function) or (hasattr(o, 'func') and o.func == V):
            res += list(o.args)
        else:
            res.append(o)
    unk = list(s.z) + list(lam)
    Mx, b = sp.linear_eq_to_matrix(res, unk)
    syms = list(s.state) + list(s.params)
    rng = random.Random(7)
    worst = 0.0
    fM = sp.lambdify(syms, Mx, 'numpy'); fb = sp.lambdify(syms, b, 'numpy')
    ft = sp.lambdify(syms, truth[s.sid]['acc'], 'numpy')
    for _ in range(5):
        pt = [math.exp(rng.uniform(math.log(.5), math.log(2))) for _ in syms]
        A = np.array(fM(*pt), float); bb = np.array(fb(*pt), float).ravel()
        if A.shape[0] > A.shape[1]:
            sol = np.linalg.lstsq(A, bb, rcond=None)[0]
        else:
            sol = np.linalg.solve(A, bb)
        tr = np.array(ft(*pt), float).ravel()
        worst = max(worst, np.max(np.abs(sol[:s.n] - tr)) / max(1e-12, np.max(np.abs(tr))))
    return worst, Mx.shape


def cost(s, lam, outs, code, level='D1'):
    p = Proxy(s, lam)
    items = [(None, o) for o in outs]
    rep = R.realize(p, items, dict(form='implicit', variant='tension'), frozenset(), level, code, 16)
    return rep.cost


def el_cost(s, code, level='D1'):
    rep = R.realize(s, *R.build_el_G(s, frozenset()), frozenset(['EC']), level, code, 16)
    return rep.cost


def implicit_cost(s, code, level='D1'):
    rep = R.realize(s, *R.build_implicit_G(s, truth), frozenset(), level, code, 16)
    return rep.cost


if __name__ == '__main__':
    out = {}
    for code in ['C1', 'C2', 'C3']:
        if code == 'C2':
            TR.FREQ.from_json(LT.fit_freq(truth, 'D1'))
        for sid in ['G01', 'G03', 'G04', 'G05', 'G07']:
            s = C[sid]
            P = {str(p): p for p in s.params}
            if sid == 'G01':
                lam, outs = pendulum1(s, P['m'], P['l'])
            elif sid == 'G03':
                lam, outs = chain_items(s, [P['m1'], P['m2']], [P['l1'], P['l2']])
            elif sid == 'G04':
                lam, outs = chain_items(s, [P['m1'], P['m2'], P['m3']], [P['l1'], P['l2'], P['l3']])
            elif sid == 'G05':
                lam, outs = atwood_items(s, P['m1'], P['m2'])
            else:
                lam, outs = cart_items(s, P['M'], P['m'], P['l'])
            err, shp = numeric_check(s, lam, outs)
            ct = cost(s, lam, outs, code)
            ce = el_cost(s, code)
            ci = implicit_cost(s, code)
            ex = R.realize(s, *R.build_explicit_G(s, truth), frozenset(), 'D1', code, 16).cost
            out[(code, sid)] = dict(tension=ct, el=ce, implicit=ci, explicit=ex, err=err)
            print(code, sid, 'sys', shp, 'relerr %.1e' % err, 'tension %.1f  EL %.1f  implicit-M %.1f  explicit %.1f'
                  % (ct, ce, ci, ex), flush=True)
    json.dump({'%s|%s' % k: v for k, v in out.items()},
              open('/home/user/claude_cloud/research/pilot_e/verify_r0_fairness-circularity/newton_tension.json', 'w'), indent=1)


def temps_variant():
    for code in ['C1', 'C3']:
        for sid in ['G03', 'G04']:
            s = C[sid]
            P = {str(p): p for p in s.params}
            ms = [P['m1'], P['m2']] + ([P['m3']] if sid == 'G04' else [])
            ls = [P['l1'], P['l2']] + ([P['l3']] if sid == 'G04' else [])
            lam, temps, outs = chain_items_temps(s, ms, ls)
            p = Proxy(s, lam)
            rep = R.realize(p, temps + [(None, o) for o in outs], dict(form='implicit', variant='tension-temps'),
                            frozenset(), 'D1', code, 16)
            print('temps-variant', code, sid, 'tension(recursive temps) %.1f' % rep.cost)

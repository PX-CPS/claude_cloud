"""Per-cell Track G degeneracy with per-law slopes (the abbreviation formula savings = u*(p-c) predicts
phi(f,k) = beta_f * u_fk with a law-specific per-use saving beta_f).  Leave-one-system-out CV-R2.
Targets: Shapley phi(f,k), per-system V_add(f,k) = L_k(empty) - L_k({f}), per-system V_loo(f,k)."""
import json, numpy as np
from load import *
TAGS = ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16', 'C1-D0-B16', 'C1-D2-B16']
CELLS = cells()
laws = [f for f in LAW9]
def cvr2(X, y, groups):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for g in sorted(set(groups)):
        te = np.array([gg == g for gg in groups])
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None); pred[te] = X[te] @ b
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def r2in(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def uEC(sid, f, dd):
    # pre-justified "uses" for EC: G family -> n^3 (Christoffel terms replaced); P family -> #pairs * n (H2: pair terms repeated in each eq)
    s = byid[sid]
    if f != 'EC':
        return dd['u']
    return s.dof ** 3 if s.fam == 'G' else dd['npair'] * dd['nbody']
MODELS = {
  'spec_M1 (lawFE + r,u,P,n)': lambda sid, f, dd: [1.0 * (f == g) for g in laws] + [dd['r'], dd['u'], dd['P'], dd['n']],
  'lawslope_u (9 params, no intercept)': lambda sid, f, dd: [dd['u'] * (f == g) for g in laws],
  'lawslope_r (9)': lambda sid, f, dd: [dd['r'] * (f == g) for g in laws],
  'lawslope_u_ECn3pairs (9)': lambda sid, f, dd: [uEC(sid, f, dd) * (f == g) for g in laws],
  'lawFE+lawslope_u_ECn3pairs (18)': lambda sid, f, dd: [1.0 * (f == g) for g in laws] + [uEC(sid, f, dd) * (f == g) for g in laws],
}
out = {}
for tag in TAGS:
    j = job(tag); ps = j['values']['phi_sys']; Lk = j['Lk']
    out[tag] = {}
    for target in ('phi', 'add', 'loo'):
        y, groups, rows = [], [], []
        for sid, f, dd in CELLS:
            L = np.array(Lk[sid]); b = bit(f)
            val = dict(phi=ps[sid][LAW9.index(f)], add=L[0] - L[b], loo=L[FULL ^ b] - L[FULL])[target]
            y.append(val); groups.append(sid); rows.append((sid, f, dd))
        out[tag][target] = {}
        for name, fx in MODELS.items():
            X = [fx(*r) for r in rows]
            out[tag][target][name] = dict(cv=cvr2(X, y, groups), r2=r2in(X, y))
        # "size" model: per-law slope on baseline length L_k(empty)
        X = [[Lk[sid][0] * (f == g) for g in laws] for sid, f, dd in rows]
        out[tag][target]['lawslope_Lk0 (9, not a count)'] = dict(cv=cvr2(X, y, groups), r2=r2in(X, y))
for tag in TAGS:
    print(tag)
    for target in out[tag]:
        for name, o in out[tag][target].items():
            print('   %-4s %-40s CV-R2=%7.3f  in=%.3f' % (target, name, o['cv'], o['r2']))
json.dump(out, open('percell_counts.json', 'w'), indent=1)

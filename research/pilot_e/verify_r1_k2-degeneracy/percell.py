"""K2-deg-G re-examined with simple, pre-justified count models (verify r1).

Rationale for the per-cell 'uses' model: abbreviation savings are u*(p-c), i.e. a LAW-SPECIFIC per-use
saving times a count of uses.  The spec M1 uses a COMMON slope for u across laws (plus law intercepts),
which cannot represent u*(p-c) with different p per law; it is therefore not the natural count model.
u_abbr is read off each law's WITH-encoding (reps.py), not tuned on phi:
  N2 nbody (one /m_i per body output) | N3 npair | PC iso*nbody (omitted linear combination)
  LC ncen | HK spr1, spr2 (1D and 2D templates differ in size) | UG ngrav | DIST ndist | KE nkin
  EC famG*gam (gam = n^2(n+1)/2 Christoffel-type terms) and famP*npair (pair terms stored once in V, H2)
Targets: per-system Shapley phi(f,k), V_add(f,k)=L_k(0)-L_k({f}), V_loo(f,k)=L_k(LAW9\\f)-L_k(LAW9).
Leave-one-SYSTEM-out CV-R2 (as spec 7.2)."""
import json
import numpy as np
from load import *

CELLS = cells()
laws = list(LAW9)
UA = {'N2': ['nbody'], 'N3': ['npair'], 'PC': ['isoN'], 'LC': ['ncen'], 'EC': ['ECG', 'ECP'],
      'HK': ['spr1', 'spr2'], 'UG': ['ngrav'], 'DIST': ['ndist'], 'KE': ['nkin']}
COLS = [(f, c) for f in laws for c in UA[f]]


def ua_val(f, c, dd):
    if c == 'isoN':
        return dd['iso'] * dd['nbody']
    if c == 'ECG':
        return dd['famG'] * dd['gam']
    if c == 'ECP':
        return (1 - dd['famG']) * dd['npair']
    return dd[c]


def X_ua(sid, f, dd, laws_on=None):
    return [ua_val(g, c, dd) if (f == g) else 0.0 for g, c in COLS]


MODELS = {
    'spec_M1 [lawFE + common r,u,P,n] (13)': lambda sid, f, dd: [1.0 * (f == g) for g in laws] + [dd['r'], dd['u'], dd['P'], dd['n']],
    'lawslope_u_spec (9)': lambda sid, f, dd: [dd['u'] * (f == g) for g in laws],
    'lawslope_r_spec (9)': lambda sid, f, dd: [dd['r'] * (f == g) for g in laws],
    'lawslope_u_abbr (11)': X_ua,
    'lawslope_u_abbr + lawFE (20)': lambda sid, f, dd: X_ua(sid, f, dd) + [1.0 * (f == g) for g in laws],
}
TAGS = ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16', 'C1-D0-B16', 'C1-D2-B16', 'C1-D2M-B16',
        'C4-D2-B16', 'C2-D2-B16', 'C3-D2-B16']
out = {}
for tag in TAGS:
    j = job(tag)
    ps = j['values']['phi_sys']
    Lk = j['Lk']
    out[tag] = {}
    for target in ('phi', 'add', 'loo'):
        y, groups, rows = [], [], []
        for sid, f, dd in CELLS:
            L = np.array(Lk[sid])
            b = bit(f)
            val = dict(phi=ps[sid][LAW9.index(f)], add=L[0] - L[b], loo=L[FULL ^ b] - L[FULL])[target]
            y.append(val); groups.append(sid); rows.append((sid, f, dd))
        o = {}
        for name, fx in MODELS.items():
            X = [fx(*r) for r in rows]
            r2, _ = r2in(X, y)
            o[name] = dict(cv=cvr2(X, y, groups), r2=r2)
        # size control (not a count): per-law slope on the system's baseline length
        X = [[Lk[sid][0] * (f == g) for g in laws] for sid, f, dd in rows]
        r2, _ = r2in(X, y)
        o['lawslope_Lk0 (9; size, not count)'] = dict(cv=cvr2(X, y, groups), r2=r2)
        # sub-populations: substitution laws only / EC only, with u_abbr
        for sub, keep in (('subst-laws only', lambda f: f != 'EC'), ('EC only', lambda f: f == 'EC')):
            idx = [i for i, r in enumerate(rows) if keep(r[1])]
            Xs = [X_ua(*rows[i]) for i in idx]
            # drop all-zero columns
            A = np.array(Xs); nz = np.abs(A).sum(0) > 0; A = A[:, nz]
            ys = [y[i] for i in idx]; gs = [groups[i] for i in idx]
            r2s, _ = r2in(A, ys)
            o['lawslope_u_abbr | ' + sub] = dict(cv=cvr2(A, ys, gs), r2=r2s, n=len(idx))
        out[tag][target] = o

for tag in TAGS:
    print(tag)
    for target in out[tag]:
        for name, o in out[tag][target].items():
            print('   %-4s %-45s CV-R2=%7.3f  in=%.3f' % (target, name, o['cv'], o['r2']))
json.dump(out, open('percell.json', 'w'), indent=1)

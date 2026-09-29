"""Within-law count degeneracy: for each law, V_add(f,k) and phi(f,k) regressed on ONE pre-justified count
(slope only, and slope+intercept), leave-one-system-out.  Counts are fixed a priori by the abbreviation
formula savings = u*(p-c): the number of places the law's construct is applied.
  N2: #bodies (the decoder applies /m_i once per body output);  N3: #pairs;  PC: #bodies-1 (length of the
  implicit argument list m_1 O_1 + ... + m_{N-1} O_{N-1});  LC: #central terms;  HK: #springs;
  UG: #gravity terms;  DIST: #distinct distances;  KE: #kinetic terms;
  EC: family G n^3 (Christoffel terms of f), family P #pairs*#bodies (each pair term repeated in each body eq.)."""
import json, numpy as np
from load import *
def cnt(s, f, dd):
    if f == 'N2': return dd['nbody'] if s.fam == 'P' else 0
    if f == 'N3': return dd['npair']
    if f == 'PC': return dd['nbody'] - 1
    if f == 'LC': return dd['ncen']
    if f == 'HK': return dd['nspr']
    if f == 'UG': return dd['ngrav']
    if f == 'DIST': return dd['u']
    if f == 'KE': return dd['u']
    if f == 'EC': return s.dof ** 3 if s.fam == 'G' else dd['npair'] * dd['nbody']
def loo(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i] @ b
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / sst) if sst > 1e-9 else float('nan')
out = {}
for tag in ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16', 'C1-D2-B16']:
    j = job(tag); ps = j['values']['phi_sys']; Lk = j['Lk']; out[tag] = {}
    for f in LAW9:
        rows = [(sid, dd) for sid, ff, dd in cells() if ff == f]
        x = [cnt(byid[sid], f, dd) for sid, dd in rows]
        res = {'n': len(rows)}
        for tgt in ('add', 'phi', 'loo'):
            y = []
            for sid, dd in rows:
                L = np.array(Lk[sid]); b = bit(f)
                y.append(dict(add=L[0] - L[b], loo=L[FULL ^ b] - L[FULL], phi=ps[sid][LAW9.index(f)])[tgt])
            res[tgt] = dict(slope=loo([[v] for v in x], y), slope_int=loo([[1, v] for v in x], y),
                            ys=[round(v, 1) for v in y])
        res['x'] = x
        out[tag][f] = res
        print('%s %-4s n=%2d  add: CV(slope)=%6.3f CV(a+b)=%6.3f | phi: %6.3f %6.3f | loo: %6.3f %6.3f' % (
            tag, f, len(rows), res['add']['slope'], res['add']['slope_int'], res['phi']['slope'], res['phi']['slope_int'],
            res['loo']['slope'], res['loo']['slope_int']))
json.dump(out, open('perlaw_counts.json', 'w'), indent=1)

"""H1, H2, 7.3 refit with an a-priori term count, Track R identity check (verify r1)."""
import json
import numpy as np
from load import *

a = json.load(open(os.path.join(RES, 'analysis.json')))
out = {}

# ---- 7.3 refit: a-priori Christoffel-type count gam = n^2(n+1)/2 (not n^3 picked after seeing data)
def loo(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i] @ b
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())

dup = set(a['EC_scaling']['duplicates']) if isinstance(a['EC_scaling']['duplicates'], (list, dict)) else set()
out['73'] = {}
for c, v in a['EC_scaling']['per_code'].items():
    pts = {k: p for k, p in v['points'].items() if k not in dup}
    ks = sorted(pts)
    y = [pts[k]['surplus_on'] for k in ks]
    G = [1.0 if pts[k]['fam'] == 'G' else 0.0 for k in ks]
    n = [pts[k]['n'] for k in ks]
    pairs = [pts[k]['pairs'] for k in ks]
    models = {
        'famP + famG*gam  (gam=n^2(n+1)/2, a priori)': [[1 - g, g * k * k * (k + 1) / 2] for g, k in zip(G, n)],
        'famP + famG*r_fk*n  (spec r_fk times n)': [[1 - g, g * k * k * (k - 1) / 2] for g, k in zip(G, n)],
        'famG*n  (linear)': [[1 - g, g * k] for g, k in zip(G, n)],
        'const only': [[1.0] for _ in ks],
    }
    out['73'][c] = {m: loo(X, y) for m, X in models.items()}
    out['73'][c]['famP_values'] = sorted(set(round(yy, 3) for yy, g in zip(y, G) if g == 0))
    out['73'][c]['famG_points'] = {k: (pts[k]['n'], round(pts[k]['surplus_on'], 1)) for k in ks if pts[k]['fam'] == 'G'}
    out['73'][c]['npts'] = len(ks)
    print('7.3', c, len(ks), {m: round(r, 3) for m, r in out['73'][c].items() if isinstance(r, float)},
          'famP distinct values', out['73'][c]['famP_values'])
print('   famG points C1', out['73']['C1']['famG_points'])

# ---- H2: EC per-system V_add and V_loo (C1..C4, D1): 1-DOF negative? growth with pairs/DOF?
out['H2'] = {}
for c in ['C1', 'C2', 'C3', 'C4']:
    j = job('%s-D1-B16' % c)
    rows = []
    for s in corp:
        if 'EC' not in s.applicable:
            continue
        L = np.array(j['Lk'][s.sid]); b = bit('EC')
        rows.append((s.sid, s.dof, (s.n_pairs() if s.fam == 'P' else 0), round(L[0] - L[b], 1), round(L[FULL ^ b] - L[FULL], 1),
                     round(j['values']['phi_sys'][s.sid][LAW9.index('EC')], 1)))
    out['H2'][c] = rows
    one = [r for r in rows if r[1] == 1]
    print('H2', c, '1-DOF (sid,dof,pairs,Vadd,Vloo,phi):', one)
    print('    >1-DOF:', [r for r in rows if r[1] > 1])

# ---- H1: N2 in C4 vs constant precision B (linear in B => pure count of tied constants)
out['H1'] = {}
for lev in ['D0', 'D1', 'D2']:
    ys = {}
    for B in (8, 16, 32):
        j = job('C4-%s-B%d' % (lev, B))
        ys[B] = j['values']['values']['N2']['Phi']
    slope1 = (ys[16] - ys[8]) / 8; slope2 = (ys[32] - ys[16]) / 16
    out['H1'][lev] = dict(Phi_N2=ys, dPhi_dB=[slope1, slope2])
    print('H1 C4 %s Phi_N2 by B' % lev, {k: round(v, 1) for k, v in ys.items()}, 'dPhi/dB = %.2f, %.2f' % (slope1, slope2))
print('H1 TrackR Mach', a['trackR']['mach'])

# ---- Track R: I_R1 / codim constant?
ratios = sorted(set(round(c['I_R1']['8'] / c['codim'], 6) for c in a['trackR']['cells'] if c['codim'] > 0))
out['trackR_R1_ratio'] = ratios
print('TrackR I_R1(b=8)/codim distinct values:', ratios, ' (b+0.5*log2(2pi) = %.6f)' % (8 + 0.5 * np.log2(2 * np.pi)))
for code in ('I_R2', 'I_R3'):
    r = [c[code] / c['codim'] for c in a['trackR']['cells'] if c['codim'] > 0]
    print('   %s/codim range %.2f..%.2f' % (code, min(r), max(r)))
r = [c['I_R4']['1000'] / c['codim'] for c in a['trackR']['cells'] if c['codim'] > 0]
print('   I_R4/codim range %.2f..%.2f' % (min(r), max(r)))
print('POS per-system bits', [(p['sid'], round(p.get('bits', float('nan')), 2)) for p in a['trackR']['pos'] if 'bits' in p])
json.dump(out, open('hazards.json', 'w'), indent=1, default=float)

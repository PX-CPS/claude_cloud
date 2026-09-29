"""How much do single corpus-selection decisions drive the law ranking?  Uses the additive per-system
decomposition Phi(f) = sum_k phi(f,k) + phi_ls(f) (miner off).  phi_ls is held fixed (approximation:
HK's statement depends on the templates used; recomputing it only moves HK)."""
import sys, json, itertools, numpy as np
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import analysis as AN
from common import LAW9
R = '/home/user/claude_cloud/research/pilot_e/pilot_v1/results/'
codes = ['C1', 'C2', 'C3', 'C4']
J = {c: json.load(open(R + 'job_%s-D1-B16.json' % c)) for c in codes}
def Phi(c, drop=(), dup=(), charged=False):
    v = J[c]['values_charged' if charged else 'values']
    ps, ls = v['phi_sys'], np.array(v['phi_ls'])
    tot = ls.copy()
    for sid, p in ps.items():
        if sid in drop: continue
        tot += np.array(p) * (2 if sid in dup else 1)
    return dict(zip(LAW9, tot))
for label, kw in [('full', {}), ('drop P11', dict(drop=('P11',))), ('drop G04', dict(drop=('G04',))),
                  ('drop P11,G04', dict(drop=('P11', 'G04'))), ('drop P10,P11', dict(drop=('P10', 'P11'))),
                  ('dup P11 (add a 2nd 5-body system)', dict(dup=('P11',))),
                  ('dup P06,P12 (spring systems)', dict(dup=('P06', 'P12'))),
                  ('drop family G', dict(drop=('G01','G02','G03','G04','G05','G06','G07')))]:
    for ch in (False, True):
        P = {c: Phi(c, charged=ch, **kw) for c in codes}
        k = AN.k1_test(P, codes)
        rk = {c: sorted(LAW9, key=lambda f: -P[c][f])[:3] for c in codes}
        print('%-38s %s min rho %.3f flips %d fired %s top3 %s' % (label, 'chg ' if ch else 'unch', k['min_rho'], len(k['flips']), k['fired'], rk))
# share of Phi from top-2 systems
for c in codes:
    ps = J[c]['values']['phi_sys']
    for fi, f in enumerate(LAW9):
        if f in ('EC', 'UG', 'N2'):
            vals = sorted((p[fi] for p in ps.values()), reverse=True)
            tot = sum(v for v in vals if v > 0)
            print(c, f, 'top-2 systems share of positive per-system value: %.2f' % (sum(vals[:2]) / tot))

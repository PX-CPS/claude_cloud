"""Fairness/circularity lens (r1): where does each law's Phi come from, and how much is fixed by corpus
selection?  Uses only saved per-system Shapley decompositions (miner off)."""
import sys, json, numpy as np
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1)
import analysis as AN, corpus as CP
from common import LAW9
R = PV1 + '/results/'
codes = ['C1', 'C2', 'C3', 'C4']
def load(c, lv='D1', extra=''):
    return json.load(open(R + 'job_%s-%s-B16%s.json' % (c, lv, extra)))
def Phi(J, drop=(), keep=None, charged=False, weight=None):
    v = J['values_charged' if charged else 'values']
    tot = np.array(v['phi_ls'], float)
    for sid, p in v['phi_sys'].items():
        if sid in drop or (keep is not None and sid not in keep): continue
        w = 1.0 if weight is None else weight.get(sid, 1.0)
        tot += w * np.array(p)
    return dict(zip(LAW9, tot))
corp = {s.sid: s for s in CP.corpus()}
famG = [k for k, s in corp.items() if s.fam == 'G']
out = {}
for lv in ['D1', 'D2', 'D2M']:
    J = {c: load(c, lv) for c in codes}
    print('==== level', lv)
    for c in codes:
        ps = J[c]['values']['phi_sys']; ls = J[c]['values']['phi_ls']
        for f in ['EC', 'UG', 'N2', 'PC', 'HK', 'DIST']:
            fi = LAW9.index(f)
            g = sum(ps[k][fi] for k in famG); p = sum(ps[k][fi] for k in ps if k not in famG)
            top = sorted(((ps[k][fi], k) for k in ps), reverse=True)[:3]
            print('%s %-4s ls %7.1f  famP %8.1f famG %8.1f  top3 %s' % (c, f, ls[fi], p, g, [(k, round(x, 1)) for x, k in top]))
    for label, kw in [('full', {}), ('drop famG', dict(drop=famG)), ('drop G04', dict(drop=('G04',))),
                      ('drop P11', dict(drop=('P11',))), ('drop P10,P11', dict(drop=('P10', 'P11'))),
                      ('famG only', dict(keep=famG))]:
        for ch in ([False, True] if lv == 'D1' else [False]):
            P = {c: Phi(J[c], charged=ch, **kw) for c in codes}
            k = AN.k1_test(P, codes)
            rk = {c: [(f, round(P[c][f])) for f in sorted(LAW9, key=lambda f: -P[c][f])[:3]] for c in codes}
            print('%-12s %s minrho %.3f fired %s top3 %s' % (label, 'chg ' if ch else 'unch', k['min_rho'], k['fired'], rk))
            out['%s|%s|%s' % (lv, label, ch)] = dict(min_rho=k['min_rho'], fired=k['fired'], top3=rk)
json.dump(out, open('/home/user/claude_cloud/research/pilot_e/verify_r1_fairness-circularity/decomp.json', 'w'), indent=1)

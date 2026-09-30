"""Law-level degeneracy (9 points): Phi_c(f), V_mid_c(f) vs single simple counts (+intercept), LOLO CV-R2,
Spearman.  Also: which single cells carry each law's value (concentration)."""
import json
import numpy as np
from scipy.stats import spearmanr
from load import *
from percell import ua_val, UA, CELLS

a = json.load(open(os.path.join(RES, 'analysis.json')))
TAGS = ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16', 'C1-D2-B16']
feat = {}
for f in LAW9:
    cs = [dd for sid, ff, dd in CELLS if ff == f]
    feat[f] = dict(K=len(cs), sum_u=sum(dd['u'] for dd in cs), sum_r=sum(dd['r'] for dd in cs),
                   sum_uabbr=sum(sum(ua_val(f, c, dd) for c in UA[f]) for dd in cs),
                   sum_n=sum(dd['n'] for dd in cs), sum_P=sum(dd['P'] for dd in cs))


def loo_1(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        A = np.c_[np.ones(m.sum()), x[m]]
        b, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        pred[i] = b[0] + b[1] * x[i]
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


out = {}
for tag in TAGS:
    j = job(tag)
    Phi = {f: j['values']['values'][f]['Phi'] for f in LAW9}
    Vmid = {f: j['values']['values'][f]['mid'] for f in LAW9}
    Ls = j['law_single']
    ps = j['values']['phi_sys']
    o = {}
    for tname, T in (('Phi', Phi), ('Vmid', Vmid)):
        y = [T[f] for f in LAW9]
        o[tname] = {}
        for fn in ('K', 'sum_u', 'sum_r', 'sum_uabbr', 'sum_n', 'sum_P'):
            x = [feat[f][fn] for f in LAW9]
            o[tname][fn] = dict(rho=float(spearmanr(x, y)[0]), loo_cv_r2=loo_1(x, y))
        # gross value (add back statement): sum_k phi(f,k)
        yg = [sum(ps[s][LAW9.index(f)] for s in ps) for f in LAW9]
        o[tname + '_gross_vs_sum_uabbr'] = dict(rho=float(spearmanr([feat[f]['sum_uabbr'] for f in LAW9], yg)[0]))
    # concentration: share of sum_k max(phi,0) in the top-2 systems
    conc = {}
    for f in LAW9:
        v = sorted([ps[s][LAW9.index(f)] for s in ps], reverse=True)
        pos = sum(x for x in v if x > 0)
        conc[f] = dict(top2_share=(v[0] + v[1]) / pos if pos > 0 else None,
                       top=[(s, round(ps[s][LAW9.index(f)], 1)) for s in sorted(ps, key=lambda s: -ps[s][LAW9.index(f)])[:3]])
    o['concentration'] = conc
    out[tag] = o
    print(tag)
    for tname in ('Phi', 'Vmid'):
        for fn, r in o[tname].items():
            print('   %-5s ~ %-10s rho=%6.3f  LOLO-CV-R2=%7.3f' % (tname, fn, r['rho'], r['loo_cv_r2']))
    for f in LAW9:
        print('   conc %-4s top2share=%s top=%s' % (f, None if conc[f]['top2_share'] is None else round(conc[f]['top2_share'], 2), conc[f]['top']))
print(json.dumps(feat))
json.dump(dict(features=feat, fits=out), open('lawlevel.json', 'w'), indent=1)

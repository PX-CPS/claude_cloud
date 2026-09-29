"""Refit the spec-7.3 EC surplus with simple, pre-justifiable count models that respect family structure.
Pre-justification: (a) family P: EC's surplus given all other laws is the formulation-flag (-1 bit) because
Newtonian templates (N3/LC/UG/HK) already store each pair term once (H2); (b) family G: the implicit baseline
stores M (n^2 entries) and f, whose Coriolis part has O(n^3) Christoffel terms; L stores O(n^2) kinetic terms.
So surplus ~ fam_G * poly(n).  We try only a few models, LOSO CV."""
import json, numpy as np
d = json.load(open('/home/user/claude_cloud/research/pilot_e/pilot_v1/results/analysis.json'))
def cv(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def ins(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), b
models = {
 'spec (1,n,pairs,P,famG)': lambda p: [1, p['n'], p['pairs'], p['P'], p['G']],
 'famP, famG*n^3 (2 params)': lambda p: [1 - p['G'], p['G'] * p['n'] ** 3],
 'famP, famG, famG*n^3': lambda p: [1 - p['G'], p['G'], p['G'] * p['n'] ** 3],
 'famP, famG*n^2, famG*n^3': lambda p: [1 - p['G'], p['G'] * p['n'] ** 2, p['G'] * p['n'] ** 3],
 'famP, famG*P^2 ': lambda p: [1 - p['G'], p['G'] * p['P'] ** 2],
 'famP, famG*n(n-1)/2 *n (r_fk * n)': lambda p: [1 - p['G'], p['G'] * p['n'] * p['n'] * (p['n'] - 1) / 2],
}
out = {}
for c, v in d['EC_scaling']['per_code'].items():
    pts = v['points']
    rows = [dict(sid=k, n=p['n'], pairs=p['pairs'], P=p['P'], G=1.0 if p['fam'] == 'G' else 0.0, y=p['surplus_on']) for k, p in pts.items()]
    y = [r['y'] for r in rows]
    out[c] = {}
    for name, fx in models.items():
        X = [fx(r) for r in rows]
        r2, b = ins(X, y)
        out[c][name] = dict(cv=cv(X, y), r2=r2, coef=[float(x) for x in b])
    out[c]['G_points'] = {r['sid']: (r['n'], round(r['y'], 1)) for r in rows if r['G']}
    out[c]['P_points_unique_y'] = sorted(set(round(r['y'], 3) for r in rows if not r['G']))
    out[c]['npts'] = len(rows)
for c in out:
    print(c, out[c]['npts'], out[c]['G_points'], out[c]['P_points_unique_y'])
    for name in models:
        o = out[c][name]; print('   %-40s CV-R2=%.4f in=%.4f coef=%s' % (name, o['cv'], o['r2'], np.round(o['coef'], 2)))
json.dump(out, open('ec_scaling_refit.json', 'w'), indent=1)

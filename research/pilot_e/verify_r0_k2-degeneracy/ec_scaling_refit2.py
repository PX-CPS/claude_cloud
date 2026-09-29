import json, numpy as np
d = json.load(open('/home/user/claude_cloud/research/pilot_e/pilot_v1/results/analysis.json'))
def cv(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i] @ b
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
res = {}
for c, v in d['EC_scaling']['per_code'].items():
    pts = {k: p for k, p in v['points'].items() if k not in ('SNL1', 'SNL3')}   # exact duplicates of G01 / G04
    rows = [(k, p['n'], 1.0 if p['fam'] == 'G' else 0.0, p['surplus_on']) for k, p in pts.items()]
    y = [r[3] for r in rows]
    X = [[1 - r[2], r[2] * r[1] ** 3] for r in rows]
    Xs = [[1, r[1], r[1] and 0 or 0, r[2]] for r in rows]
    g = [(k, n, yy) for k, n, G, yy in rows if G]
    yg = [x[2] for x in g]
    res[c] = dict(n_all=len(rows), cv_all_famP_famGn3=cv(X, y),
                  n_G=len(g), cv_G_n3=cv([[x[1] ** 3] for x in g], yg),
                  cv_G_1_n3=cv([[1, x[1] ** 3] for x in g], yg),
                  cv_G_rfk_times_n=cv([[x[1] ** 2 * (x[1] - 1) / 2] for x in g], yg),
                  famP_values=sorted(set(round(r[3], 2) for r in rows if not r[2])))
    print(c, {k: (round(v2, 4) if isinstance(v2, float) else v2) for k, v2 in res[c].items()})
json.dump(res, open('ec_scaling_refit2.json', 'w'), indent=1)

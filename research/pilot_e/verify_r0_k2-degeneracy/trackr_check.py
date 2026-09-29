"""Track R degeneracy: I_R1(b) vs codim*(b + 0.5*log2(2*pi)); and R2..R4 vs codim (LOO over cells), plus
dependence on a single count n (DOF) for EC cells (codim_EC = n(n-1)/2 + n^2 by the spec's equations)."""
import json, math, numpy as np
d = json.load(open('/home/user/claude_cloud/research/pilot_e/pilot_v1/results/analysis.json'))
cells = d['trackR']['cells']
def loo(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float); pred = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i] @ b
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res = {}
mx = max(abs(c['I_R1']['8'] - c['codim'] * (8 + 0.5 * math.log2(2 * math.pi))) for c in cells)
res['R1_b8_max_abs_dev_from_codim_formula'] = mx
for key, get in [('R1', lambda c: c['I_R1']['8']), ('R2', lambda c: c['I_R2']), ('R3', lambda c: c['I_R3']), ('R4', lambda c: c['I_R4']['1000'])]:
    y = [get(c) for c in cells]
    res[key] = dict(loo_codim_slope=loo([[c['codim']] for c in cells], y),
                    loo_nDOF_poly_EC_only=loo([[c['N'] * (c['N'] - 1) / 2 + c['N'] ** 2] for c in cells if c['law'] == 'EC'],
                                              [get(c) for c in cells if c['law'] == 'EC']))
print(json.dumps(res, indent=1))
print([(c['sid'], c['law'], c['N'], c['codim']) for c in cells])
json.dump(res, open('trackr_check.json', 'w'), indent=1)

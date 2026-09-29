"""Is Track-G phi(f,k) explained by the abbreviation formula u_fk*(p_f-c_f) (law-specific slope on uses)?"""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import corpus as CP, analysis as AN
from common import LAW9
R = '/home/user/claude_cloud/research/pilot_e/pilot_v1/results/'
corp = CP.corpus()
def extra_counts(s, f):
    # alternative natural 'use' counts: EC -> #potential terms + #kinetic terms (P) / #kin terms*dof (G)
    if f == 'EC':
        if s.fam == 'P':
            return len(s.potential_terms()) + s.n
        return s.n * len(s.kin_terms)
    return AN.counts(s, f)[1]
for tag in ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16', 'C1-D2-B16']:
    j = json.load(open(R + 'job_%s.json' % tag))
    ps = j['values']['phi_sys']
    rows = []
    for s in corp:
        for fi, f in enumerate(LAW9):
            if f in s.applicable:
                r, u = AN.counts(s, f)
                rows.append((s.sid, f, ps[s.sid][fi], u, r, extra_counts(s, f), s.dof, len(s.params)))
    y = np.array([r[2] for r in rows]); groups = [r[0] for r in rows]
    def design(kind):
        X = []
        for sid, f, _, u, r, u2, n, P in rows:
            v = []
            for g in LAW9:
                if kind == 'u': v += [u if f == g else 0]
                if kind == 'u2': v += [u2 if f == g else 0]
                if kind == 'u2+fe': v += [u2 if f == g else 0, 1 if f == g else 0]
                if kind == 'u2+n': v += [u2 if f == g else 0, n if f == g else 0]
            X.append(v)
        return np.array(X, float)
    out = {}
    for kind in ['u', 'u2', 'u2+fe', 'u2+n']:
        X = design(kind)
        out[kind] = (round(AN.cv_r2(X, y, groups), 3), round(AN.insample(X, y)[1], 3))
    print(tag, 'n_cells', len(rows), 'LOSO-CV R2 / in-sample R2:', out)

import sys, json, numpy as np
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import corpus as CP, analysis as AN
from common import LAW9
R = '/home/user/claude_cloud/research/pilot_e/pilot_v1/results/'
corp = CP.corpus()
def u2(s, f):
    if f == 'EC':
        return len(s.potential_terms()) + s.n if s.fam == 'P' else s.n * len(s.kin_terms)
    return AN.counts(s, f)[1]
for tag in ['C1-D1-B16', 'C2-D1-B16', 'C3-D1-B16', 'C4-D1-B16']:
    j = json.load(open(R + 'job_%s.json' % tag))
    Lk = j['Lk']
    rows = []
    for s in corp:
        for fi, f in enumerate(LAW9):
            if f in s.applicable:
                v = Lk[s.sid][0] - Lk[s.sid][1 << fi]
                rows.append((s.sid, f, v, u2(s, f)))
    y = np.array([r[2] for r in rows]); groups = [r[0] for r in rows]
    X = np.array([[r[3] if r[1] == g else 0 for g in LAW9] for r in rows], float)
    b, r2 = AN.insample(X, y)
    print(tag, 'add-one per-cell ~ law-specific slope*u : LOSO CV', round(AN.cv_r2(X, y, groups), 3), 'in', round(r2, 3),
          'slopes', dict(zip(LAW9, np.round(b, 1))))
    # per-law details for EC
    print('   EC cells', [(r[0], round(r[2]), r[3]) for r in rows if r[1] == 'EC'])

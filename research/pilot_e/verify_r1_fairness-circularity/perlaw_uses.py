"""Per-law: is phi(f,k) proportional to the number of uses the corpus designer put into system k?
Fit phi = a_f + b_f*u_fk on applicable cells per law (in-sample and LOSO), D1, miner off."""
import sys, json, numpy as np
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1)
import analysis as AN, corpus as CP
from common import LAW9
corp = CP.corpus()
out = {}
for c in ['C1', 'C2', 'C3', 'C4']:
    J = json.load(open(PV1 + '/results/job_%s-D1-B16.json' % c))
    ps = J['values']['phi_sys']
    for f in LAW9:
        fi = LAW9.index(f)
        rows = [(s.sid, AN.counts(s, f)[1], ps[s.sid][fi]) for s in corp if f in s.applicable]
        if f == 'EC':
            rows = [(sid, s.n_pairs() if s.fam == 'P' else 0, ps[sid][fi]) for (sid, _, _), s in zip(rows, [x for x in corp if 'EC' in x.applicable]) if s.fam == 'P']
        u = np.array([r[1] for r in rows], float); y = np.array([r[2] for r in rows])
        if len(set(u)) < 2: continue
        X = np.c_[np.ones_like(u), u]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # LOSO
        pred = np.zeros_like(y)
        for i in range(len(y)):
            msk = np.arange(len(y)) != i
            bb, *_ = np.linalg.lstsq(X[msk], y[msk], rcond=None); pred[i] = X[i] @ bb
        cv = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out['%s|%s' % (c, f)] = dict(n=len(y), slope=b[1], icpt=b[0], r2=r2, cv=cv)
        print('%s %-4s n=%2d  phi = %7.1f + %6.1f*u   R2 %.3f  LOSO-CV %.3f   %s' % (c, f + ('(famP,pairs)' if f == 'EC' else ''), len(y), b[0], b[1], r2, cv,
              [(r[0], r[1], round(r[2], 1)) for r in rows] if c == 'C1' and f in ('UG', 'HK', 'EC') else ''))
json.dump(out, open('/home/user/claude_cloud/research/pilot_e/verify_r1_fairness-circularity/perlaw_uses.json', 'w'), indent=1)

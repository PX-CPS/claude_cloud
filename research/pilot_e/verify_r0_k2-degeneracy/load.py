import sys, json, os
import numpy as np
PV = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV)
import corpus as CP
from common import LAW9
from analysis import counts, cv_r2, insample
RES = os.path.join(PV, 'results')
corp = CP.corpus()
byid = {s.sid: s for s in corp}
def job(tag):
    return json.load(open(os.path.join(RES, 'job_%s.json' % tag)))
FULL = 511
def bit(f): return 1 << LAW9.index(f)
def feats(s, f):
    r, u = counts(s, f)
    d = dict(r=r, u=u, n=s.dof, P=len(s.params), fam=1.0 if s.fam == 'G' else 0.0)
    if s.fam == 'P':
        d.update(npair=s.n_pairs(), ncen=s.n_central(), nspr=s.n_springs(), ngrav=s.n_grav(),
                 nforce=s.n_force_terms(), nbody=s.n, ninter=len(s.inter))
    else:
        d.update(npair=0, ncen=0, nspr=0, ngrav=0, nforce=0, nbody=s.n, ninter=0)
    return d
def cells():
    out = []
    for s in corp:
        for f in LAW9:
            if f in s.applicable:
                out.append((s.sid, f, feats(s, f)))
    return out

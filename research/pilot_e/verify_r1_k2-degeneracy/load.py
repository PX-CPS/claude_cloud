"""Read-only loaders for the pilot_v1 outputs (verify r1, K2 lens)."""
import sys, json, os
import numpy as np
PV = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV)
import corpus as CP
from common import LAW9
from analysis import counts
RES = os.path.join(PV, 'results')
corp = CP.corpus()
byid = {s.sid: s for s in corp}
FULL = 511


def job(tag):
    return json.load(open(os.path.join(RES, 'job_%s.json' % tag)))


def bit(f):
    return 1 << LAW9.index(f)


def feats(s, f):
    """Spec counts plus finer, a-priori 'uses' counts read off each law's WITH-encoding."""
    r, u = counts(s, f)
    G = s.fam == 'G'
    d = dict(r=r, u=u, n=s.dof, P=len(s.params), famG=1.0 if G else 0.0)
    if not G:
        spr1 = sum(1 for it in s.inter if it[0] == 'spring1')
        spr2 = sum(1 for it in s.inter if it[0] == 'spring2')
        d.update(npair=s.n_pairs(), ncen=s.n_central(), spr1=spr1, spr2=spr2, ngrav=s.n_grav(),
                 nbody=s.n, ndist=(s.n_central() if s.d == 2 else 0), nkin=s.n,
                 iso=1.0 if s.isolated else 0.0)
    else:
        d.update(npair=0, ncen=0, spr1=0, spr2=0, ngrav=0, nbody=s.n, ndist=0, nkin=len(s.kin_terms), iso=0.0)
    # Christoffel-type term count of a generalized-coordinate EOM: n equations x n(n+1)/2 velocity products
    d['gam'] = (s.n ** 2 * (s.n + 1) / 2.0) if G else 0.0
    return d


def cells():
    out = []
    for s in corp:
        for f in LAW9:
            if f in s.applicable:
                out.append((s.sid, f, feats(s, f)))
    return out


def cvr2(X, y, groups):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pred = np.zeros_like(y)
    for g in sorted(set(groups)):
        te = np.array([gg == g for gg in groups])
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
        pred[te] = X[te] @ b
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def r2in(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()), b

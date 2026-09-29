"""Fairness probe F6: 'applies to system k' is hand-assigned by physics labels.  If applicability
were decided by decodability (the law's template reproduces M_k exactly, checked by U5), UG's
inverse-square template also encodes Coulomb (P15): Fg(-ke, qA, qB, ...) / Vg(-ke, qA, qB, ...).
Recompute Phi with UG applicable to P15 under that encoding (C1, C3, C4 at D1)."""
import os, sys, json
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import sympy as sp
import reps as R, corpus as CP, lattice as LT
from trees import Fg, Vg
from common import LAW9

_tf, _pi = R.template_force, R.potential_items


def template_force(sysm, it, i, S):
    if it[0] == 'coul' and 'UG' in S:
        a, b, qa, qb = it[1:]
        j = b if i == a else a
        qi, qj = (qa, qb) if i == a else (qb, qa)
        return Fg(-CP.ke, qi, qj, *sysm.pos(i), *sysm.pos(j))
    return _tf(sysm, it, i, S)


def potential_items(sysm, S, cost_term):
    pots = _pi(sysm, S, cost_term)
    if 'UG' not in S:
        return pots
    out, k = [], 0
    cons = [it for it in sysm.inter if it[0] in CP.CONS]
    for it, p in zip(cons, pots):
        if it[0] == 'coul':
            a, b, qa, qb = it[1:]
            tm = Vg(-CP.ke, qa, qb, *sysm.pos(a), *sysm.pos(b))
            if cost_term(tm) < cost_term(p):
                p = tm
        out.append(p)
    return out


R.template_force = template_force
R.potential_items = potential_items
code = sys.argv[1]
for s in CP.corpus():
    if s.sid == 'P15':
        s.applicable = s.applicable | {'UG'}
j = LT.Job(code, 'D1', B=16, check=True, log=lambda *a: None).run()
v = j.values()
print(code, 'maxerr %.1e' % max(e[-1] for e in j.errors),
      {f: round(v['values'][f]['Phi'], 1) for f in LAW9}, 'UG phi on P15: %.1f' % v['phi_sys']['P15'][LAW9.index('UG')])
json.dump({f: v['values'][f]['Phi'] for f in LAW9}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_decodable_%s.json' % code), 'w'))

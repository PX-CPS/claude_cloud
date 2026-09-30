"""Fairness probe (r1 re-run of r0 F6): 'applies to system k' is a designer label.  P15 (three charges) is
labelled NOT UG-applicable although UG's own templates decode it exactly (Fg(-ke,q_i,q_j,..), Vg(-ke,..)).
Recompute Phi with UG applicable to P15 under that encoding, repaired pipeline, D1, and re-run the K1 test
against the saved values of the other codes.  Run: PYTHONHASHSEED=0 python3 ug_p15.py CODE"""
import os, sys, json
os.environ.setdefault('PYTHONHASHSEED', '0')
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import reps as R, corpus as CP, lattice as LT
from trees import Fg, Vg
from common import LAW9
OUT = os.path.dirname(os.path.abspath(__file__))
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
    out = []
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
vc = j.values(charged=True)
res = dict(code=code, maxerr=max(e[-1] for e in j.errors),
           Phi={f: v['values'][f]['Phi'] for f in LAW9}, Phi_charged={f: vc['values'][f]['Phi'] for f in LAW9},
           phi_sys=v['phi_sys'], phi_ls=v['phi_ls'], UG_phi_P15=v['phi_sys']['P15'][LAW9.index('UG')])
print(code, 'maxerr %.1e' % res['maxerr'], {f: round(x, 1) for f, x in res['Phi'].items()}, 'UG on P15 %.1f' % res['UG_phi_P15'])
json.dump(res, open(os.path.join(OUT, 'ug_p15_%s.json' % code), 'w'), indent=1)

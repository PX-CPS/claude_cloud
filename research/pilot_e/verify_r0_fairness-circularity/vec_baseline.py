"""Fairness probe F7: the base grammar already has Mul and the V(.,.) constructor, and the decoder
multiplies scalars by vectors.  So a central force can be written s*V(dx,dy) WITHOUT LC, which is
what Cn(s,dx,dy) abbreviates.  The pilot's no-LC baseline never offers this (explicit components are
merged into V(sum_x, sum_y)).  Offer s*V(dx,dy) as a separate summand to every theory (C1..C4, D1)."""
import os, sys, json
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import sympy as sp
import reps as R, corpus as CP, lattice as LT
from trees import V
from common import LAW9
_tf = R.template_force


def template_force(sysm, it, i, S):
    t = _tf(sysm, it, i, S)
    if t is None and it[0] in CP.CENTRAL and sysm.d == 2:
        _, cen = R.explicit_force(sysm, it, i, S)
        s, dx, dy = cen
        return sp.Mul(s, V(dx, dy), evaluate=False) if False else s * V(dx, dy)
    return t


R.template_force = template_force
code = sys.argv[1]
level = sys.argv[2] if len(sys.argv) > 2 else 'D1'
j = LT.Job(code, level, B=16, check=True, log=lambda *a: None).run()
v = j.values()
print(code, level, 'maxerr %.1e' % max(e[-1] for e in j.errors),
      {f: round(v['values'][f]['Phi'], 1) for f in LAW9},
      'LC add/loo %.1f/%.1f' % (v['values']['LC']['add'], v['values']['LC']['loo']))
json.dump({f: v['values'][f] for f in LAW9}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_vec_%s_%s.json' % (code, level)), 'w'))

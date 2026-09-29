"""Fairness probe F1: give the no-N2 baseline a generic 'common divisor' variant.
Without N2 each output a_i may be written as (sum of force-unit terms)/m_i  (pure algebra:
factoring a common 1/m_i), in addition to the pilot's per-term division.  Everything else
unchanged.  Run: PYTHONHASHSEED=0 python3 fair_n2.py CODE LEVEL [variant]
variant: 'fair' (with factored variant) or 'orig'."""
import os, sys, json, math
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1)
os.chdir(PV1)
import sympy as sp
import reps as R
import corpus as CP
import lattice as LT
from common import LAW9

_orig_build_all = R.build_all


def build_newton_factored(sysm, S, ct):
    S2 = frozenset(S) | {'N2'}
    items, meta = R.build_newton(sysm, S2, ct)
    temps = [(n, e) for n, e in items if n is not None]
    outs = [e for n, e in items if n is None]
    outs = [o / sysm.M(i) for o, i in zip(outs, sysm.bodies)]
    meta = dict(meta)
    meta['n2'] = False
    meta['variant'] = 'factored'
    return temps + [(None, o) for o in outs], meta


def build_all_fair(sysm, form, S, level, code, B, truth):
    out = _orig_build_all(sysm, form, S, level, code, B, truth)
    if sysm.fam == 'P' and form == 'newton' and 'N2' not in S and len(sysm.inter) > 0:
        ct = R.make_cost_term(sysm, form, S, level, code, B)
        out.append(R.realize(sysm, *build_newton_factored(sysm, S, ct), S, level, code, B))
    return out


if __name__ == '__main__':
    code, level = sys.argv[1], sys.argv[2]
    variant = sys.argv[3] if len(sys.argv) > 3 else 'fair'
    B = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    if variant == 'fair':
        R.build_all = build_all_fair
    j = LT.Job(code, level, B=B, check=True, log=lambda *a: None).run()
    v = j.values(charged=False)
    vc = j.values(charged=True)
    maxerr = max((e[-1] for e in j.errors), default=0.0)
    nfact = sum(1 for (sid, m), (rep, fl) in j.choice.items() if rep.meta.get('variant') == 'factored')
    res = dict(code=code, level=level, B=B, variant=variant, maxerr=maxerr, n_factored_chosen=nfact,
               Phi={f: v['values'][f]['Phi'] for f in LAW9},
               Phi_charged={f: vc['values'][f]['Phi'] for f in LAW9},
               mid={f: v['values'][f]['mid'] for f in LAW9},
               phi_sys_N2={sid: p[0] for sid, p in v['phi_sys'].items()},
               L0=v['L0'], Lfull=v['Lfull'], gap=v['efficiency_gap'])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res_%s_%s_%s_B%d.json' % (variant, code, level, B))
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('code', 'level', 'variant', 'maxerr', 'n_factored_chosen', 'L0', 'Lfull')}))
    print({f: round(x, 1) for f, x in res['Phi'].items()})

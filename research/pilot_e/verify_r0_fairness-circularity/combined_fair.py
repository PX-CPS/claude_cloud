"""Combined fair baseline F1 (factored 1/m_i without N2) + F7 (s*V(dx,dy) for central terms,
base grammar).  Miner off.  Reports Phi (uncharged / charged), V_mid, per code/level."""
import os, sys, json
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
HERE = '/home/user/claude_cloud/research/pilot_e/verify_r0_fairness-circularity'
sys.path.insert(0, HERE)
import reps as R, lattice as LT
from common import LAW9
code, level = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], code, level]
import vec_baseline_patch  # noqa  (patches R.template_force)
import fair_n2
R.build_all = fair_n2.build_all_fair
j = LT.Job(code, level, B=16, check=True, log=lambda *a: None).run()
v, vc = j.values(), j.values(charged=True)
res = dict(maxerr=max(e[-1] for e in j.errors), law_single=j.law_single, dec=j.dec,
           Phi={f: v['values'][f]['Phi'] for f in LAW9}, Phi_ch={f: vc['values'][f]['Phi'] for f in LAW9},
           mid={f: v['values'][f]['mid'] for f in LAW9}, mid_ch={f: vc['values'][f]['mid'] for f in LAW9},
           phi_sys=v['phi_sys'])
json.dump(res, open(os.path.join(HERE, 'res_comb_%s_%s.json' % (code, level)), 'w'))
print(code, level, 'maxerr %.1e' % res['maxerr'], {f: round(x) for f, x in res['Phi'].items()},
      'EC charged', round(res['Phi_ch']['EC']), 'mid', {f: round(x) for f, x in res['mid'].items()})

import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, lattice as LT
from common import LAW9
code, level = sys.argv[1], sys.argv[2]
truth = CP.build_truth()
job = LT.Job(code, level, 16, check=True, truth=truth, log=lambda *a: None).run()
v = job.values(False)
vc = job.values(True)
out = dict(code=code, level=level, FACTOR=os.environ.get('FACTOR', '1'),
           Phi={f: v['values'][f]['Phi'] for f in LAW9}, mid={f: v['values'][f]['mid'] for f in LAW9},
           Phi_charged={f: vc['values'][f]['Phi'] for f in LAW9},
           L0=v['L0'], Lfull=v['Lfull'], gap=v['efficiency_gap'],
           maxerr=max(e[-1] for e in job.errors),
           phi_sys=v['phi_sys'])
json.dump(out, open('var_%s_%s_F%s.json' % (code, level, out['FACTOR']), 'w'))
print(code, level, 'F', out['FACTOR'], 'maxerr %.2e' % out['maxerr'], {f: round(x, 1) for f, x in out['Phi'].items()})

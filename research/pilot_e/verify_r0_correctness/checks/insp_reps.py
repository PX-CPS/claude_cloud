import sys, os
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp
import corpus as CP, reps as R, trees as TR, lattice as LT
from common import LAW9
truth = CP.build_truth()
byid = CP.by_id()
code = sys.argv[1]; level = sys.argv[2]; sid = sys.argv[3]
Ts = [frozenset(x.split(',')) if x != '-' else frozenset() for x in sys.argv[4:]]
job = LT.Job(code, level, 16, check=True, truth=truth)
s = byid[sid]
for T in Ts:
    Tk = T & s.applicable
    c, rep, flag = job.sys_best(s, frozenset(Tk))
    print('==', sid, code, level, sorted(T), 'cost', round(c, 3), 'flag', flag, 'form', rep.form, rep.meta.get('variant'), 'consts', {k: str(v) for k, v in rep.consts.items()} if rep.consts else '')
    for f in R.formulations(s, level):
        if f == 'el' and 'EC' not in Tk: continue
        for a in job.alts.get((sid, f, R.form_subset(s, f, Tk)), []):
            print('   alt', f, a.meta.get('variant'), round(a.cost, 3))
    for lab, tid in rep.items:
        print('   ', lab, sp.sstr(TR.to_sympy_print(tid)), ' size', TR.SIZE[tid])
print('max U5 err', max(e[-1] for e in job.errors))

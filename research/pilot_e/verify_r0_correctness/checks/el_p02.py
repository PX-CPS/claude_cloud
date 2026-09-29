import sys
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp
import corpus as CP, reps as R, trees as TR, lattice as LT
truth = CP.build_truth(); s = CP.by_id()['P02']
job = LT.Job('C1', 'D1', 16, check=False, truth=truth)
job.sys_best(s, frozenset({'EC'}))
a = job.alts[('P02','el',frozenset())][0]
for lab, tid in a.items:
    print(lab, sp.srepr(TR.to_sympy_print(tid)))
    def walk(i, d=0):
        k, l, ch = TR.NODES[i]; print('  '*d, k, l, TR.profile(i)[0]); [walk(c, d+1) for c in ch]
    walk(tid)

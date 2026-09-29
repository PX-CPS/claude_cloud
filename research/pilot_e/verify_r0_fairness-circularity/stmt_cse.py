"""Fairness probe F2: law statements are coded without CSE (DEVIATIONS C7), unlike every
system representation.  Recompute HK/UG statement costs with sympy.cse applied to each
statement item (shared subterms become local lets, each paying one extra item header)."""
import os, sys
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import sympy as sp
import laws as LW, reps as R, trees as TR
from common import gamma_len, LAW9

def cse_cost(f, T, used, code, level='D1'):
    nlib = len(R.lib_names(T)) + 5
    from common import builtins_for
    nb = len(builtins_for(level))
    tot = 0.0
    for e, fm in LW.law_items(f, 'DIST' in T, used):
        repl, red = sp.cse([e], symbols=sp.numbered_symbols('lt'), optimizations='basic')
        formals = list(fm)
        for s_, v_ in repl:
            tot += LW.item_cost(v_, formals, code, nlib, nb, text_extra=3)
            formals = formals + [s_]
        tot += LW.item_cost(red[0], formals, code, nlib, nb)
    return tot

for code in ['C1', 'C2', 'C3']:
    if code == 'C2':
        import lattice as LT, corpus as CP
        TR.FREQ.from_json(LT.fit_freq(CP.build_truth(), 'D1'))
    for f, used in (('HK', {'Fh1', 'Fh2', 'Vh1', 'Vh2'}), ('UG', set())):
        for dist in (False, True):
            T = frozenset([f] + (['DIST'] if dist else []))
            a = LW.law_cost(f, T, used, code, 'D1')
            b = cse_cost(f, T, used, code)
            print(code, f, 'DIST' if dist else 'noDIST', 'orig %.1f  cse %.1f  diff %.1f' % (a, b, a - b))

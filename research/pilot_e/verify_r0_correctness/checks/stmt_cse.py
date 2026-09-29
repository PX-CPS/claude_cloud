import sys
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp
import laws as LW, trees as TR, reps as R
from common import gamma_len, builtins_for
def cost_cse(f, T, used, code='C1'):
    nlib = len(R.lib_names(T)) + 5; nb = len(builtins_for('D1'))
    items = LW.law_items(f, 'DIST' in T, used)
    plain = sum(LW.item_cost(e, fm, code, nlib, nb) for e, fm in items)
    tot = 0
    for e, fm in items:
        repl, red = sp.cse([e], symbols=sp.numbered_symbols('_x'), optimizations='basic')
        holes = {s: j for j, s in enumerate(fm)}
        c = gamma_len(len(fm) + 1) + gamma_len(len(repl) + 2)
        for k, (n, ex) in enumerate(repl):
            h = dict(holes); h.update({repl[j][0]: len(fm) + j for j in range(k)})
            tid = TR.from_sympy(ex, 'LAW', law_mode=True, holes=h)
            c += TR.tree_cost(tid, code, {'hole': len(h)}, nlib, nb)
        h = dict(holes); h.update({repl[j][0]: len(fm) + j for j in range(len(repl))})
        tid = TR.from_sympy(red[0], 'LAW', law_mode=True, holes=h)
        c += TR.tree_cost(tid, code, {'hole': len(h)}, nlib, nb)
        tot += min(c, LW.item_cost(e, fm, code, nlib, nb))
    return plain, tot
print('UG', cost_cse('UG', frozenset({'UG'}), set()))
print('HK all4', cost_cse('HK', frozenset({'HK'}), {'Fh1', 'Fh2', 'Vh1', 'Vh2'}))

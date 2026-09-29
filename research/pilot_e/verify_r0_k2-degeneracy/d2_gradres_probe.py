"""H2/K2-collapse probe at D2: compare the EL representation (EC in theory) with the base-grammar
gradient-residual representation (Lt temp + n outputs Dt(Dq(Lt,v_i)) - Dq(Lt,q_i)), which is available to
every theory at D2.  If EC's per-system saving equals the cost of the n residual outputs, EC at D2 is a
macro with an implicit loop over coordinates (same structure as PC's implicit last output)."""
import sys, math
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp
import corpus as CP, reps as R, trees as TR
truth = CP.build_truth()
byid = {s.sid: s for s in CP.corpus()}
for code in ['C1', 'C3']:
    for sid in ['G03', 'G04', 'G07']:
        s = byid[sid]
        S0 = frozenset()
        el = R.realize(s, *R.build_el_G(s, S0), S0, 'D2', code, 16)
        gr = R.realize(s, *R.build_gradres_G(s), S0, 'D2', code, 16)
        ctx = R.SysCtx(s, 'implicit', S0, 'D2', code)
        print(code, sid, 'n=%d' % s.n, 'EL cost %.1f (%d items)' % (el.cost, len(el.items)), ' gradres cost %.1f (%d items)' % (gr.cost, len(gr.items)))
        for lab, tid in gr.items:
            print('    gr item', lab, TR.to_sympy_print(tid) if hasattr(TR, 'to_sympy_print') else tid)
        for lab, tid in el.items:
            print('    el item', lab, str(TR.to_sympy_print(tid))[:200])

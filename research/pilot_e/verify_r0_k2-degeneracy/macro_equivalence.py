"""K2-collapse by explicit macro rewriting.
(1) EC at D2, family G: EL rep vs base-grammar gradient-residual rep (Lt temp + n outputs E(Lt,q_i,v_i)).
    Per-system EC saving is decomposed into n * (cost of one EL-operator output) + overhead, i.e. a macro
    applied once per coordinate (an implicit loop).  We also compute the saving left over if the base grammar
    had one 'map' call per system: MAPCALL = kind(3) + log2(lib+1) + 1 leaf.
(2) PC, family P isolated systems: saving vs the cost of the base-grammar output-referencing item
    a_N = -sum m_i O_i / m_N (spec base mechanism iii).
All costs recomputed with the pilot's own realize()/items_cost (read-only import)."""
import sys, math, json
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import numpy as np
import corpus as CP, reps as R, trees as TR
from common import LAW9
truth = CP.build_truth()
byid = {s.sid: s for s in CP.corpus()}
RES = '/home/user/claude_cloud/research/pilot_e/pilot_v1/results/'
out = {'EC_D2': {}, 'PC_D1': {}}
for code in ['C1', 'C2', 'C3', 'C4']:
    job = json.load(open(RES + 'job_%s-D2-B16.json' % code))
    if code == 'C2':
        import lattice as LT; TR.FREQ.from_json(job['freq']) if 'freq' in job else LT.fit_freq(truth, 'D2')
    for sid in ['G03', 'G04', 'G07']:
        s = byid[sid]
        S = frozenset({'EC', 'KE'}) & s.applicable
        el = R.realize(s, *R.build_el_G(s, frozenset()), frozenset(), 'D2', code, 16)
        gr = R.realize(s, *R.build_gradres_G(s), frozenset(), 'D2', code, 16)
        ctx = R.SysCtx(s, 'implicit', frozenset(), 'D2', code)
        # cost of the residual outputs alone, in the gradres rep's scope
        ntemps = sum(1 for l, _ in gr.items if l is not None)
        tc = 'C1' if code in ('C4', 'C5') else code
        outc = []
        oi = 0
        for lab, tid in gr.items:
            if lab is None:
                sizes = {'state': ctx.n_state, 'param': ctx.n_param if code != 'C4' else len(gr.consts), 'temp': ntemps, 'out': oi}
                outc.append(TR.tree_cost(tid, tc, sizes, ctx.nlib, ctx.nbuiltin)); oi += 1
        Lk = np.array(job['Lk'][sid]); b = 1 << LAW9.index('EC')
        add = Lk[0] - Lk[b]; loo = Lk[511 ^ b] - Lk[511]
        mapcall = 3 + math.log2(len(R.lib_names(frozenset())) + 2) + math.log2(ctx.n_state + ctx.n_param + ntemps)
        out['EC_D2'][code + ':' + sid] = dict(n=s.n, EL=el.cost, gradres=gr.cost, gap=gr.cost - el.cost,
                                               residual_outputs=outc, sum_residuals=sum(outc), add=add, loo=loo,
                                               baseline_form=job['choice_summary']['0'][sid]['variant'],
                                               saving_left_with_mapcall=gr.cost - sum(outc) + mapcall - el.cost)
        o = out['EC_D2'][code + ':' + sid]
        print('EC D2 %s %s n=%d EL=%.1f gradres=%.1f gap=%.1f  sum(residual outputs)=%.1f (per-output %s)  V_add=%.1f V_loo=%.1f base=%s  left-with-mapcall=%.1f' % (
            code, sid, s.n, el.cost, gr.cost, gr.cost - el.cost, sum(outc), np.round(outc, 1), add, loo, o['baseline_form'], o['saving_left_with_mapcall']))
# PC
for code in ['C1', 'C2', 'C3', 'C4']:
    job = json.load(open(RES + 'job_%s-D1-B16.json' % code))
    for sid in ['P06', 'P09', 'P10', 'P11', 'P12', 'P15']:
        s = byid[sid]
        Lk = np.array(job['Lk'][sid]); b = 1 << LAW9.index('PC'); bn2 = 1 << LAW9.index('N2')
        for T, mask in (('{}', 0), ('{N2}', bn2)):
            S = frozenset([f for f in LAW9 if mask >> LAW9.index(f) & 1])
            ct = R.make_cost_term(s, 'newton', S, 'D1', code, 16)
            std = R.realize(s, *R.build_newton(s, S, ct), S, 'D1', code, 16)
            orf = R.realize(s, *R.build_newton(s, S, ct, outref=True), S, 'D1', code, 16)
            SP = S | {'PC'}
            ct2 = R.make_cost_term(s, 'newton', SP, 'D1', code, 16)
            pc = R.realize(s, *R.build_newton(s, SP, ct2), SP, 'D1', code, 16)
            v = Lk[mask] - Lk[mask | b]
            out['PC_D1'][code + ':' + sid + T] = dict(std=std.cost, outref=orf.cost, pc=pc.cost, V_PC=v,
                                                       outref_minus_pc=orf.cost - pc.cost, std_minus_outref=std.cost - orf.cost)
            print('PC D1 %s %s T=%-5s std=%.1f outref=%.1f PC=%.1f  V_PC=%.1f  outref-PC=%.1f (cost of the implicit last output beyond base out-ref)' % (
                code, sid, T, std.cost, orf.cost, pc.cost, v, orf.cost - pc.cost))
json.dump(out, open('macro_equivalence.json', 'w'), indent=1, default=float)

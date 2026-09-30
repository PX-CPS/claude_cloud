"""Rewrite law WITH-encodings as generic corpus-level macros and compare savings (verify r1, K2-collapse).

(A) EC at D2M, family G.  The base grammar already has the one-item residual form
      [Lt := T - V ;  Map(Dt(Dq(Lt,qb^v)) - Dq(Lt,qb))]
    The pilot's miner may NOT mine Map bodies (bound variables are excluded from patterns, R7 fix), so this
    identical item is re-paid in every system.  A generic 1-hole corpus template ELmap(h) := Map(Dt(Dq(h,vb))-Dq(h,qb))
    replaces it by one Call(Lt).  We compute, per system, the EC saving left after that macro:
        left_k = cost(EL rep) - [cost(gradres_map rep) - cost(Map item) + cost(Call ELmap(Lt))]
    and the one-off definition cost of ELmap.
(B) PC at D1: saving of PC vs the base output-referencing form (last body a_N = -sum O_j or -sum m_j O_j/m_N),
    per system; compared with the tree cost of that one referencing item (= the implicit-argument macro body).
(C) N2 at D1: saving of N2 vs the factored base form (sum F)/m_i, per body.
All costs via the pilot's own realize()/tree_cost (read-only import)."""
import sys, math, json
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import numpy as np
import corpus as CP, reps as R, trees as TR, lattice as LT
from common import LAW9, gamma_len
from load import job, bit, byid

truth = CP.build_truth()
out = {'EC_D2M': {}, 'PC_D1': {}, 'N2_D1': {}}


def set_code(code, level):
    if code == 'C2':
        j = job('C2-%s-B16' % level)
        if 'freq' in j:
            TR.FREQ.from_json(j['freq'])
        else:
            LT.fit_freq(truth, level)


def item_costs(rep, sysm, code, level):
    ctx = R.SysCtx(sysm, rep.form, rep.S, level, code, rep.meta.get('variant'))
    ntemps = sum(1 for l, _ in rep.items if l is not None)
    tc = 'C1' if code in ('C4', 'C5') else code
    npar = ctx.n_param if code != 'C4' else len(rep.consts) + sum(1 for s in sysm.params if s in R.licensed_symbols(sysm, rep.S))
    res = []
    ti = oi = 0
    for lab, tid in rep.items:
        if lab is not None:
            sizes = {'state': ctx.n_state, 'param': npar, 'temp': ti, 'out': 0}; ti += 1
        else:
            sizes = {'state': ctx.n_state, 'param': npar, 'temp': ntemps, 'out': oi}; oi += 1
        res.append((lab, tid, TR.tree_cost(tid, tc, sizes, ctx.nlib, ctx.nbuiltin), sizes, ctx))
    return res


# ---------------- (A) EC at D2M
for code in ['C1', 'C2', 'C3', 'C4']:
    set_code(code, 'D2M')
    jb = job('%s-D2M-B16' % code)
    tot_left, tot_rep = 0.0, 0.0
    defcost = None
    for sid in ['G01', 'G03', 'G04', 'G05', 'G06', 'G07']:
        s = byid[sid]
        S0 = frozenset()
        el = R.realize(s, *R.build_el_G(s, frozenset()), frozenset({'EC'}), 'D2M', code, 16)
        gm = R.realize(s, *R.build_gradres_map_G(s), S0, 'D2M', code, 16)
        ic = item_costs(gm, s, code, 'D2M')
        mapc = [c for lab, tid, c, sz, ctx in ic if lab is None]
        assert len(mapc) == 1
        lab, tid, cmap, sz, ctx = [x for x in ic if x[0] is None][0]
        # call to a 1-hole library template with one leaf argument (the temp Lt), coded like the pilot's calls
        if code == 'C3':
            call = (len('Zx(T0)')) * math.log2(95)
        else:
            tc = 'C1' if code == 'C4' else code
            leaf = TR.mk('Sym', ('s', s.params[0]))  # any in-scope leaf: same cost as the temp leaf in C1/C4
            # kind(Call)+log2(nlib+1) + one leaf in the same scope
            call_node = TR.mk('Call', ('M', 999), (leaf,))
            call = TR.tree_cost(call_node, tc, sz, ctx.nlib + 1, ctx.nbuiltin)
        # template definition: the Map body with the hole; approx = the Map item cost in a scope of 1 hole
        if defcost is None:
            tc = 'C1' if code == 'C4' else code
            defcost = TR.tree_cost(tid, tc, {'hole': 1, 'state': 2}, ctx.nlib, ctx.nbuiltin) if code != 'C3' else cmap
        Lk = np.array(jb['Lk'][sid])
        b = bit('EC')
        add, loo = Lk[0] - Lk[b], Lk[511 ^ b] - Lk[511]
        left = el.cost - (gm.cost - cmap + call)
        out['EC_D2M'][code + ':' + sid] = dict(n=s.n, EL=el.cost, gradres_map=gm.cost, map_item=cmap, call=call,
                                               left_after_macro=left, V_add=float(add), V_loo=float(loo),
                                               base_choice=jb['choice_summary']['0'][sid]['variant'])
        tot_left += left
        tot_rep += add
        print('EC D2M %s %s n=%d EL=%.1f gradres_map=%.1f map_item=%.1f call=%.1f  left-after-ELmap-macro=%.1f   pilot V_add=%.1f V_loo=%.1f base=%s' % (
            code, sid, s.n, el.cost, gm.cost, cmap, call, left, add, loo, jb['choice_summary']['0'][sid]['variant']))
    out['EC_D2M'][code + ':TOTAL_G'] = dict(sum_left=tot_left, ELmap_def=defcost, sum_V_add_G=tot_rep,
                                            Phi_EC=jb['values']['values']['EC']['Phi'])
    print('   %s TOTAL family G: sum V_add=%.1f; sum left after ELmap macro=%.1f; ELmap definition ~%.1f; Phi_EC(all)=%.1f' % (
        code, tot_rep, tot_left, defcost, jb['values']['values']['EC']['Phi']))

# ---------------- (B) PC and (C) N2 at D1
for code in ['C1', 'C2', 'C3', 'C4']:
    set_code(code, 'D1')
    jb = job('%s-D1-B16' % code)
    for sid in ['P06', 'P09', 'P10', 'P11', 'P12', 'P15']:
        s = byid[sid]
        Lk = np.array(jb['Lk'][sid])
        for Tn, S in (('{N2}', frozenset({'N2'})), ('{}', frozenset())):
            ct = R.make_cost_term(s, 'newton', S, 'D1', code, 16)
            reps_noPC = R.build_all(s, 'newton', S, 'D1', code, 16, truth)
            best = min(reps_noPC, key=lambda r: r.cost)
            SP = S | {'PC'}
            reps_PC = R.build_all(s, 'newton', SP, 'D1', code, 16, truth)
            bpc = min(reps_PC, key=lambda r: r.cost)
            # cost of the last (referencing) output in the best no-PC rep, if it is an outref variant
            ic = item_costs(best, s, code, 'D1')
            last = [c for lab, tid, c, sz, ctx in ic if lab is None][-1]
            mask = 0 if Tn == '{}' else bit('N2')
            V = Lk[mask] - Lk[mask | bit('PC')]
            out['PC_D1'][code + ':' + sid + Tn] = dict(nbody=s.n, best_noPC=best.cost, variant=best.meta.get('variant'),
                                                      PC=bpc.cost, saving=best.cost - bpc.cost,
                                                      last_output_item=last, side_bits_PC=bpc.side_bits,
                                                      side_bits_noPC=best.side_bits, V_from_Lk=float(V))
            print('PC D1 %s %s T=%-4s N=%d noPC-best=%.1f(%s) PC=%.1f saving=%.1f | last-output item=%.1f PCside=%.1f | Lk-diff(incl flags)=%.1f' % (
                code, sid, Tn, s.n, best.cost, best.meta.get('variant'), bpc.cost, best.cost - bpc.cost, last, bpc.side_bits, V))
    # N2 per body
    for sid in ['P01', 'P02', 'P05', 'P06', 'P07', 'P10', 'P11', 'P12', 'P14', 'P16']:
        s = byid[sid]
        Lk = np.array(jb['Lk'][sid])
        v = Lk[0] - Lk[bit('N2')]
        out['N2_D1'][code + ':' + sid] = dict(nbody=s.n, V_add=float(v), per_body=float(v) / s.n)
    print('N2 %s V_add per body:' % code, {k.split(':')[1]: round(v['per_body'], 1) for k, v in out['N2_D1'].items() if k.startswith(code)})
json.dump(out, open('macro_rewrite.json', 'w'), indent=1, default=float)

"""Fairness probe F5: the EL decoder applies its residual to EVERY coordinate and knows the
(q_i, v_i) pairing for free.  Give the baseline the same 'for each coordinate' primitive at D2:
the gradres residual Dt(Dq(Lt,v))-Dq(Lt,q) is written once (+3 bits Map node) instead of n times.
Family G only (where EC's D1/D2 value concentrates)."""
import os, sys, math
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import sympy as sp
import reps as R, corpus as CP, trees as TR, lattice as LT
truth = CP.build_truth()
for code in ['C1', 'C2', 'C3']:
    if code == 'C2':
        TR.FREQ.from_json(LT.fit_freq(truth, 'D2'))
    tot_now = tot_map = 0
    for s in CP.corpus():
        if s.fam != 'G' or 'EC' not in s.applicable:
            continue
        items, meta = R.build_gradres_G(s)
        full = R.realize(s, items, meta, frozenset(), 'D2', code, 16).cost
        one = R.realize(s, items[:2], meta, frozenset(), 'D2', code, 16).cost + 3
        others = [R.realize(s, *R.build_implicit_G(s, truth), frozenset(), 'D2', code, 16).cost,
                  R.realize(s, *R.build_explicit_G(s, truth), frozenset(), 'D2', code, 16).cost]
        el = R.realize(s, *R.build_el_G(s, frozenset()), frozenset(['EC']), 'D2', code, 16).cost
        nf_no, nf_ec = 2, 3
        v_now = min([full] + others) + math.log2(nf_no) - (min([full, el] + others) + math.log2(nf_ec))
        v_map = min([one] + others) + math.log2(nf_no) - (min([one, el] + others) + math.log2(nf_ec))
        tot_now += v_now; tot_map += v_map
        print(code, s.sid, 'n=%d gradres %.1f  gradres+map %.1f  EL %.1f | EC per-system value (no other laws): now %.1f  with map %.1f'
              % (s.n, full, one, el, v_now, v_map))
    print(code, 'family-G EC value total: now %.1f  with map %.1f' % (tot_now, tot_map))

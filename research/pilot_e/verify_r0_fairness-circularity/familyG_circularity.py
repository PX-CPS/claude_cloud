"""Family G: truth = EL(T - V) (corpus.GSys.truth).  How much of EC's per-system value is the
expansion of EC's own decoder, as a function of how much of that decoder the baseline machine has?"""
import sys, os, math, json
os.environ.setdefault('PYTHONHASHSEED', '0')
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp
import corpus as CP, reps as R, trees as TR
from common import gamma_len
truth = CP.build_truth()
out = {}
for code in ['C1', 'C3']:
    for s in CP.corpus():
        if s.fam != 'G' or 'EC' not in s.applicable:
            continue
        row = {}
        for level in ['D0', 'D1', 'D2']:
            S = frozenset()
            costs = {}
            for form in R.formulations(s, level):
                for r in R.build_all(s, form, frozenset({'EC'}) if form == 'el' else S, level, code, 16, truth):
                    costs[form + ':' + r.meta['variant']] = r.cost
            nf = len(R.formulations(s, level))
            best_noEC = min(v for k, v in costs.items() if not k.startswith('el')) + math.log2(nf - 1 if nf > 2 else 1)
            with_EC = min(costs.values()) + math.log2(nf)
            row[level] = dict(costs={k: round(v, 1) for k, v in costs.items()}, value=round(best_noEC - with_EC, 1))
        # D2 + map primitive: gradres residual written once for all coordinates  ('Map' node: 3 bits kind + 0 args)
        g = row['D2']['costs']['implicit:gradres']
        el = row['D2']['costs']['el:std']
        # gradres = gamma(#items+1) + Lt def + n residuals; with a map the n residuals collapse to 1 residual
        # template applied to all i -> cost ~ el + (one residual pattern with index holes) + temp overhead.
        n = s.n
        per_res = (g - el) / n if n else 0
        row['D2_gradres_minus_el'] = round(g - el, 1)
        row['D2_per_dof_residual_cost'] = round(per_res, 1)
        out['%s:%s' % (code, s.sid)] = row
        print(code, s.sid, 'n', n, {lv: row[lv]['value'] for lv in ['D0', 'D1', 'D2']},
              'D2 gradres-el', row['D2_gradres_minus_el'], 'per-DOF', row['D2_per_dof_residual_cost'],
              'D1 costs', row['D1']['costs'])
json.dump(out, open('familyG_circularity.json', 'w'), indent=1)

#!/usr/bin/env python3
"""Leave-one-system-out and tie-margin sensitivity of the K1 test with the new codes (D1, uncharged)."""
import json, os, numpy as np
import analyze_r1 as A
HERE = os.path.dirname(os.path.abspath(__file__))
LAW9 = A.LAW9
P = {c: A.pilot(c, 'D1') for c in ('C1', 'C2', 'C3', 'C4')}
N = {c: A.mine(c, 'D1') for c in ('TBX', 'EQW', 'NCT', 'C2F')}
out = {}
def phis_drop(sets, drop):
    r = {}
    for c, d in sets.items():
        tot = np.array(d['phi_ls'], float)
        for sid, v in d['phi_sys'].items():
            if sid not in drop:
                tot = tot + np.array(v, float)
        r[c] = dict(zip(LAW9, tot))
    return r
sids = sorted(P['C1']['phi_sys'])
for c in N:
    sets = {**P, c: N[c]}
    fired = []
    for s in sids:
        r = A.k1(phis_drop(sets, {s}))
        fired.append(s if not (r['fired_a'] or r['fired_b']) else None)
    out['loso_primary+' + c] = [s for s in fired if s]
    print('primary+%s: K1 NOT fired when dropping' % c, [s for s in fired if s])
# drop all three N-body gravity systems at once
for c in N:
    r = A.k1(phis_drop({**P, c: N[c]}, {'P09', 'P10', 'P11'}))
    out['drop_gravity_primary+' + c] = dict(tops=r['tops'], min_rho=r['min_rho'], flips=r['flips'])
    print('drop P09-P11, primary+%s: tops %s min_rho %.3f nflips %d' % (c, r['tops'], r['min_rho'], len(r['flips'])))
r = A.k1(phis_drop(N, {'P09', 'P10', 'P11'}))
print('drop P09-P11, new-only: tops', r['tops'], 'min_rho %.3f' % r['min_rho'])
# tie margin: largest margin m such that top(a) still < (1-m) max_b for some pair
for c in N:
    ph = {**{k: v['Phi'] for k, v in P.items()}, c: N[c]['Phi']}
    top = max(LAW9, key=ph[c].get)
    ratios = {b: ph[b][top] / max(ph[b].values()) for b in ph if b != c}
    tb = {b: max(LAW9, key=ph[b].get) for b in ph if b != c}
    ratios2 = {b: ph[c][tb[b]] / max(ph[c].values()) for b in ph if b != c}
    out['margin_' + c] = dict(top_new=top, ratio_in_primary=ratios, ratio_primary_top_in_new=ratios2)
    print('%s top=%s; Phi_b(top_new)/max_b:' % (c, top), {b: round(x, 3) for b, x in ratios.items()},
          '; Phi_new(top_b)/max_new:', {b: round(x, 3) for b, x in ratios2.items()})
json.dump(out, open(os.path.join(HERE, 'results', 'loso_r1.json'), 'w'), indent=1)

#!/usr/bin/env python3
import os, sys, json, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pilot_v1'))
import numpy as np
from common import LAW9, PHYS7
import analysis as AN

R = json.load(open(os.path.join(HERE, 'results', 'phi_all.json')))
# cross-check against the pilot's own numbers
pil = json.load(open(os.path.join(os.path.dirname(HERE), 'pilot_v1', 'results', 'analysis.json')))
for c in ('C1', 'C2', 'C3', 'C4'):
    a = pil['tables']['%s-D1-B16' % c]['values']
    d = max(abs(a[f]['Phi'] - R['%s-D1' % c]['uncharged']['Phi'][f]) for f in LAW9)
    print('reproduction max |dPhi|', c, round(d, 6))

PRIM = ['C1', 'C2', 'C3', 'C4']
NEWC = ['C6', 'C7', 'C9', 'C10', 'C11']
ALL = PRIM + NEWC
out = {}


def P(code, lv='D1', ch=False):
    return R['%s-%s' % (code, lv)]['charged' if ch else 'uncharged']['Phi']


def table(lv, ch):
    print('\n== level %s charged=%s ==' % (lv, ch))
    print('%-5s' % 'law' + ''.join('%9s' % c for c in ALL))
    for f in LAW9:
        print('%-5s' % f + ''.join('%9.1f' % P(c, lv, ch)[f] for c in ALL))
    tops = {c: max(LAW9, key=lambda f: P(c, lv, ch)[f]) for c in ALL}
    print('top  ' + ''.join('%9s' % tops[c] for c in ALL))
    ranks = {c: sorted(LAW9, key=lambda f: -P(c, lv, ch)[f]) for c in ALL}
    for c in ALL:
        print(c, ' > '.join(ranks[c]))
    return tops, ranks


for lv in ('D0', 'D1', 'D2'):
    for ch in ((False, True) if lv != 'D2' else (False,)):
        tops, ranks = table(lv, ch)
        key = '%s_%s' % (lv, 'charged' if ch else 'uncharged')
        Phi = {c: P(c, lv, ch) for c in ALL}
        o = {}
        # each new code against the four primary codes (the K1-primary rule applied to the extended set)
        o['extended_all9'] = AN.k1_test(Phi, ALL)
        o['each_new_vs_primary'] = {n: AN.k1_test(Phi, PRIM + [n]) for n in NEWC}
        o['pair_rho'] = {'%s-%s' % (a, b): AN.spearman([Phi[a][f] for f in LAW9], [Phi[b][f] for f in LAW9])
                         for a, b in itertools.combinations(ALL, 2)}
        o['pair_rho_7law'] = {'%s-%s' % (a, b): AN.spearman([Phi[a][f] for f in PHYS7], [Phi[b][f] for f in PHYS7])
                              for a, b in itertools.combinations(ALL, 2)}
        o['tops'] = tops
        o['ranks'] = ranks
        o['ratio_EC_to_best_other'] = {c: Phi[c]['EC'] / max(Phi[c][f] for f in LAW9 if f != 'EC') for c in ALL}
        out[key] = o
        k = o['extended_all9']
        print('extended 9 codes: min rho %.3f, #flips %d, fired %s' % (k['min_rho'], len(k['flips']), k['fired']))
        for n in NEWC:
            kk = o['each_new_vs_primary'][n]
            rh = [v for p, v in kk['pairs'].items() if n in p.split('-')]
            print('  %s vs C1-C4: rho(new,prim) min %.3f  flips %d  top %s  fired %s' %
                  (n, min(rh), len([x for x in kk['flips'] if n in (x['a'], x['b'])]), kk['tops'][n], kk['fired']))
        print('  EC/best-other:', {c: round(v, 2) for c, v in o['ratio_EC_to_best_other'].items()})
        print('  min rho 9-law pairs incl new:', round(min(o['pair_rho'].values()), 3),
              ' 7-law:', round(min(o['pair_rho_7law'].values()), 3))

# machine axis for each new code
print('\n== machine axis (D0/D1/D2, uncharged) per code ==')
mach = {}
for c in ALL:
    Phi = {lv: P(c, lv) for lv in ('D0', 'D1', 'D2')}
    k = AN.k1_test(Phi, ['D0', 'D1', 'D2'])
    mach[c] = k
    print(c, 'tops', k['tops'], 'min rho %.3f' % k['min_rho'], 'fired', k['fired'])
out['machine'] = mach

# bootstrap on the extended 9-code set at D1 uncharged
ps = {c: R['%s-D1' % c]['uncharged']['phi_sys'] for c in ALL}
pl = {c: R['%s-D1' % c]['uncharged']['phi_ls'] for c in ALL}
bs = AN.bootstrap_k1(ps, pl, ALL, n=1000)
out['bootstrap_extended_D1'] = bs
print('\nbootstrap extended D1 uncharged: P(fire) %.3f P(flip) %.3f' % (bs['P_fire'], bs['P_flip']))

# leave-one-system-out: does removing a single system flip the top law in any code? (D1 uncharged)
sids = sorted(ps['C1'])
lo = {}
for s in sids:
    Phi = {c: dict(zip(LAW9, np.sum([ps[c][t] for t in sids if t != s], axis=0) + np.array(pl[c]))) for c in ALL}
    k = AN.k1_test(Phi, ALL)
    lo[s] = dict(tops=k['tops'], fired=k['fired'], min_rho=k['min_rho'])
drops = {s: v for s, v in lo.items() if v['fired']}
print('drop-one-system fires on extended set:', {s: (v['tops'], round(v['min_rho'], 3)) for s, v in drops.items()})
out['drop_one_system'] = lo

# statement costs
out['law_single'] = {c: R['%s-D1' % c]['law_single'] for c in ALL}
out['dec'] = {c: R['%s-D1' % c]['dec'] for c in ALL}
print('\nstatement costs L(f) at D1:')
for c in ALL:
    print(c, {f: round(v, 1) for f, v in out['law_single'][c].items()}, 'dec', round(out['dec'][c], 1))
json.dump(out, open(os.path.join(HERE, 'results', 'k1_extended.json'), 'w'), indent=1, default=str)

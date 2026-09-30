#!/usr/bin/env python3
"""K1 statistics for the pilot's primary codes plus the new r1 codes."""
import os, sys, json, itertools
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), 'pilot_v1')
LAW9 = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG', 'DIST', 'KE']
PHYS7 = LAW9[:7]
R1 = json.load(open(os.path.join(HERE, 'results', 'phi_r1.json')))


def pilot(code, level, B=16, charged=False):
    d = json.load(open(os.path.join(PILOT, 'results', 'job_%s-%s-B%d.json' % (code, level, B))))
    v = d['values_charged' if charged else 'values']
    return dict(Phi={f: v['values'][f]['Phi'] for f in LAW9}, phi_sys=v['phi_sys'], phi_ls=v['phi_ls'])


def mine(code, level, B=16, unit=4, charged=False):
    r = R1['%s-%s-B%d-U%g' % (code, level, B, unit)]
    return r['charged' if charged else 'uncharged']


def k1(phis, laws=LAW9):
    names = list(phis)
    pairs, flips = {}, []
    for a, b in itertools.combinations(names, 2):
        pairs[a + '~' + b] = float(spearmanr([phis[a][f] for f in laws], [phis[b][f] for f in laws])[0])
    tops = {c: max(laws, key=lambda f: phis[c][f]) for c in names}
    for a in names:
        for b in names:
            if a != b:
                mb = max(phis[b][f] for f in laws)
                if phis[b][tops[a]] < 0.9 * mb:
                    flips.append('%s:%s->%s:%s' % (a, tops[a], b, tops[b]))
    mn = min(pairs.values())
    return dict(min_rho=mn, argmin=min(pairs, key=pairs.get), pairs=pairs, tops=tops, flips=flips,
                fired_a=mn < 0.7, fired_b=bool(flips))


def rank(phi, laws=LAW9):
    return ' > '.join('%s(%.0f)' % (f, phi[f]) for f in sorted(laws, key=lambda f: -phi[f]))


def boot(sets, nb=1000, seed=20260929):
    """sets: name -> dict(phi_sys, phi_ls); resample systems; P(fire), P(flip), rho CI."""
    rng = np.random.default_rng(seed)
    sids = sorted(next(iter(sets.values()))['phi_sys'])
    nfire = nflip = 0
    mins = []
    for _ in range(nb):
        idx = rng.integers(0, len(sids), len(sids))
        phis = {}
        for c, d in sets.items():
            tot = np.array(d['phi_ls'], float)
            for i in idx:
                tot = tot + np.array(d['phi_sys'][sids[i]], float)
            phis[c] = dict(zip(LAW9, tot))
        r = k1(phis)
        nfire += r['fired_a'] or r['fired_b']
        nflip += r['fired_b']
        mins.append(r['min_rho'])
    return dict(P_fire=nfire / nb, P_flip=nflip / nb, min_rho_CI=[float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))])


if __name__ == '__main__':
    out = {}
    for lv in ('D0', 'D1', 'D2'):
        for charged in (False, True):
            P = {c: pilot(c, lv, charged=charged) for c in ('C1', 'C2', 'C3', 'C4')}
            N = {c: mine(c, lv, charged=charged) for c in ('TBX', 'EQW', 'NCT', 'C2F')}
            N['C4B4'] = mine('C4', lv, B=4, charged=charged)
            N['C4B64'] = mine('C4', lv, B=64, charged=charged)
            tag = lv + ('_charged' if charged else '')
            res = {}
            print('\n=====', tag)
            for c, d in list(P.items()) + list(N.items()):
                print('  %-6s %s' % (c, rank(d['Phi'])))
            for c in ('TBX', 'EQW', 'NCT', 'C2F'):
                r = k1({**{k: v['Phi'] for k, v in P.items()}, c: N[c]['Phi']})
                r7 = k1({**{k: v['Phi'] for k, v in P.items()}, c: N[c]['Phi']}, PHYS7)
                vs = {k: r['pairs'][k] for k in r['pairs'] if c in k.split('~')}
                res['primary+' + c] = dict(all9=r, phys7=dict(min_rho=r7['min_rho'], tops=r7['tops'], flips=r7['flips']))
                print('  primary+%-4s min_rho %.3f (%s)  new-vs-primary rho %s  top %s  flips(new) %s | phys7 min_rho %.3f' % (
                    c, r['min_rho'], r['argmin'], ' '.join('%.2f' % x for x in vs.values()), r['tops'][c],
                    [f for f in r['flips'] if c in f], r7['min_rho']))
            newset = {c: N[c]['Phi'] for c in ('TBX', 'EQW', 'NCT', 'C2F')}
            r = k1(newset)
            res['new_only'] = r
            print('  NEW-ONLY {TBX,EQW,NCT,C2F}: min_rho %.3f (%s) tops %s nflips %d' % (r['min_rho'], r['argmin'], r['tops'], len(r['flips'])))
            r = k1({**{k: v['Phi'] for k, v in P.items()}, **newset})
            res['all8'] = r
            print('  ALL8: min_rho %.3f (%s) fired_a %s fired_b %s' % (r['min_rho'], r['argmin'], r['fired_a'], r['fired_b']))
            prec = k1({'B4': N['C4B4']['Phi'], 'B8': pilot('C4', lv, 8, charged)['Phi'], 'B16': P['C4']['Phi'],
                       'B32': pilot('C4', lv, 32, charged)['Phi'], 'B64': N['C4B64']['Phi']})
            res['precision_B4_B64'] = prec
            print('  PRECISION B4..B64: tops %s flips %s min_rho %.3f' % (prec['tops'], prec['flips'][:4], prec['min_rho']))
            if lv == 'D1' and not charged:
                for c in ('TBX', 'EQW', 'NCT', 'C2F'):
                    b = boot({**P, c: N[c]})
                    res['boot_primary+' + c] = b
                    print('  boot primary+%s %s' % (c, b))
                b = boot({c: N[c] for c in ('TBX', 'EQW', 'NCT', 'C2F')})
                res['boot_new_only'] = b
                print('  boot new-only', b)
            out[tag] = res
    # machine axis within each new code
    print('\n===== machine axis (uncharged)')
    mach = {}
    for c in ('TBX', 'EQW', 'NCT', 'C2F'):
        r = k1({lv: mine(c, lv)['Phi'] for lv in ('D0', 'D1', 'D2')})
        mach[c] = r
        print('  %-4s tops %s min_rho %.3f flips %s' % (c, r['tops'], r['min_rho'], r['flips']))
    out['machine'] = mach
    # unit sensitivity
    print('\n===== unit sensitivity (D1)')
    us = {}
    for c in ('EQW', 'NCT'):
        r = k1({'U%d' % u: mine(c, 'D1', unit=u)['Phi'] for u in (2, 4, 8)})
        us[c] = r
        print('  %s tops %s min_rho %.3f' % (c, r['tops'], r['min_rho']))
    out['unit'] = us
    json.dump(out, open(os.path.join(HERE, 'results', 'k1_r1.json'), 'w'), indent=1)

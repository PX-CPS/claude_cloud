"""Fairness/circularity lens (r1): from saved per-system lengths L*_k(T) (miner off), how much of EC's value
and label rests on (a) family G (systems whose ground truth is BY CONSTRUCTION EL(T-V)), (b) the largest
systems; also which laws' Phi is carried by a single system.  Also: the D2M residual-string arithmetic."""
import sys, json, math, numpy as np
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1)
import analysis as AN, corpus as CP
from common import LAW9, gamma_len
R = PV1 + '/results/'
corp = {s.sid: s for s in CP.corpus()}
famG = [k for k, s in corp.items() if s.fam == 'G']
full, ecb = 511, 1 << LAW9.index('EC')
out = {}


def load(c, lv):
    return json.load(open(R + 'job_%s-%s-B16.json' % (c, lv)))


def vmid_EC(J, keep):
    """V_add/V_loo/V_mid of EC from per-system tables restricted to `keep` (law statements unchanged)."""
    Lk = J['Lk']
    LS = np.array(J['LS'])
    CC = np.array(J.get('CC') or [0.0] * 512)
    L = LS + CC + sum(np.array(Lk[s]) for s in keep)
    add = L[0] - L[ecb]
    loo = L[full ^ ecb] - L[full]
    return dict(add=float(add), loo=float(loo), mid=float(0.5 * (add + loo)))


for lv in ['D1', 'D2', 'D2M']:
    for c in ['C1', 'C2', 'C3', 'C4']:
        J = load(c, lv)
        dec = J['dec']
        tau = 2 * float(np.median([J['law_single'][f] for f in LAW9]))
        allk = sorted(J['Lk'])
        r_all = vmid_EC(J, allk)
        r_noG = vmid_EC(J, [k for k in allk if k not in famG])
        r_G = vmid_EC(J, famG)
        r_noBig = vmid_EC(J, [k for k in allk if k not in ('G04', 'G03', 'P11')])
        ch = (lv == 'D1')
        out['%s|%s' % (lv, c)] = dict(tau=tau, dec=dec, all=r_all, noG=r_noG, onlyG=r_G, no_G03_G04_P11=r_noBig)
        print('%s %s tau %5.1f dec %5.1f | EC V_mid all %7.1f  noFamG %7.1f  famG-only %7.1f  drop{G03,G04,P11} %7.1f%s'
              % (lv, c, tau, dec, r_all['mid'], r_noG['mid'], r_G['mid'], r_noBig['mid'],
                 ('   (charged: all %.1f noFamG %.1f)' % (r_all['mid'] - dec, r_noG['mid'] - dec)) if ch else ''))

# single-system concentration of Phi (D1, C1): share of the largest per-system contribution
print('\nD1 per-law concentration (miner off): law, Phi, top system and its share of sum_k phi(f,k)')
for c in ['C1', 'C4']:
    J = load(c, 'D1')
    ps = J['values']['phi_sys']; ls = J['values']['phi_ls']
    for f in LAW9:
        fi = LAW9.index(f)
        tot = sum(ps[k][fi] for k in ps)
        top = max(ps, key=lambda k: ps[k][fi])
        pos = sum(max(ps[k][fi], 0) for k in ps)
        out['conc|%s|%s' % (c, f)] = dict(Phi=tot + ls[fi], top=top, top_val=ps[top][fi], share_of_positive=ps[top][fi] / pos if pos > 0 else None)
        print('  %s %-4s Phi %7.1f  sum_k %7.1f  top %s %7.1f  share of positive mass %.2f' % (c, f, tot + ls[fi], tot, top, ps[top][fi], ps[top][fi] / pos if pos else float('nan')))

# D2M residual string arithmetic (C1): cost of Map(Dt(Dq(Lt,v^)) - Dq(Lt,q^)) per family-G system
print('\nD2M residual-string cost (C1) vs EC per-system Shapley:')
J = load('C1', 'D2M')
ps = J['values']['phi_sys']
for sid in ['G03', 'G04', 'G07']:
    s = corp[sid]
    scope = 2 * s.n + len(s.params) + 2 + 1        # state + params + 2 binders + 1 temp (Lt)
    leaf = 3 + math.log2(scope)
    func = 3 + math.log2(9)
    cost = func + (3 + gamma_len(1)) + func + 2 * func + (3 + gamma_len(1) + gamma_len(2) + 1) + 4 * leaf
    # extra leaf-scope cost of Lt's body under gradres_map (scope +3 vs the EL form's scope)
    print('  %s residual string ~%.1f bits; EC phi(k) = %.1f' % (sid, cost, ps[sid][LAW9.index('EC')]))
    out['resid|%s' % sid] = dict(string_bits=cost, phi=ps[sid][LAW9.index('EC')])
json.dump(out, open('/home/user/claude_cloud/research/pilot_e/verify_r1_fairness-circularity/decomp2.json', 'w'), indent=1)

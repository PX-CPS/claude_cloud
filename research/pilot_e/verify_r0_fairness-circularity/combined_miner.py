"""K2-collapse labels (spec section 8, primary setting D1, EC charged) under the combined fair
baseline F1+F7.  Same miner, same thresholds; gate controls reported."""
import os, sys, json, statistics
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
HERE = '/home/user/claude_cloud/research/pilot_e/verify_r0_fairness-circularity'
sys.path.insert(0, HERE)
import reps as R, lattice as LT
from common import LAW9
code = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else 'fair'
if variant == 'fair':
    import vec_baseline_patch  # noqa
    import fair_n2
    R.build_all = fair_n2.build_all_fair
j = LT.Job(code, 'D1', B=16, check=False, log=lambda *a: None).run()
FULL = 511
masks = [0, FULL] + [1 << i for i in range(9)] + [FULL ^ (1 << i) for i in range(9)]
Lon, Loff = {}, {}
for m in masks:
    r = j.mine(m, check=False)
    ec = (m >> LAW9.index('EC')) & 1
    Lon[m] = r['L_on'] + (j.dec if ec else 0.0)
    Loff[m] = r['L_off'] + (j.dec if ec else 0.0)
tau = 2 * statistics.median([j.law_single[f] for f in LAW9])
out = {}
for i, f in enumerate(LAW9):
    b = 1 << i
    vm = 0.5 * ((Loff[0] - Loff[b]) + (Loff[FULL ^ b] - Loff[FULL]))
    vp = 0.5 * ((Lon[0] - Lon[b]) + (Lon[FULL ^ b] - Lon[FULL]))
    if vm <= 0:
        lab = 'no-value'
    elif vp < tau or vp < 0.2 * vm:
        lab = 'abbreviation'
    elif vp >= tau and vp >= 0.5 * vm:
        lab = 'constraint'
    else:
        lab = 'mixed'
    out[f] = dict(V_mid=vm, Vplus_mid=vp, label=lab)
json.dump(dict(tau=tau, labels=out), open(os.path.join(HERE, 'res_miner_%s_%s.json' % (variant, code)), 'w'), indent=1)
print(code, variant, 'tau %.1f' % tau, {f: '%s %.0f/%.0f' % (d['label'], d['V_mid'], d['Vplus_mid']) for f, d in out.items()})

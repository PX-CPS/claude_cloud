"""Fairness probe: C2's frequencies are fitted on the EMPTY-theory (all-explicit) representations, i.e. the
code is tuned to the WITHOUT baseline.  K1-primary fires only through the C2 top-law flip (EC 503.9 < 0.9 x
UG 573.4).  Refit C2 on (a) the LAW9-theory representations, (b) the union of empty and LAW9 representations,
and re-run the K1-primary test against the saved C1/C3/C4 values (D1, uncharged, miner off).
Run: PYTHONHASHSEED=0 python3 c2_refit.py {full|union}"""
import os, sys, json
os.environ.setdefault('PYTHONHASHSEED', '0')
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import trees as TR, lattice as LT, analysis as AN
from common import LAW9
OUT = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1]


def fit_freq_alt(truth, level):
    J1 = LT.Job('C1', level, 16, check=False, log=lambda *a: None).run()
    trees = []
    masks = [511] if mode == 'full' else [0, 511]
    for m in masks:
        for s in J1.corp:
            rep, _ = J1.choice[(s.sid, m)]
            trees += [tid for _, tid in rep.items]
    TR.FREQ.fit(trees)
    return TR.FREQ.to_json()


LT.fit_freq = fit_freq_alt
J = LT.Job('C2', 'D1', 16, check=False, log=lambda *a: None).run()
v = J.values()
Phi = {'C2': {f: v['values'][f]['Phi'] for f in LAW9}}
for c in ['C1', 'C3', 'C4']:
    Jc = json.load(open(PV1 + '/results/job_%s-D1-B16.json' % c))
    Phi[c] = {f: Jc['values']['values'][f]['Phi'] for f in LAW9}
k1 = AN.k1_test(Phi, ['C1', 'C2', 'C3', 'C4'])
res = dict(mode=mode, freq=J.freq, Phi_C2=Phi['C2'], k1=dict(min_rho=k1['min_rho'], tops=k1['tops'], flips=k1['flips'], fired=k1['fired']))
print(mode, 'C2 Phi', {f: round(x, 1) for f, x in Phi['C2'].items()})
print('K1-primary: min rho %.3f tops %s fired %s flips %d' % (k1['min_rho'], k1['tops'], k1['fired'], len(k1['flips'])))
json.dump(res, open(os.path.join(OUT, 'c2_refit_%s.json' % mode), 'w'), indent=1)

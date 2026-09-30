"""EC at the PRIMARY K2-collapse setting (D1, charged, spec miner) on the corpus WITHOUT the 7 family-G
systems (whose ground truth is EL(T-V) by construction and which are closed to every other law by the
applicability matrix).  Also per-system EC add/loo (miner off) at D1 and D2M.
Run: PYTHONHASHSEED=0 python3 famP_only.py CODE"""
import os, sys, json
os.environ.setdefault('PYTHONHASHSEED', '0')
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import numpy as np
import lattice as LT, miner as MN, corpus as CP
from common import LAW9
OUT = os.path.dirname(os.path.abspath(__file__))
code = sys.argv[1]
J = LT.Job(code, 'D1', 16, check=False, log=lambda *a: None).run()
full, ec = 511, 1 << LAW9.index('EC')
famP = [s for s in J.corp if s.fam == 'P']
Lon = {}
for m in (0, ec, full ^ ec, full):
    sysreps = [(s, J.choice[(s.sid, m)][0], J.choice[(s.sid, m)][1]) for s in famP]
    extra = float(J.LS[m] + J.CC[m])
    mn = MN.Miner(code, 16, 'D1', sysreps, extra, log=lambda *a: None)
    Lon[m] = mn.run()
Loff = {m: float(J.LS[m] + J.CC[m] + sum(J.Lk[s.sid][m] for s in famP)) for m in (0, ec, full ^ ec, full)}
dec = J.dec
res = {}
for tag, L in (('off', Loff), ('on', Lon)):
    add = L[0] - L[ec] - dec
    loo = L[full ^ ec] - L[full] - dec
    res[tag] = dict(add=add, loo=loo, mid=0.5 * (add + loo))
import statistics
tau = 2 * statistics.median([J.law_single[f] for f in LAW9])
import analysis as AN
lab = AN.classify(res['off']['mid'], res['on']['mid'], tau)
print(code, 'family-P only, D1 charged: V_mid %.1f  V+_mid %.1f  tau %.1f  label %s' % (res['off']['mid'], res['on']['mid'], tau, lab))
# per-system EC add / loo (miner off), full corpus
per = {}
for s in J.corp:
    a = J.Lk[s.sid]
    per[s.sid] = dict(add=float(a[0] - a[ec]), loo=float(a[full ^ ec] - a[full]))
print('per-system EC add/loo (D1, miner off):', {k: (round(v['add'], 1), round(v['loo'], 1)) for k, v in per.items() if abs(v['add']) + abs(v['loo']) > 2})
json.dump(dict(code=code, famP_only=res, tau=tau, dec=dec, label=lab, per_system=per),
          open(os.path.join(OUT, 'famP_only_%s.json' % code), 'w'), indent=1)

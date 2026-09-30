"""Fairness probe: the K2-collapse miner forbids exactly the pattern that would abbreviate EC at D2M.
R7(b) excludes any pattern containing a Map binder, and is_local() treats the canonical bound variables
_Bq/_Bv (identical symbols in every system) as system-local, so they can never be concrete template leaves.
Templates are expanded textually (U6 checks identity of the expanded trees), so a pattern
Map(Dt(Dq(h0,_Bv)) - Dq(h0,_Bq)) with h0 := the local Lagrangian temp is capture-free.
Here: 'b' labels are global and Map patterns are allowed; U5/U6 are re-checked."""
import os, sys, json, time
os.environ.setdefault('PYTHONHASHSEED', '0')
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import miner as MN, lattice as LT, analysis as AN
from common import LAW9
code, level, patched = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
if patched:
    MN.is_local = lambda label: label[0] in ('t', 'o', 'c', 'h')
    MN.Miner.has_map = lambda self, i, memo=None: False
J = LT.Job(code, level, 16, check=True, log=lambda *a: None).run()
full = 511; ec = 1 << LAW9.index('EC')
Lon, res = {}, {}
for m in (0, ec, full ^ ec, full):
    r = J.mine(m, check=True)
    Lon[m] = r['L_on']
    res[m] = dict(L_off=r['L_off'], L_on=r['L_on'], U5=r['U5_maxerr'], U6=r['U6_roundtrip'], ntpl=r['ntemplates'])
add_off = J.L[0] - J.L[ec]; loo_off = J.L[full ^ ec] - J.L[full]
add_on = Lon[0] - Lon[ec]; loo_on = Lon[full ^ ec] - Lon[full]
out = dict(code=code, level=level, patched=patched, runs=res,
           Vmid_off=0.5 * (add_off + loo_off), Vplus=dict(add=add_on, loo=loo_on, mid=0.5 * (add_on + loo_on)))
print(json.dumps(out))
json.dump(out, open('/home/user/claude_cloud/research/pilot_e/verify_r1_fairness-circularity/map_miner_%s_%s_%d.json' % (code, level, patched), 'w'), indent=1)

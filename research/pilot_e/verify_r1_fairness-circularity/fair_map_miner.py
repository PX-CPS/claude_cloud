"""Fairness probe (r1): the K2-collapse instrument is barred from abstracting the ONE string that carries
EC's residual value at D2M.  At D2M the no-EC baseline of a family-G system is
    Lt = T - V ;  Map( Dt(Dq(Lt, v^)) - Dq(Lt, q^) )
and the EC form is  L = T - V  (+ flag).  The difference per system is the fixed 11-node residual string.
The pilot's miner (a) forbids any pattern containing Map (R7b) and (b) labels the binders ('b', sid, ..)
system-locally, so the string can never be an exact cross-system repeat.

Here the binders get one corpus-wide label ('b','ALL',..) (the decoder binds the same label), patterns
containing Map are allowed, and Map patterns are always scored.  U5 (numeric decoding of the mined
representations) and U6 (round trip) are re-checked.  Everything else is the pilot's own code.

Run:  PYTHONHASHSEED=0 python3 fair_map_miner.py CODE   (CODE in C1..C4), level D2M.
"""
import os, sys, json, time
os.environ.setdefault('PYTHONHASHSEED', '0')
PV1 = '/home/user/claude_cloud/research/pilot_e/pilot_v1'
sys.path.insert(0, PV1); os.chdir(PV1)
import trees as TR, miner as MN, lattice as LT, analysis as AN, semantics as SM, reps as R
from common import LAW9

code = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else 'fair'     # 'fair' or 'spec'
OUT = os.path.dirname(os.path.abspath(__file__))

if variant == 'fair':
    _orig_label = TR.local_label_of

    def local_label_of(sym, sid):
        n = sym.name
        if n.startswith('_B'):
            return ('b', 'ALL', n[2:])
        return _orig_label(sym, sid)
    TR.local_label_of = local_label_of

    class SysProxy:
        """decoder binds Map variables under ('b', sysm.sid, ..); make sid read 'ALL' there only."""
        def __init__(self, s):
            object.__setattr__(self, '_s', s)

        def __getattr__(self, k):
            return 'ALL' if k == 'sid' else getattr(self._s, k)

    MN.is_local = lambda label: label[0] in ('t', 'o', 'c', 'h')
    MN.Miner.has_map = lambda self, i, memo=None: False

    _cands = MN.Miner.candidates

    def candidates(self):
        """spec ranking, but Map-containing patterns are always among the scored ones."""
        ranked = _cands(self)
        # regenerate the full candidate dict to find Map patterns (cheap: reuse the method with big topk)
        old = self.topk
        self.topk = 10 ** 9
        allc = _cands(self)
        self.topk = old
        maps = [p for p in allc if self._has_map_real(p)]
        return maps + [p for p in ranked if p not in set(maps)]

    def _has_map_real(self, i, memo={}):
        r = memo.get(i)
        if r is None:
            kind, label, ch = TR.NODES[i]
            r = (kind == 'Func' and label == 'Map') or any(self._has_map_real(c) for c in ch)
            memo[i] = r
        return r
    MN.Miner._has_map_real = _has_map_real
    MN.Miner.candidates = candidates

t0 = time.time()
J = LT.Job(code, 'D2M', 16, check=False, log=lambda *a: None)
J.decoders = {s.sid: SM.Decoder(s, J.truth) for s in J.corp}
if variant == 'fair':
    for d in J.decoders.values():
        d.sysm = SysProxy(d.sysm)
J.check = True
J.run()
maxerr_reps = max((e[-1] for e in J.errors), default=0.0)
full = 511
ec = 1 << LAW9.index('EC')
Lon, runs = {}, {}
for m in (0, ec, full ^ ec, full):
    r = J.mine(m, check=True)
    Lon[m] = r['L_on']
    runs[m] = dict(L_off=r['L_off'], L_on=r['L_on'], U5=r['U5_maxerr'], U6=r['U6_roundtrip'], ntpl=r['ntemplates'],
                   history=r['history'])
vals_off = J.values()['values']
add_on = Lon[0] - Lon[ec]
loo_on = Lon[full ^ ec] - Lon[full]
vplus = dict(add=add_on, loo=loo_on, mid=0.5 * (add_on + loo_on))
law_single = J.law_single
import statistics
tau = 2 * statistics.median([law_single[f] for f in LAW9])
lab = AN.classify(vals_off['EC']['mid'], vplus['mid'], tau)
out = dict(code=code, level='D2M', variant=variant, maxerr_reps=maxerr_reps, tau=tau, runs=runs,
           V_mid_EC=vals_off['EC'], Vplus_EC=vplus, Phi={f: vals_off[f]['Phi'] for f in LAW9}, label_EC=lab,
           time=time.time() - t0)
print(json.dumps({k: out[k] for k in ('code', 'variant', 'maxerr_reps', 'tau', 'label_EC', 'time')}))
print('EC V_mid %.1f (add %.1f loo %.1f)   V+_mid %.1f (add %.1f loo %.1f)' % (
    vals_off['EC']['mid'], vals_off['EC']['add'], vals_off['EC']['loo'], vplus['mid'], vplus['add'], vplus['loo']))
print('runs', {m: (round(r['L_off'], 1), round(r['L_on'], 1), r['U5'], r['U6'], r['ntpl']) for m, r in runs.items()})
json.dump(out, open(os.path.join(OUT, 'fair_map_miner_%s_%s.json' % (code, variant)), 'w'), indent=1)

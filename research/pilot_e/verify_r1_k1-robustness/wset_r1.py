#!/usr/bin/env python3
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'; os.execv(sys.executable, [sys.executable] + sys.argv)
ws = sys.argv[1]; lv = sys.argv[2]
os.environ['R1_WSET'] = ws
sys.path.insert(0, HERE)
import codes_r1, lattice as LT
from common import LAW9
j = LT.Job('EQW', lv, B=16, check=False, log=lambda *a: None).run()
out = {}
for ch in (False, True):
    v = j.values(charged=ch)
    out['charged' if ch else 'uncharged'] = {f: v['values'][f]['Phi'] for f in LAW9}
p = os.path.join(HERE, 'results', 'eqw_wsets.json')
R = json.load(open(p)) if os.path.exists(p) else {}
R[ws + '-' + lv] = out
json.dump(R, open(p, 'w'), indent=1)
print(ws, lv, {f: round(x, 1) for f, x in out['uncharged'].items()})

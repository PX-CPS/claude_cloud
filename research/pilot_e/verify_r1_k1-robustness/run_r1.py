#!/usr/bin/env python3
"""Recompute Track-G Shapley values under new codes with the pilot's own machinery (miner off)."""
import os, sys, json, multiprocessing as mp, time
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import codes_r1  # noqa: patches the pilot modules
from common import LAW9
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)
FULL = (1 << 9) - 1


def work(args):
    code, level, B, unit = args
    os.environ['R1_UNIT'] = str(unit)
    codes_r1.UNIT = float(unit)
    import lattice as LT, trees as TR
    t0 = time.time()
    if code == 'C2F':
        j1 = LT.Job('C1', level, B=16, check=False, log=lambda *a: None).run()
        trees = []
        for s in j1.corp:
            rep, _ = j1.choice[(s.sid, FULL)]
            trees += [tid for _, tid in rep.items]
        TR.FREQ.fit(trees)
    j = LT.Job(code, level, B=B, check=(level == 'D1'), log=lambda *a: None).run()
    res = dict(code=code, level=level, B=B, unit=unit)
    for ch in (False, True):
        v = j.values(charged=ch)
        res['charged' if ch else 'uncharged'] = dict(
            Phi={f: v['values'][f]['Phi'] for f in LAW9}, add={f: v['values'][f]['add'] for f in LAW9},
            loo={f: v['values'][f]['loo'] for f in LAW9}, L0=v['L0'], Lfull=v['Lfull'], gap=v['efficiency_gap'],
            phi_sys=v['phi_sys'], phi_ls=v['phi_ls'])
    res['law_single'] = j.law_single
    res['dec'] = j.dec
    res['max_err'] = max((e[-1] for e in j.errors), default=None)
    res['choice_full'] = {s.sid: j.choice[(s.sid, FULL)][0].form + ':' + j.choice[(s.sid, FULL)][0].meta['variant'] for s in j.corp}
    res['choice_empty'] = {s.sid: j.choice[(s.sid, 0)][0].form + ':' + j.choice[(s.sid, 0)][0].meta['variant'] for s in j.corp}
    if code == 'C2F':
        res['freq'] = TR.FREQ.to_json()
    res['time'] = time.time() - t0
    return res


if __name__ == '__main__':
    jobs = []
    for lv in ('D0', 'D1', 'D2'):
        for c in ('C1', 'TBX', 'EQW', 'NCT', 'C2F'):
            jobs.append((c, lv, 16, 4))
        for B in (4, 64):
            jobs.append(('C4', lv, B, 4))
    for u in (2, 8):
        for c in ('EQW', 'NCT'):
            jobs.append((c, 'D1', 16, u))
    only = sys.argv[1:]
    if only:
        jobs = [j for j in jobs if '%s-%s' % (j[0], j[1]) in only]
    path = os.path.join(OUT, 'phi_r1.json')
    R = json.load(open(path)) if os.path.exists(path) else {}
    with mp.get_context('fork').Pool(4) as pool:
        for r in pool.imap_unordered(work, jobs):
            key = '%s-%s-B%d-U%g' % (r['code'], r['level'], r['B'], r['unit'])
            R[key] = r
            top = max(r['uncharged']['Phi'], key=r['uncharged']['Phi'].get)
            print('done', key, 'maxerr', r['max_err'], 'gap %.1e' % r['uncharged']['gap'], 'top', top,
                  {f: round(x, 1) for f, x in r['uncharged']['Phi'].items()}, '%.0fs' % r['time'], flush=True)
            json.dump(R, open(path, 'w'))

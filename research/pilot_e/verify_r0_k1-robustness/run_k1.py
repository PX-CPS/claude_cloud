#!/usr/bin/env python3
"""Recompute Track-G Shapley rankings under the pilot's primary codes and the new codes; K1 tests."""
import os, sys, json, itertools, multiprocessing as mp
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import newcodes  # noqa: patches
from common import LAW9, PHYS7
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)


def work(args):
    code, level, B = args
    import newcodes  # noqa
    import lattice as LT
    j = LT.Job(code, level, B=B, check=(code in newcodes.NEW and level == 'D1'), log=lambda *a: None).run()
    res = dict(code=code, level=level, B=B)
    for ch in (False, True):
        v = j.values(charged=ch)
        res['charged' if ch else 'uncharged'] = dict(Phi={f: v['values'][f]['Phi'] for f in LAW9},
                                                     add={f: v['values'][f]['add'] for f in LAW9},
                                                     L0=v['L0'], Lfull=v['Lfull'], gap=v['efficiency_gap'],
                                                     phi_sys=v['phi_sys'], phi_ls=v['phi_ls'])
    res['law_single'] = j.law_single
    res['dec'] = j.dec
    res['max_err'] = max((e[-1] for e in j.errors), default=None)
    return res


if __name__ == '__main__':
    codes = ['C1', 'C2', 'C3', 'C4', 'C6', 'C7', 'C9', 'C10', 'C11']
    jobs = [(c, lv, 16) for c in codes for lv in ('D0', 'D1', 'D2')]
    R = {}
    with mp.get_context('fork').Pool(4) as pool:
        for r in pool.imap_unordered(work, jobs):
            R['%s-%s' % (r['code'], r['level'])] = r
            print('done', r['code'], r['level'], 'maxerr', r['max_err'], 'gap', r['uncharged']['gap'], flush=True)
    json.dump(R, open(os.path.join(OUT, 'phi_all.json'), 'w'), indent=0)

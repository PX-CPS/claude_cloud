"""Fairness repairs of the Track-G lattice (does NOT modify pilot_v1 files; monkeypatches in-process).

Variants (flags):
  F  : N2-fair baseline. Without N2, the coder may also write the N2 representation with each body's
       output divided by its mass, a_i = (sum_j F_ij)/m_i (a plain tree, no law needed).  min over both.
  S  : side-information charge.  The N2 decoder reads body i's mass and the PC decoder reads all masses
       (semantics.decode: point[s.masses[bi]]) for free.  Charge a mass pointer per body:
       log2|params| (C1/C2/C4) or 2 chars*log2(95) (C3).  PC pays it only when N2 is absent.
  X  : law statements CSE'd like models (sympy.cse per statement item; temps join the formal scope).
  Q  : EC decoder charged (spec 1.5 rules) -- already available as values(charged=True).
"""
import sys, os, math, json, itertools, time
os.environ.setdefault('PYTHONHASHSEED', '0')
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import numpy as np, sympy as sp
import corpus as CP, reps as R, lattice as LT, laws as LW, trees as TR, analysis as AN
from common import LAW9, LOG95, gamma_len, builtins_for

FLAGS = set(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else set()
CODES = sys.argv[2].split(',') if len(sys.argv) > 2 else ['C1', 'C2', 'C3', 'C4']
LEVEL = sys.argv[3] if len(sys.argv) > 3 else 'D1'
CHECK = (len(sys.argv) > 4 and sys.argv[4] == 'check')

_orig_build_all = R.build_all
_orig_law_cost = LW.law_cost


def factored(sysm, S, ct):
    items, meta = R.build_newton(sysm, frozenset(S) | {'N2'}, ct)
    outs_idx = [i for i, (l, _) in enumerate(items) if l is None]
    new = list(items)
    for bi, idx in enumerate(outs_idx):
        l, e = items[idx]
        new[idx] = (None, e / sysm.M(bi + 1))
    meta = dict(meta); meta['n2'] = False; meta['variant'] = 'factored'
    return new, meta


def side_bits(sysm, code, nb):
    if code == 'C3':
        return nb * 2 * LOG95
    return nb * math.log2(max(len(sysm.params), 2))


def build_all(sysm, form, S, level, code, B, truth):
    out = _orig_build_all(sysm, form, S, level, code, B, truth)
    if sysm.fam == 'P' and form == 'newton':
        if 'F' in FLAGS and 'N2' not in S:
            ct = R.make_cost_term(sysm, form, S, level, code, B)
            out.append(R.realize(sysm, *factored(sysm, S, ct), S, level, code, B))
        if 'S' in FLAGS:
            for r in out:
                nb = 0
                if r.meta.get('n2'):
                    nb = sysm.n
                elif r.meta.get('pc') and code != 'C4':      # C4 already charges PC mass ratios (C10)
                    nb = sysm.n
                r.cost += side_bits(sysm, code, nb)
    return out


def law_cost_cse(f, T, used_templates, code, level):
    nlib = len(R.lib_names(T)) + 5
    nb = len(builtins_for(level))
    tot = 0.0
    for e, fm in LW.law_items(f, 'DIST' in T, used_templates):
        repl, red = sp.cse([e], symbols=sp.numbered_symbols('_s'), optimizations='basic')
        if not repl:
            tot += LW.item_cost(e, fm, code, nlib, nb)
            continue
        formals = list(fm) + [s for s, _ in repl]
        holes = {s: j for j, s in enumerate(formals)}
        c = gamma_len(len(fm) + 1) + gamma_len(len(repl) + 2)
        for s_, body in list(repl) + [(None, red[0])]:
            tid = TR.from_sympy(body, 'LAW', law_mode=True, holes=holes)
            if code == 'C3':
                c += TR.c3_len(tid) * LOG95 + 3 * LOG95
            else:
                tc = 'C1' if code in ('C4', 'C5') else code
                c += TR.tree_cost(tid, tc, {'hole': len(formals)}, nlib, nb)
        tot += min(c, LW.item_cost(e, fm, code, nlib, nb))
    return tot


R.build_all = build_all
LT.R.build_all = build_all
if 'X' in FLAGS:
    LW.law_cost = law_cost_cse
    LT.LW.law_cost = law_cost_cse

res = {}
for code in CODES:
    t0 = time.time()
    j = LT.Job(code, LEVEL, B=16, check=CHECK, log=lambda *a: None).run()
    maxerr = max((e[-1] for e in j.errors), default=0.0)
    for ch in (False, True):
        v = j.values(charged=ch)
        res['%s%s' % (code, '-chg' if ch else '')] = {f: v['values'][f]['Phi'] for f in LAW9}
        if not ch:
            res[code + '-stmt'] = dict(j.law_single)
            res[code + '-phi_sys'] = v['phi_sys']
            res[code + '-mid'] = {f: v['values'][f]['mid'] for f in LAW9}
    print(code, LEVEL, 'flags', sorted(FLAGS), 'U5 maxerr', maxerr, 'time %.0fs' % (time.time() - t0), flush=True)
    print('  Phi   :', {f: round(res[code][f], 1) for f in LAW9})
    print('  Phichg:', {f: round(res[code + '-chg'][f], 1) for f in LAW9})
    print('  stmt  :', {f: round(res[code + '-stmt'][f], 1) for f in LAW9})
if len(CODES) > 1:
    for tag in ('', '-chg'):
        Phi = {c: res[c + tag] for c in CODES}
        k = AN.k1_test(Phi, CODES)
        print('K1-primary%s: min rho %.3f tops %s flips %d fired %s' % (tag, k['min_rho'], k['tops'], len(k['flips']), k['fired']))
name = 'fair_%s_%s_%s.json' % (''.join(sorted(FLAGS)) or 'none', LEVEL, '-'.join(CODES))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), 'w'), indent=1)

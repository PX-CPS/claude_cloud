#!/usr/bin/env python3
"""Single entry point: regenerates every number of pilot E-v1 (repair round r1).

    python3 run_all.py            # full run on 4 cores

Writes pilot_v1/results/*.json, pilot_v1/results/run.log, and
research/pilot_e/results.json + research/pilot_e/results.md.

Repair round r1 (see DEVIATIONS.txt, section R): the main pipeline carries the bug fixes and the
added fair-baseline alternatives; the pre-repair numbers are kept in pilot_v1/orig_v1/ and a variant
with the added baseline alternatives switched off ('-noaddbase') is re-run and reported alongside.
"""
import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json, time, math, itertools, multiprocessing as mp, lzma, hashlib, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import LAW9, PHYS7, PRIMARY, EXTRA_CODES, LEVELS, SEED, RESULTS, HERE
import corpus as CP

T_START = time.time()
LOGF = os.path.join(RESULTS, 'run.log')
PILOT_E = os.path.dirname(HERE)
FULL = (1 << 9) - 1
DEFAULT_CFG = dict(common.CFG)
NOADD = dict(n2_factored=False, lc_vec=False)
ABLATIONS = {'n2_factored': dict(n2_factored=False), 'lc_vec': dict(lc_vec=False), 'stmt_cse': dict(stmt_cse=False),
             'side_info': dict(side_info=False), 'gradres_scope': dict(gradres_scope=False),
             'all_off': dict(n2_factored=False, lc_vec=False, stmt_cse=False, side_info=False, gradres_scope=False)}


def log(*a):
    msg = '[%7.1fs] ' % (time.time() - T_START) + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOGF, 'a') as fh:
        fh.write(msg + '\n')


def miner_masks():
    ms = [0, FULL] + [1 << i for i in range(9)] + [FULL ^ (1 << i) for i in range(9)]
    return ms


def apply_cfg(cfg):
    common.CFG.clear()
    common.CFG.update(DEFAULT_CFG)
    common.CFG.update(cfg or {})


# ---------------------------------------------------------------- worker
def job_worker(args):
    code, level, B, miner, tag, cfg = args
    apply_cfg(cfg)
    import lattice as LT, trees as TR
    t0 = time.time()
    lines = []
    j = LT.Job(code, level, B=B, check=True, log=lambda *a: lines.append(' '.join(map(str, a)))).run()
    out = dict(tag=tag, code=code, level=level, B=B, cfg=dict(common.CFG), nreps=len(j.reps))
    if code == 'C2':
        out['freq'] = TR.FREQ.to_json()
    out['values'] = j.values(charged=False)
    out['values_charged'] = j.values(charged=True)
    out['law_single'] = j.law_single
    out['dec'] = j.dec
    errs = {}
    for sid, form, S, variant, err in j.errors:
        k = form + ':' + variant
        errs[k] = max(errs.get(k, 0.0), err)
    out['U5_max_err_by_form'] = errs
    out['U4_max_err_el'] = max([e[-1] for e in j.errors if e[1] == 'el'], default=0.0)
    out['U3_gap'] = out['values']['efficiency_gap']
    out['U3_gap_charged'] = out['values_charged']['efficiency_gap']
    out['Lk'] = {sid: [float(x) for x in arr] for sid, arr in j.Lk.items()}
    out['LS'] = [float(x) for x in j.LS]
    out['CC'] = [float(x) for x in j.CC]
    out['choice_summary'] = {m: {s.sid: dict(form=j.choice[(s.sid, m)][0].form,
                                            variant=j.choice[(s.sid, m)][0].meta['variant'],
                                            cost=j.choice[(s.sid, m)][0].cost, flag=j.choice[(s.sid, m)][1],
                                            side_bits=getattr(j.choice[(s.sid, m)][0], 'side_bits', 0.0))
                                 for s in j.corp} for m in (0, FULL) + tuple(1 << i for i in range(9))}
    if code == 'C4':
        out['pos_off'] = {0: j.pos_counts(0), FULL: j.pos_counts(FULL)}
    if 'v1' in miner:
        out['miner'] = {}
        for m in miner_masks():
            out['miner'][m] = j.mine(m)
    if 'v2' in miner:
        out['miner_v2'] = {}
        for m in miner_masks():
            out['miner_v2'][m] = j.mine(m, v2=True)
    if tag == 'C1-D1-B16':
        c8 = []
        for m in range(1 << 9):
            txt = j.corpus_text(m).encode()
            c8.append(8.0 * len(lzma.compress(txt, preset=9 | lzma.PRESET_EXTREME)))
        out['C8_L_all'] = c8
    out['time'] = time.time() - t0
    out['log'] = lines
    return out


def miner_shapley_worker(args):
    code, level, B, masks = args
    apply_cfg({})
    import lattice as LT
    j = LT.Job(code, level, B=B, check=False, log=lambda *a: None).run()
    res = {}
    for m in masks:
        r = j.mine(m, check=False)
        res[m] = r['L_on']
    return code, level, res


def surplus_worker(args):
    code, level = args
    apply_cfg({})
    import lattice as LT, scaling as SC
    truth = dict(CP.build_truth())
    truth.update(SC.truth())
    systems = [s for s in CP.corpus() if 'EC' in s.applicable] + SC.systems()
    j = LT.Job(code, level, B=16, check=True, log=lambda *a: None, corp=systems, truth=truth)
    out = {}
    for s in systems:
        r = j.ec_surplus(s)
        r.update(n=s.dof, pairs=s.n_pairs(), P=len(s.params), fam=s.fam, sid=s.sid)
        out[s.sid] = r
    errs = [e[-1] for e in j.errors]
    return code, level, out, max(errs) if errs else 0.0


# scaling-series points that are structurally identical to corpus systems (R14: removed in 7.3)
SCALING_DUPLICATES = {'SNL1': 'G01', 'SNL3': 'G04', 'SNB2': 'P09', 'SCH4': 'P06'}


# ---------------------------------------------------------------- main
def main():
    open(LOGF, 'w').close()
    log('pilot E-v1 (repair r1) start; SPEC sha256:', open(os.path.join(HERE, 'SPEC.sha256')).read().split()[0])
    import tests, analysis as AN, trackr as TRR, lattice as LT, laws as LW, reps as REPS
    apply_cfg({})
    truth = CP.build_truth()
    RES = dict(spec_sha256=open(os.path.join(HERE, 'SPEC.sha256')).read().split()[0], cfg_main=dict(common.CFG))
    # ---- static unit tests
    ut = {}
    ut['U1'] = tests.U1()
    ut['U7'] = tests.U7()
    ut['U8'] = tests.U8()
    log('U1', ut['U1'][0], 'U7', ut['U7'][0], 'U8', ut['U8'][0], ut['U8'][1])
    RES['dec_rules_cse_temps'] = LW.dec_rules_cse_check()
    # ---- Track G jobs
    jobs = []
    for level in LEVELS:
        for code in PRIMARY:
            jobs.append((code, level, 16, 'v1v2' if level in ('D1', 'D2') else 'v1', '%s-%s-B16' % (code, level), {}))
        jobs.append(('C4', level, 8, '', 'C4-%s-B8' % level, {}))
        jobs.append(('C4', level, 32, '', 'C4-%s-B32' % level, {}))
        jobs.append(('C5', level, 16, '', 'C5-%s-B16' % level, {}))
        for code in EXTRA_CODES:
            jobs.append((code, level, 16, '', '%s-%s-B16' % (code, level), {}))
        for code in PRIMARY:
            jobs.append((code, level, 16, 'v1' if level == 'D1' else '', '%s-%s-B16-noaddbase' % (code, level), NOADD))
    for code in PRIMARY:
        jobs.append((code, 'D2M', 16, 'v1', '%s-D2M-B16' % code, {}))
    jobs.append(('C1', 'D1', 16, 'v1', 'C1-D1-B16-dup', {}))
    for name, cfg in ABLATIONS.items():
        jobs.append(('C1', 'D1', 16, '', 'C1-D1-B16-abl-%s' % name, cfg))
    # heaviest first
    jobs.sort(key=lambda x: (-len(x[3]), x[4]))
    J = {}
    with mp.get_context('fork').Pool(4) as pool:
        for out in pool.imap_unordered(job_worker, jobs):
            J[out['tag']] = out
            log('job', out['tag'], 'done in %.0fs' % out['time'], 'nreps', out['nreps'])
            with open(os.path.join(RESULTS, 'job_%s.json' % out['tag']), 'w') as fh:
                json.dump(out, fh, indent=0, default=str)
    apply_cfg({})
    # ---- dynamic unit tests
    a, b = J['C1-D1-B16'], J['C1-D1-B16-dup']
    u2 = (a['values']['L_all'] == b['values']['L_all']) and \
         all(a['miner'][m]['L_on'] == b['miner'][m]['L_on'] for m in a['miner'])
    ut['U2'] = (bool(u2), dict(compared='C1-D1 L(S|T) for 512 theories + 20 miner-on totals, two processes'))
    gaps = {t: max(abs(o['U3_gap']), abs(o['U3_gap_charged'])) for t, o in J.items()}
    ut['U3'] = (max(gaps.values()) < 1e-6, dict(max_gap=max(gaps.values())))
    u4 = max(o['U4_max_err_el'] for o in J.values())
    ut['U4'] = (u4 < 1e-9, dict(max_rel_err_EL=u4))
    u5 = max(max(o['U5_max_err_by_form'].values()) for o in J.values())
    u5m = max([r['U5_maxerr'] for o in J.values() for r in o.get('miner', {}).values()], default=0.0)
    u5m2 = max([r['U5_maxerr'] for o in J.values() for r in o.get('miner_v2', {}).values()], default=0.0)
    ut['U5'] = (u5 < 1e-9 and u5m < 1e-9 and u5m2 < 1e-9,
                dict(max_rel_err_reps=u5, max_rel_err_miner=u5m, max_rel_err_miner_v2=u5m2,
                     by_form={t: o['U5_max_err_by_form'] for t, o in J.items()}))
    u6 = all(r['U6_roundtrip'] for o in J.values() for r in o.get('miner', {}).values())
    u6b = all(r['U6_roundtrip'] for o in J.values() for r in o.get('miner_v2', {}).values())
    ut['U6'] = (bool(u6 and u6b), dict(n_runs=sum(len(o.get('miner', {})) for o in J.values()),
                                       n_runs_v2_modAC=sum(len(o.get('miner_v2', {})) for o in J.values()),
                                       v1_exact=bool(u6), v2_mod_AC=bool(u6b)))
    RES['unit_tests'] = {k: dict(passed=bool(v[0]), detail=v[1]) for k, v in ut.items()}
    log('unit tests:', {k: bool(v[0]) for k, v in ut.items()})
    if not all(v[0] for v in ut.values()):
        log('UNIT TEST FAILURE -> stop before statistics')
        json.dump(RES, open(os.path.join(RESULTS, 'unit_tests.json'), 'w'), indent=1, default=str)
        sys.exit(1)
    # ---- Track R
    lawc1 = {f: J['C1-D1-B16']['law_single'][f] for f in LAW9}
    tr = TRR.run_trackR(truth, lawc1, log=log)
    json.dump(tr, open(os.path.join(RESULTS, 'trackR.json'), 'w'), indent=1, default=str)
    # ================================================================ statistics
    corp = CP.corpus()
    V = lambda tag, charged=False: {f: J[tag]['values_charged' if charged else 'values']['values'][f] for f in LAW9}
    Phi = lambda tag, charged=False: {f: V(tag, charged)[f]['Phi'] for f in LAW9}
    # ---- K1 (decisive: spec definitions, uncharged, C1-C4)
    K1 = {}
    K1['primary'] = AN.k1_test({c: Phi('%s-D1-B16' % c) for c in PRIMARY}, PRIMARY)
    K1['machine'] = AN.k1_test({lv: Phi('C1-%s-B16' % lv) for lv in LEVELS}, LEVELS)
    K1['precision'] = AN.k1_test({b: Phi('C4-D1-B%d' % b) for b in (8, 16, 32)}, [8, 16, 32], crit_a=False)
    K1['precision_all_levels'] = {lv: AN.k1_test({b: Phi('C4-%s-B%d' % (lv, b)) for b in (8, 16, 32)}, [8, 16, 32], crit_a=False)
                                  for lv in LEVELS}
    K1['fired'] = bool(K1['primary']['fired'] or K1['machine']['fired'] or K1['precision']['fired'])
    K1['subtests_fired'] = [k for k in ('primary', 'machine', 'precision') if K1[k]['fired']]
    # reported
    K1['primary_7law'] = AN.k1_test({c: Phi('%s-D1-B16' % c) for c in PRIMARY}, PRIMARY, laws=PHYS7)
    K1['primary_charged'] = AN.k1_test({c: Phi('%s-D1-B16' % c, True) for c in PRIMARY}, PRIMARY)
    K1['machine_charged'] = AN.k1_test({lv: Phi('C1-%s-B16' % lv, lv != 'D2') for lv in LEVELS}, LEVELS)
    K1['machine_D2M'] = AN.k1_test({lv: Phi('C1-%s-B16' % lv) for lv in ('D0', 'D1', 'D2M')}, ['D0', 'D1', 'D2M'])
    K1['primary_by_level'] = {lv: AN.k1_test({c: Phi('%s-%s-B16' % (c, lv)) for c in PRIMARY}, PRIMARY)
                              for lv in LEVELS + ['D2M']}
    K1['machine_by_code'] = {c: AN.k1_test({lv: Phi('%s-%s-B16' % (c, lv)) for lv in LEVELS}, LEVELS) for c in PRIMARY}
    K1['C5_vs_primary'] = {c: AN.spearman([Phi('C5-D1-B16')[f] for f in LAW9], [Phi('%s-D1-B16' % c)[f] for f in LAW9])
                           for c in PRIMARY}
    ALLC = PRIMARY + EXTRA_CODES
    K1['extended'] = {}
    for lv in LEVELS:
        for ch in ((False, True) if lv != 'D2' else (False,)):
            key = '%s_%s' % (lv, 'charged' if ch else 'uncharged')
            PhiX = {c: Phi('%s-%s-B16' % (c, lv), ch) for c in ALLC}
            K1['extended'][key] = dict(all=AN.k1_test(PhiX, ALLC),
                                       each_extra_vs_primary={n: AN.k1_test(PhiX, PRIMARY + [n]) for n in EXTRA_CODES})
    K1['extended_machine'] = {c: AN.k1_test({lv: Phi('%s-%s-B16' % (c, lv)) for lv in LEVELS}, LEVELS) for c in EXTRA_CODES}
    K1['noaddbase_primary'] = AN.k1_test({c: Phi('%s-D1-B16-noaddbase' % c) for c in PRIMARY}, PRIMARY)
    K1['noaddbase_machine'] = AN.k1_test({lv: Phi('C1-%s-B16-noaddbase' % lv) for lv in LEVELS}, LEVELS)
    # C8
    c8 = np.array(J['C1-D1-B16']['C8_L_all'])
    phi8 = LT.shapley_from_table(c8)
    Phi8 = dict(zip(LAW9, [float(x) for x in phi8]))
    K1['C8'] = dict(Phi=Phi8, L0=float(c8[0]), Lfull=float(c8[FULL]),
                    spearman={c: AN.spearman([Phi8[f] for f in LAW9], [Phi('%s-D1-B16' % c)[f] for f in LAW9]) for c in PRIMARY},
                    top=max(LAW9, key=lambda f: Phi8[f]))
    # bootstrap, leave-one-system-out, per-system normalisation
    boot = {}
    loso = {}
    for name, tags, codes in (('primary', {c: '%s-D1-B16' % c for c in PRIMARY}, PRIMARY),
                              ('machine', {lv: 'C1-%s-B16' % lv for lv in LEVELS}, LEVELS),
                              ('extended', {c: '%s-D1-B16' % c for c in ALLC}, ALLC)):
        ps = {c: J[t]['values']['phi_sys'] for c, t in tags.items()}
        pl = {c: J[t]['values']['phi_ls'] for c, t in tags.items()}
        if name != 'extended':
            boot[name] = AN.bootstrap_k1(ps, pl, codes, n=1000)
        loso[name] = AN.k1_loso(ps, pl, codes)
    K1['bootstrap'] = boot
    K1['loso'] = loso
    norm = {}
    for c in ALLC:
        t = '%s-D1-B16' % c
        L0k = {sid: J[t]['Lk'][sid][0] for sid in J[t]['Lk']}
        norm[c] = AN.phi_normalized(J[t]['values']['phi_sys'], J[t]['values']['phi_ls'], L0k)
    K1['normalized'] = dict(Phi=norm, primary=AN.k1_test({c: norm[c] for c in PRIMARY}, PRIMARY),
                            extended=AN.k1_test(norm, ALLC))
    K1['fragile'] = bool((not K1['primary']['fired']) and boot['primary']['P_fire'] > 0.5)
    log('K1 fired:', K1['fired'], K1['subtests_fired'])
    # ---- K2-deg-G
    phis = {c: J['%s-D1-B16' % c]['values']['phi_sys'] for c in PRIMARY}
    K2G = AN.k2_deg_G(phis, {c: Phi('%s-D1-B16' % c) for c in PRIMARY}, corp)
    log('K2-deg-G fired:', K2G['fired'])
    # ---- K2-deg-R (spec cells; LC now with the first-order condition) + EC_sym alternative (reported)
    K2R = AN.k2_deg_R(tr, lawc1)
    cells_sym = [c for c in tr['cells'] if c['law'] != 'EC'] + [c for c in tr['cells_extra'] if c['law'] == 'EC_sym']
    K2R_sym = AN.k2_deg_R(tr, lawc1, cells=cells_sym)
    log('K2-deg-R fired:', K2R['fired'], '(EC_sym alternative:', K2R_sym['fired'], ')')
    # ---- K2-collapse
    rows, _K = AN.count_table(corp)
    uses = {f: float(sum(r['u'] for r in rows if r['law'] == f)) for f in LAW9}
    nlib9 = len(REPS.lib_names(frozenset(LAW9)))

    def pointer(level):
        fq = J['C2-%s-B16' % level]['freq']
        return {'C1': 3 + math.log2(nlib9 + 1), 'C4': 3 + math.log2(nlib9 + 1),
                'C2': -math.log2(fq['kind']['Call']) + math.log2(nlib9 + 1), 'C3': 4 * math.log2(95)}

    def pos_detail(level, key):
        tag4 = 'C4-%s-B16' % level
        Lpos = J[tag4]['law_single']['POS']
        po = J[tag4]['pos_off']
        pm = {int(m): r['pos_sign_known'] for m, r in J[tag4][key].items()}
        pos = dict(add=po[0] - Lpos, loo=po[FULL] - Lpos, n_sign_off={'empty': po[0], 'LAW9': po[FULL]},
                   n_sign_on={'empty': pm[0], 'LAW9': pm[FULL]}, L_POS=Lpos,
                   plus_add=pm[0] - Lpos, plus_loo=pm[FULL] - Lpos)
        pos['V_mid'] = 0.5 * (pos['add'] + pos['loo'])
        pos['Vplus_mid'] = 0.5 * (pos['plus_add'] + pos['plus_loo'])
        return pos

    def settings(level, charged, key='miner', suffix=''):
        off, on = {}, {}
        for c in PRIMARY:
            tag = '%s-%s-B16%s' % (c, level, suffix)
            off[c] = V(tag, charged)
            Lon = {int(m): r['L_on'] for m, r in J[tag][key].items()}
            on[c] = AN.vplus_from_miner(Lon, J[tag]['dec'] if charged else 0.0)
        ls = {c: J['%s-%s-B16%s' % (c, level, suffix)]['law_single'] for c in PRIMARY}
        pos = pos_detail(level, key) if not suffix else None
        return AN.collapse_setting(off, on, ls, pos=pos, uses=uses, pointer=pointer(level if level != 'D2M' else 'D2'))

    def settings_sym(key='miner'):
        """R15 (reported): D1-charged with the differentiation library buyable by EVERY theory:
        L_sym(T) = min(L_D1(T) + dec [EC in T], L_D2(T) + dec)."""
        off, on, ls, phi = {}, {}, {}, {}
        ecb = np.array([1.0 if m >> LAW9.index('EC') & 1 else 0.0 for m in range(1 << 9)])
        for c in PRIMARY:
            j1, j2 = J['%s-D1-B16' % c], J['%s-D2-B16' % c]
            dec = j1['dec']
            L1, L2 = np.array(j1['values']['L_all']), np.array(j2['values']['L_all'])
            Ls = np.minimum(L1 + dec * ecb, L2 + dec)
            phi[c] = dict(zip(LAW9, [float(x) for x in LT.shapley_from_table(Ls)]))
            o = {}
            for i, f in enumerate(LAW9):
                bb = 1 << i
                add = Ls[0] - Ls[bb]
                loo = Ls[FULL ^ bb] - Ls[FULL]
                o[f] = dict(add=float(add), loo=float(loo), mid=float(0.5 * (add + loo)), Phi=phi[c][f])
            off[c] = o
            Lon = {int(m): min(j1[key][m]['L_on'] + dec * ecb[int(m)], j2[key][m]['L_on'] + dec) for m in j1[key]}
            on[c] = AN.vplus_from_miner(Lon, 0.0)
            ls[c] = j1['law_single']
        res = AN.collapse_setting(off, on, ls, uses=uses, pointer=pointer('D1'))
        res['Phi'] = phi
        return res
    COL = {}
    COL['primary_D1_charged'] = settings('D1', True)
    COL['D1_uncharged'] = settings('D1', False)
    COL['D0_charged'] = settings('D0', True)
    COL['D0_uncharged'] = settings('D0', False)
    COL['D2'] = settings('D2', False)
    COL['D2M'] = settings('D2M', False)
    COL['D1_charged_sym'] = settings_sym()
    COL['v2_primary_D1_charged'] = settings('D1', True, key='miner_v2')
    COL['v2_D1_uncharged'] = settings('D1', False, key='miner_v2')
    COL['v2_D2'] = settings('D2', False, key='miner_v2')
    COL['v2_D1_charged_sym'] = settings_sym(key='miner_v2')
    COL['noaddbase_D1_charged'] = settings('D1', True, suffix='-noaddbase')
    prim = COL['primary_D1_charged']
    gate = dict(DIST=prim['labels']['DIST'] in ('abbreviation', 'no-value'),
                KE=prim['labels']['KE'] in ('abbreviation', 'no-value'),
                POS=prim['POS']['label'] == 'constraint')
    gate['passed'] = all(gate.values())
    constraint_laws = [f for f in PHYS7 if prim['labels'][f] == 'constraint']
    # DIST recovery (instrument validity, reported): share of V_mid(DIST) recovered by the miner
    recov = {}
    for name in ('primary_D1_charged', 'v2_primary_D1_charged', 'D2', 'v2_D2'):
        st = COL[name]
        recov[name] = {c: (1 - st['per_code'][c]['DIST']['Vplus_mid'] / st['per_code'][c]['DIST']['V_mid'])
                       if st['per_code'][c]['DIST']['V_mid'] > 0 else None for c in PRIMARY}
    # ---- EC scaling (7.3), only if EC is constraint-type at the primary setting
    ECS = None
    if 'EC' in constraint_laws:
        log('EC constraint-type at primary setting -> running EC scaling test (7.3)')
        res = {}
        with mp.get_context('fork').Pool(4) as pool:
            for code, level, out, err in pool.imap_unordered(surplus_worker, [(c, 'D1') for c in PRIMARY]):
                res[code] = dict(points=out, U5_max_err=err)
                log('surplus', code, 'done; U5 max err', err)
        ECS = dict(per_code={}, duplicates=SCALING_DUPLICATES)
        for c in PRIMARY:
            pts = res[c]['points']
            sids = sorted(pts)
            fits_all = AN.fit_models_73(pts)
            fits_dedup = AN.fit_models_73(pts, dedup=tuple(SCALING_DUPLICATES))
            dup_check = {d: [pts[d]['surplus_on'], pts[o]['surplus_on']] for d, o in SCALING_DUPLICATES.items()}
            ECS['per_code'][c] = dict(cv_r2=fits_dedup['spec [1, n, pairs, P, famG]']['cv_r2'],
                                      cv_r2_with_duplicates=fits_all['spec [1, n, pairs, P, famG]']['cv_r2'],
                                      fits_dedup=fits_dedup, fits_with_duplicates=fits_all, duplicate_values=dup_check,
                                      points={s: dict(surplus_on=pts[s]['surplus_on'], surplus_off=pts[s]['surplus_off'],
                                                      n=pts[s]['n'], pairs=pts[s]['pairs'], P=pts[s]['P'], fam=pts[s]['fam'],
                                                      U6=pts[s]['plus']['U6'] and pts[s]['minus']['U6'])
                                              for s in sids}, U5_max_err=res[c]['U5_max_err'])
        ECS['count_degenerate'] = sum(1 for c in PRIMARY if ECS['per_code'][c]['cv_r2'] > 0.9) >= 3
        ECS['count_degenerate_with_duplicates'] = sum(1 for c in PRIMARY if ECS['per_code'][c]['cv_r2_with_duplicates'] > 0.9) >= 3
        for mname in ('famP, famG*n^3', 'famP, famG, famG*n^3'):
            ECS['family_model_would_fire_' + mname] = sum(
                1 for c in PRIMARY if ECS['per_code'][c]['fits_dedup'][mname]['cv_r2'] > 0.9) >= 3
        if ECS['count_degenerate']:
            constraint_laws = [f for f in constraint_laws if f != 'EC']
    if not gate['passed']:
        collapse = 'INDETERMINATE'
    else:
        collapse = 'fired' if not constraint_laws else 'not fired'
    # R16: operative verdict when the gate fails = the spec's analytic fallback: every substitution-decoder
    # law (N2, N3, PC, LC, HK, UG) is abbreviation-type (formalization A P1 / B Props 4-5); EC, the only
    # non-substitution decoder, keeps its measured label (after the 7.3 removal).
    if collapse == 'INDETERMINATE':
        ec_constraint = 'EC' in constraint_laws
        collapse_operative = 'not fired' if ec_constraint else 'fired'
        constraint_operative = ['EC'] if ec_constraint else []
    else:
        collapse_operative = collapse
        constraint_operative = constraint_laws
    HYP = AN.hypothesis_iii(prim)
    # ---- verdict
    k2_trackG_fail = K2G['fired'] or collapse_operative == 'fired'
    K2 = bool(K2R['fired'] and k2_trackG_fail)
    K2_c18 = bool(K2R['fired'] and (K2G['fired'] or collapse == 'fired'))
    survived_sem = []
    if not K2R['fired']:
        survived_sem.append('Track R (restriction)')
    if not k2_trackG_fail:
        survived_sem.append('Track G (generator)')
    if K1['fired'] or K2:
        cand = 'ABANDON'
    else:
        cand = 'SURVIVES-NARROW(%s; constraint-type laws: %s)' % (', '.join(survived_sem), constraint_operative)
    verdict = dict(K1=dict(fired=K1['fired'], subtests=K1['subtests_fired'], fragile=K1['fragile'],
                           primary_charged_fired=K1['primary_charged']['fired'],
                           machine_charged_fired=K1['machine_charged']['fired'],
                           extended_D1_uncharged_fired=K1['extended']['D1_uncharged']['all']['fired'],
                           precision_by_level={lv: K1['precision_all_levels'][lv]['fired'] for lv in LEVELS}),
                   K2_deg_R=K2R['fired'], K2_deg_R_ECsym_alternative=K2R_sym['fired'], K2_deg_G=K2G['fired'],
                   K2_collapse=collapse, K2_collapse_operative=collapse_operative,
                   K2_collapse_gate=gate, constraint_type_physical_laws=constraint_laws,
                   constraint_type_operative=constraint_operative,
                   K2_overall=K2, K2_overall_if_indeterminate_counts_as_not_fired=K2_c18, candidate=cand,
                   note_indeterminate=('The validity gate failed, so K2-collapse is INDETERMINATE (literal). Operative verdict '
                                       '(spec section 8 fallback): substitution-decoder laws take the analytic abbreviation '
                                       'result; EC keeps its measured label.') if collapse == 'INDETERMINATE' else '')
    # ---- prior predictions
    pp = []
    pp.append(dict(prediction='K2-deg-R fires with R^2 >= 0.99 in R1',
                   hit=bool(K2R['fired'] and K2R['percell']['R1']['r2'] >= 0.99),
                   observed=dict(fired=K2R['fired'], R1_r2=K2R['percell']['R1']['r2'])))
    for f in ('N2', 'N3', 'LC', 'HK', 'UG', 'DIST', 'KE'):
        pp.append(dict(prediction='%s abbreviation-type (Track G)' % f, hit=prim['labels'][f] == 'abbreviation',
                       observed=prim['labels'][f]))
    pp.append(dict(prediction='PC mixed', hit=prim['labels']['PC'] == 'mixed', observed=prim['labels']['PC']))
    ec_obs = dict(D0_uncharged=COL['D0_uncharged']['labels']['EC'], D1_uncharged=COL['D1_uncharged']['labels']['EC'],
                  D2=COL['D2']['labels']['EC'])
    pp.append(dict(prediction='EC constraint-type at D0/D1 uncharged and abbreviation-type at D2',
                   hit=ec_obs['D0_uncharged'] == 'constraint' and ec_obs['D1_uncharged'] == 'constraint' and ec_obs['D2'] == 'abbreviation',
                   observed=ec_obs))
    pp.append(dict(prediction='K1 fires on the machine axis', hit=K1['machine']['fired'],
                   observed=dict(min_rho=K1['machine']['min_rho'], flips=K1['machine']['flips'])))
    c4pairs = [k for k, v in K1['primary']['pairs'].items() if 'C4' in k and v < 0.7]
    c4flips = [f for f in K1['primary']['flips'] if 'C4' in (f['a'], f['b'])]
    pp.append(dict(prediction='K1 fires on C4 vs the symbolic codes (probably)', hit=bool(c4pairs or c4flips),
                   observed=dict(C4_pairs_rho_below_0_7=c4pairs, C4_flips=len(c4flips))))
    RES.update(dict(K1=K1, K2_deg_G=K2G, K2_deg_R=K2R, K2_deg_R_ECsym=K2R_sym, K2_collapse=COL, EC_scaling=ECS,
                    hypothesis_iii=HYP, verdict=verdict, prior_predictions_hit=pp, DIST_recovery=recov))
    # tables
    tables = {}
    for tag in sorted(J):
        if tag.endswith('dup'):
            continue
        tables[tag] = dict(L0=J[tag]['values']['L0'], Lfull=J[tag]['values']['Lfull'], cfg=J[tag]['cfg'],
                           values=J[tag]['values']['values'], values_charged=J[tag]['values_charged']['values'],
                           law_single=J[tag]['law_single'], dec=J[tag]['dec'],
                           miner={int(m): dict(L_off=r['L_off'], L_on=r['L_on'], ntemplates=r['ntemplates'], ndict=r['ndict'])
                                  for m, r in J[tag].get('miner', {}).items()},
                           miner_v2={int(m): dict(L_off=r['L_off'], L_on=r['L_on'], ntemplates=r['ntemplates'],
                                                  ndict=r['ndict'], n_mapped=r.get('n_mapped_systems'))
                                     for m, r in J[tag].get('miner_v2', {}).items()},
                           choice_summary={m: J[tag]['choice_summary'][m] for m in (0, FULL)},
                           phi_sys=J[tag]['values']['phi_sys'], phi_ls=J[tag]['values']['phi_ls'])
    RES['tables'] = tables
    RES['counts'] = dict(cells=K2G['cells'], K_f=K2G['K_f'], uses=uses)
    RES['trackR'] = tr
    # pre-repair numbers (pilot_v1/orig_v1), for the side-by-side table
    try:
        orig = json.load(open(os.path.join(HERE, 'orig_v1', 'results', 'analysis.json')))
        RES['orig_v1'] = dict(tables={t: dict(values=orig['tables'][t]['values'], law_single=orig['tables'][t]['law_single'])
                                      for t in orig['tables'] if t.endswith('B16') and t[:2] in ('C1', 'C2', 'C3', 'C4')},
                              verdict=orig['verdict'])
    except Exception as e:  # pragma: no cover
        RES['orig_v1'] = dict(error=str(e))
    RES['runtime_s_main'] = time.time() - T_START
    # ---- optional: miner-on Shapley (secondary; spec miner, D1)
    elapsed = time.time() - T_START
    RES['miner_on_shapley'] = None
    if elapsed < 3600 * 3:
        log('optional miner-on Shapley (all 512 theories, primary codes, D1)')
        rest = [m for m in range(1 << 9) if m not in miner_masks()]
        chunks = [rest[i::4] for i in range(4)]
        args = [(c, 'D1', 16, ch) for c in PRIMARY for ch in chunks]
        Lon = {c: {} for c in PRIMARY}
        t_opt = time.time()
        with mp.get_context('fork').Pool(4) as pool:
            for code, level, res in pool.imap_unordered(miner_shapley_worker, args):
                Lon[code].update(res)
        if time.time() - t_opt < 6 * 3600:
            msh = {}
            for c in PRIMARY:
                tag = '%s-D1-B16' % c
                for m, r in J[tag]['miner'].items():
                    Lon[c][int(m)] = r['L_on']
                arr = np.array([Lon[c][m] for m in range(1 << 9)])
                msh[c] = dict(zip(LAW9, [float(x) for x in LT.shapley_from_table(arr)]))
            RES['miner_on_shapley'] = dict(level='D1', Phi_plus=msh,
                                           K1_test=AN.k1_test(msh, PRIMARY),
                                           runtime_s=time.time() - t_opt)
            log('miner-on Shapley done %.0fs' % (time.time() - t_opt))
    RES['runtime_s_total'] = time.time() - T_START
    with open(os.path.join(RESULTS, 'analysis.json'), 'w') as fh:
        json.dump(RES, fh, indent=1, default=str)
    import report
    report.write(RES, os.path.join(PILOT_E, 'results.md'), os.path.join(PILOT_E, 'results.json'))
    log('verdict:', json.dumps(verdict, default=str))
    log('done in %.0fs' % (time.time() - T_START))


if __name__ == '__main__':
    main()

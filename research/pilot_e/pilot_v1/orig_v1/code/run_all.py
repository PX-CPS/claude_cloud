#!/usr/bin/env python3
"""Single entry point: regenerates every number of pilot E-v1.

    python3 run_all.py            # full run (~20-60 min on 4 cores)

Writes pilot_v1/results/*.json, pilot_v1/results/run.log, and
research/pilot_e/results.json + research/pilot_e/results.md.
"""
import os, sys
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json, time, math, itertools, multiprocessing as mp, lzma, hashlib, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LAW9, PHYS7, PRIMARY, LEVELS, SEED, RESULTS, HERE
import corpus as CP

T_START = time.time()
LOGF = os.path.join(RESULTS, 'run.log')
PILOT_E = os.path.dirname(HERE)
FULL = (1 << 9) - 1


def log(*a):
    msg = '[%7.1fs] ' % (time.time() - T_START) + ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOGF, 'a') as fh:
        fh.write(msg + '\n')


def miner_masks():
    ms = [0, FULL] + [1 << i for i in range(9)] + [FULL ^ (1 << i) for i in range(9)]
    return ms


# ---------------------------------------------------------------- worker
def job_worker(args):
    code, level, B, with_miner, tag, extra_masks = args
    import lattice as LT
    t0 = time.time()
    lines = []
    j = LT.Job(code, level, B=B, check=True, log=lambda *a: lines.append(' '.join(map(str, a)))).run()
    out = dict(tag=tag, code=code, level=level, B=B, nreps=len(j.reps))
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
                                            cost=j.choice[(s.sid, m)][0].cost, flag=j.choice[(s.sid, m)][1])
                                 for s in j.corp} for m in (0, FULL)}
    if code == 'C4':
        out['pos_off'] = {0: j.pos_counts(0), FULL: j.pos_counts(FULL)}
    if with_miner:
        out['miner'] = {}
        for m in miner_masks() + list(extra_masks):
            if m in out['miner']:
                continue
            out['miner'][m] = j.mine(m)
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
    import lattice as LT
    j = LT.Job(code, level, B=B, check=False, log=lambda *a: None).run()
    res = {}
    for m in masks:
        r = j.mine(m, check=False)
        res[m] = r['L_on']
    return code, level, res


def surplus_worker(args):
    code, level = args
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


# ---------------------------------------------------------------- main
def main():
    open(LOGF, 'w').close()
    log('pilot E-v1 start; SPEC sha256:', open(os.path.join(HERE, 'SPEC.sha256')).read().split()[0])
    import tests, analysis as AN, trackr as TRR
    truth = CP.build_truth()
    RES = dict(spec_sha256=open(os.path.join(HERE, 'SPEC.sha256')).read().split()[0])
    # ---- static unit tests
    ut = {}
    ut['U1'] = tests.U1()
    ut['U7'] = tests.U7()
    ut['U8'] = tests.U8()
    log('U1', ut['U1'][0], 'U7', ut['U7'][0], 'U8', ut['U8'][0], ut['U8'][1])
    # ---- Track G jobs
    jobs = []
    for level in LEVELS:
        for code in PRIMARY:
            jobs.append((code, level, 16, True, '%s-%s-B16' % (code, level), ()))
        jobs.append(('C4', level, 8, False, 'C4-%s-B8' % level, ()))
        jobs.append(('C4', level, 32, False, 'C4-%s-B32' % level, ()))
        jobs.append(('C5', level, 16, False, 'C5-%s-B16' % level, ()))
    jobs.append(('C1', 'D1', 16, True, 'C1-D1-B16-dup', ()))
    jobs.sort(key=lambda x: (x[0] != 'C1' or x[1] != 'D1', not x[3]))
    J = {}
    with mp.get_context('fork').Pool(4) as pool:
        for out in pool.imap_unordered(job_worker, jobs):
            J[out['tag']] = out
            log('job', out['tag'], 'done in %.0fs' % out['time'], 'nreps', out['nreps'])
            with open(os.path.join(RESULTS, 'job_%s.json' % out['tag']), 'w') as fh:
                json.dump(out, fh, indent=0, default=str)
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
    ut['U5'] = (u5 < 1e-9 and u5m < 1e-9, dict(max_rel_err_reps=u5, max_rel_err_miner=u5m,
                                                by_form={t: o['U5_max_err_by_form'] for t, o in J.items()}))
    u6 = all(r['U6_roundtrip'] for o in J.values() for r in o.get('miner', {}).values())
    ut['U6'] = (bool(u6), dict(n_runs=sum(len(o.get('miner', {})) for o in J.values())))
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
    # ---- K1
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
    K1['primary_by_level'] = {lv: AN.k1_test({c: Phi('%s-%s-B16' % (c, lv)) for c in PRIMARY}, PRIMARY) for lv in LEVELS}
    K1['machine_by_code'] = {c: AN.k1_test({lv: Phi('%s-%s-B16' % (c, lv)) for lv in LEVELS}, LEVELS) for c in PRIMARY}
    K1['C5_vs_primary'] = {c: AN.spearman([Phi('C5-D1-B16')[f] for f in LAW9], [Phi('%s-D1-B16' % c)[f] for f in LAW9])
                           for c in PRIMARY}
    # C8
    import lattice as LT
    c8 = np.array(J['C1-D1-B16']['C8_L_all'])
    phi8 = LT.shapley_from_table(c8)
    Phi8 = dict(zip(LAW9, [float(x) for x in phi8]))
    K1['C8'] = dict(Phi=Phi8, L0=float(c8[0]), Lfull=float(c8[FULL]),
                    spearman={c: AN.spearman([Phi8[f] for f in LAW9], [Phi('%s-D1-B16' % c)[f] for f in LAW9]) for c in PRIMARY},
                    top=max(LAW9, key=lambda f: Phi8[f]))
    # bootstrap
    boot = {}
    for name, tags, codes in (('primary', {c: '%s-D1-B16' % c for c in PRIMARY}, PRIMARY),
                              ('machine', {lv: 'C1-%s-B16' % lv for lv in LEVELS}, LEVELS)):
        ps = {c: J[t]['values']['phi_sys'] for c, t in tags.items()}
        pl = {c: J[t]['values']['phi_ls'] for c, t in tags.items()}
        boot[name] = AN.bootstrap_k1(ps, pl, codes, n=1000)
    K1['bootstrap'] = boot
    K1['fragile'] = bool((not K1['primary']['fired']) and boot['primary']['P_fire'] > 0.5)
    log('K1 fired:', K1['fired'], K1['subtests_fired'])
    # ---- K2-deg-G
    phis = {c: J['%s-D1-B16' % c]['values']['phi_sys'] for c in PRIMARY}
    K2G = AN.k2_deg_G(phis, {c: Phi('%s-D1-B16' % c) for c in PRIMARY}, corp)
    log('K2-deg-G fired:', K2G['fired'])
    # ---- K2-deg-R
    K2R = AN.k2_deg_R(tr, lawc1)
    log('K2-deg-R fired:', K2R['fired'])
    # ---- K2-collapse
    law_single = {c: J['%s-D1-B16' % c]['law_single'] for c in PRIMARY}

    def settings(level, charged):
        off, on = {}, {}
        for c in PRIMARY:
            tag = '%s-%s-B16' % (c, level)
            off[c] = V(tag, charged)
            Lon = {int(m): r['L_on'] for m, r in J[tag]['miner'].items()}
            on[c] = AN.vplus_from_miner(Lon, J[tag]['dec'] if charged else 0.0)
        tag4 = 'C4-%s-B16' % level
        Lpos = J[tag4]['law_single']['POS']
        po = J[tag4]['pos_off']
        pm = {int(m): r['pos_sign_known'] for m, r in J[tag4]['miner'].items()}
        pos = dict(add=po[0] - Lpos, loo=po[FULL] - Lpos, n_sign_off={'empty': po[0], 'LAW9': po[FULL]},
                   n_sign_on={'empty': pm[0], 'LAW9': pm[FULL]}, L_POS=Lpos,
                   plus_add=pm[0] - Lpos, plus_loo=pm[FULL] - Lpos)
        pos['V_mid'] = 0.5 * (pos['add'] + pos['loo'])
        pos['Vplus_mid'] = 0.5 * (pos['plus_add'] + pos['plus_loo'])
        ls = {c: J['%s-%s-B16' % (c, level)]['law_single'] for c in PRIMARY}
        return AN.collapse_setting(off, on, ls, pos=pos)
    COL = {}
    COL['primary_D1_charged'] = settings('D1', True)
    COL['D1_uncharged'] = settings('D1', False)
    COL['D0_charged'] = settings('D0', True)
    COL['D0_uncharged'] = settings('D0', False)
    COL['D2'] = settings('D2', False)
    prim = COL['primary_D1_charged']
    gate = dict(DIST=prim['labels']['DIST'] in ('abbreviation', 'no-value'),
                KE=prim['labels']['KE'] in ('abbreviation', 'no-value'),
                POS=prim['POS']['label'] == 'constraint')
    gate['passed'] = all(gate.values())
    constraint_laws = [f for f in PHYS7 if prim['labels'][f] == 'constraint']
    # ---- EC scaling (7.3), only if EC is constraint-type at the primary setting
    ECS = None
    if 'EC' in constraint_laws:
        log('EC constraint-type at primary setting -> running EC scaling test (7.3)')
        res = {}
        with mp.get_context('fork').Pool(4) as pool:
            for code, level, out, err in pool.imap_unordered(surplus_worker, [(c, 'D1') for c in PRIMARY]):
                res[code] = dict(points=out, U5_max_err=err)
                log('surplus', code, 'done; U5 max err', err)
        ECS = dict(per_code={})
        for c in PRIMARY:
            pts = res[c]['points']
            sids = sorted(pts)
            y = [pts[s]['surplus_on'] for s in sids]
            X = [[1.0, pts[s]['n'], pts[s]['pairs'], pts[s]['P'], 1.0 if pts[s]['fam'] == 'G' else 0.0] for s in sids]
            b, r2 = AN.insample(X, y)
            ECS['per_code'][c] = dict(cv_r2=AN.cv_r2(X, y, sids), r2=r2, coef=[float(x) for x in b],
                                      points={s: dict(surplus_on=pts[s]['surplus_on'], surplus_off=pts[s]['surplus_off'],
                                                      n=pts[s]['n'], pairs=pts[s]['pairs'], P=pts[s]['P'], fam=pts[s]['fam'],
                                                      U6=pts[s]['plus']['U6'] and pts[s]['minus']['U6'])
                                              for s in sids}, U5_max_err=res[c]['U5_max_err'])
        ECS['count_degenerate'] = sum(1 for c in PRIMARY if ECS['per_code'][c]['cv_r2'] > 0.9) >= 3
        if ECS['count_degenerate']:
            constraint_laws = [f for f in constraint_laws if f != 'EC']
    if not gate['passed']:
        collapse = 'INDETERMINATE'
    else:
        collapse = 'fired' if not constraint_laws else 'not fired'
    HYP = AN.hypothesis_iii(prim)
    # ---- verdict
    k2_trackG_fail = K2G['fired'] or collapse == 'fired'
    K2 = bool(K2R['fired'] and k2_trackG_fail)
    survived_sem = []
    if not K2R['fired']:
        survived_sem.append('Track R (restriction)')
    if not k2_trackG_fail:
        survived_sem.append('Track G (generator)')
    if K1['fired'] or K2:
        cand = 'ABANDON'
    else:
        cand = 'SURVIVES-NARROW(%s; constraint-type laws: %s)' % (', '.join(survived_sem), constraint_laws)
    verdict = dict(K1=dict(fired=K1['fired'], subtests=K1['subtests_fired'], fragile=K1['fragile']),
                   K2_deg_R=K2R['fired'], K2_deg_G=K2G['fired'], K2_collapse=collapse,
                   K2_collapse_gate=gate, constraint_type_physical_laws=constraint_laws,
                   K2_overall=K2, candidate=cand,
                   note_indeterminate=('K2-collapse INDETERMINATE is treated as not fired in K2_overall; '
                                       'analytic verdict applies to substitution-decoder laws') if collapse == 'INDETERMINATE' else '')
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
    RES.update(dict(K1=K1, K2_deg_G=K2G, K2_deg_R=K2R, K2_collapse=COL, EC_scaling=ECS, hypothesis_iii=HYP,
                    verdict=verdict, prior_predictions_hit=pp))
    # tables
    tables = {}
    for tag in sorted(J):
        if tag.endswith('dup'):
            continue
        tables[tag] = dict(L0=J[tag]['values']['L0'], Lfull=J[tag]['values']['Lfull'],
                           values=J[tag]['values']['values'], values_charged=J[tag]['values_charged']['values'],
                           law_single=J[tag]['law_single'], dec=J[tag]['dec'],
                           miner={int(m): dict(L_off=r['L_off'], L_on=r['L_on'], ntemplates=r['ntemplates'], ndict=r['ndict'])
                                  for m, r in J[tag].get('miner', {}).items()},
                           choice_summary=J[tag]['choice_summary'])
    RES['tables'] = tables
    RES['counts'] = dict(cells=K2G['cells'], K_f=K2G['K_f'])
    RES['trackR'] = tr
    RES['runtime_s_main'] = time.time() - T_START
    # ---- optional: miner-on Shapley (secondary; only if within 6 h wall time)
    elapsed = time.time() - T_START
    RES['miner_on_shapley'] = None
    if elapsed < 3600 * 2:
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

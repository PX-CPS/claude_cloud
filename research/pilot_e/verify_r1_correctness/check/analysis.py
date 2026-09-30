"""Decision statistics: K1 (sec 6), K2-deg-G / K2-deg-R / EC scaling (sec 7), K2-collapse (sec 8)."""
import itertools, math, statistics
import numpy as np
from scipy import stats
from common import LAW9, PHYS7, PRIMARY, SEED
import corpus as CP

HYP_ORDER = {'EC': 3, 'PC': 3, 'LC': 3, 'HK': 2, 'UG': 2, 'N2': 2, 'N3': 2, 'DIST': 1, 'KE': 1}


def spearman(a, b):
    r = stats.spearmanr(a, b).correlation
    return float(r) if r == r else float('nan')


# ---------------------------------------------------------------- K1
def k1_test(Phi, codes, laws=LAW9, crit_a=True):
    """Phi: code -> {law: value}."""
    out = dict(pairs={}, tops={}, flips=[])
    for c in codes:
        out['tops'][c] = max(laws, key=lambda f: Phi[c][f])
    rhos = []
    for a, b in itertools.combinations(codes, 2):
        rho = spearman([Phi[a][f] for f in laws], [Phi[b][f] for f in laws])
        out['pairs']['%s-%s' % (a, b)] = rho
        rhos.append(rho)
    for a, b in itertools.permutations(codes, 2):
        ta = out['tops'][a]
        mb = max(Phi[b][f] for f in laws)
        if Phi[b][ta] < 0.9 * mb:
            out['flips'].append(dict(a=a, b=b, top_a=ta, Phi_b_top_a=Phi[b][ta], max_b=mb, top_b=out['tops'][b]))
    out['min_rho'] = min(rhos) if rhos else float('nan')
    fa = crit_a and (out['min_rho'] < 0.7)
    fb = len(out['flips']) > 0
    out['fired_a'] = bool(fa)
    out['fired_b'] = bool(fb)
    out['fired'] = bool(fa or fb)
    return out


def bootstrap_k1(phi_sys, phi_ls, codes, n=1000, seed=SEED):
    """phi_sys: code -> {sid: [9]}, phi_ls: code -> [9].  Resample systems with replacement."""
    rng = np.random.default_rng(seed)
    sids = sorted(next(iter(phi_sys.values())).keys())
    M = {c: np.array([phi_sys[c][s] for s in sids]) for c in codes}
    LS = {c: np.array(phi_ls[c]) for c in codes}
    rho_s = {('%s-%s' % p): [] for p in itertools.combinations(codes, 2)}
    flips, fires = 0, 0
    for _ in range(n):
        idx = rng.integers(0, len(sids), len(sids))
        Phi = {c: dict(zip(LAW9, M[c][idx].sum(axis=0) + LS[c])) for c in codes}
        res = k1_test(Phi, codes)
        for k, v in res['pairs'].items():
            rho_s[k].append(v)
        flips += res['fired_b']
        fires += res['fired']
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in rho_s.items()}
    return dict(n=n, rho_CI95=ci, P_flip=flips / n, P_fire=fires / n)


# ---------------------------------------------------------------- counts (spec 7.1)
def counts(sysm, f):
    """(r_fk, u_fk) for an applicable cell."""
    if sysm.fam == 'G':
        n = sysm.n
        if f == 'EC':
            return n * (n - 1) // 2, 1
        if f == 'KE':
            return 0, len(sysm.kin_terms)
        return 0, 0
    d = sysm.d
    npair, ncen = sysm.n_pairs(), sysm.n_central()
    nspr, ngrav = sysm.n_springs(), sysm.n_grav()
    n = sysm.dof
    ndist = ncen if d == 2 else 0
    table = {
        'N2': (0, sysm.n_force_terms()),
        'N3': (d * npair, npair),
        'PC': ((d, 1) if sysm.isolated else (0, 0)),
        'LC': ((d - 1) * ncen, ncen),
        'EC': (n * (n - 1) // 2, 1),
        'HK': (nspr, nspr),
        'UG': (2 * ngrav, ngrav),
        'DIST': (0, ndist),
        'KE': (0, sysm.n),
    }
    return table[f]


def count_table(corp):
    rows = []
    for s in corp:
        for f in LAW9:
            if f in s.applicable:
                r, u = counts(s, f)
                rows.append(dict(sid=s.sid, law=f, r=r, u=u, P=len(s.params), n=s.dof))
    K = {f: sum(1 for s in corp if f in s.applicable) for f in LAW9}
    return rows, K


def cv_r2(X, y, groups):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pred = np.zeros_like(y)
    for g in sorted(set(groups)):
        te = np.array([gg == g for gg in groups])
        b, *_ = np.linalg.lstsq(X[~te], y[~te], rcond=None)
        pred[te] = X[te] @ b
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - pred) ** 2).sum()) / sst if sst > 0 else float('nan')


def insample(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    sst = float(((y - y.mean()) ** 2).sum())
    return b, 1 - float(((y - X @ b) ** 2).sum()) / sst if sst > 0 else float('nan')


def k2_deg_G(phi_sys_by_code, Phi_by_code, corp, codes=PRIMARY):
    rows, K = count_table(corp)
    byid = {s.sid: s for s in corp}
    out = dict(M1={}, lawlevel={}, K_f=K)
    laws_present = [f for f in LAW9 if any(r['law'] == f for r in rows)]
    for c in codes:
        y = [phi_sys_by_code[c][r['sid']][LAW9.index(r['law'])] for r in rows]
        X = [[1.0 if r['law'] == f else 0.0 for f in laws_present] + [r['r'], r['u'], r['P'], r['n']] for r in rows]
        groups = [r['sid'] for r in rows]
        b, r2 = insample(X, y)
        out['M1'][c] = dict(cv_r2=cv_r2(X, y, groups), r2=r2, n=len(y),
                            coef=dict(zip(['FE_' + f for f in laws_present] + ['beta_r', 'beta_u', 'beta_P', 'beta_n'],
                                          [float(x) for x in b])))
        # R11 (reported alongside M1): per-law slopes on uses, as the abbreviation formula u*(p-c) implies
        Xp = [[1.0 if r['law'] == f else 0.0 for f in laws_present] +
              [r['u'] if r['law'] == f else 0.0 for f in laws_present] + [r['r'], r['P'], r['n']] for r in rows]
        bp, r2p = insample(Xp, y)
        out.setdefault('M1_perlaw', {})[c] = dict(cv_r2=cv_r2(Xp, y, groups), r2=r2p, n=len(y))
        # law level
        yl = [Phi_by_code[c][f] for f in LAW9]
        Xl = []
        for f in LAW9:
            sr = sum(r['r'] for r in rows if r['law'] == f)
            sP = sum(r['P'] for r in rows if r['law'] == f)
            Xl.append([1.0, sr, K[f], sP])
        bl, r2l = insample(Xl, yl)
        out['lawlevel'][c] = dict(cv_r2=cv_r2(Xl, yl, LAW9), r2=r2l,
                                  coef=dict(zip(['intercept', 'sum_r', 'K_f', 'sum_P'], [float(x) for x in bl])))
    out['M1_fired'] = sum(1 for c in codes if out['M1'][c]['cv_r2'] > 0.9) >= 3
    out['M1_perlaw_would_fire'] = sum(1 for c in codes if out['M1_perlaw'][c]['cv_r2'] > 0.9) >= 3
    out['lawlevel_fired'] = sum(1 for c in codes if out['lawlevel'][c]['cv_r2'] > 0.9) >= 3
    out['fired'] = bool(out['M1_fired'] or out['lawlevel_fired'])
    out['cells'] = rows
    return out


# ---------------------------------------------------------------- Track R
def r_of_cell(sysm, law):
    return counts(sysm, 'EC' if law == 'EC_sym' else law)[0]


def k2_deg_R(tr, law_c1, cells=None):
    byid = CP.by_id()
    cells = tr['cells'] if cells is None else cells
    codes = {'R1': lambda c: c['I_R1'][8], 'R2': lambda c: c['I_R2'], 'R3': lambda c: c['I_R3'],
             'R4': lambda c: c['I_R4'][1000]}
    extra = {'R1_b4': lambda c: c['I_R1'][4], 'R1_b16': lambda c: c['I_R1'][16], 'R4_N100': lambda c: c['I_R4'][100]}
    out = dict(percell={}, lawlevel={}, VR={}, secondary={})
    rs = np.array([r_of_cell(byid[c['sid']], c['law']) for c in cells], float)
    cod = np.array([c['codim'] for c in cells], float)
    out['r_vs_codim'] = [dict(sid=c['sid'], law=c['law'], r=float(r), codim=int(k)) for c, r, k in zip(cells, rs, cod)]

    def fit(I):
        beta = float((I * rs).sum() / (rs ** 2).sum()) if (rs ** 2).sum() > 0 else 0.0
        res = I - beta * rs
        sst = float(((I - I.mean()) ** 2).sum())
        return dict(beta=beta, r2=1 - float((res ** 2).sum()) / sst if sst > 0 else float('nan'),
                    r2_uncentered=1 - float((res ** 2).sum()) / float((I ** 2).sum()))
    for name, fn in list(codes.items()) + list(extra.items()):
        I = np.array([fn(c) for c in cells], float)
        res = fit(I)
        # also: I ~ codim (the actual codimension) for reference
        bc = float((I * cod).sum() / (cod ** 2).sum())
        sst = float(((I - I.mean()) ** 2).sum())
        res['r2_vs_codim'] = 1 - float(((I - bc * cod) ** 2).sum()) / sst if sst > 0 else float('nan')
        (out['percell'] if name in codes else out['secondary'])[name] = res
        VR = {}
        for f in ['EC', 'PC', 'LC', 'N3']:
            vals = [fn(c) for c in cells if c['law'] in (f, f + '_sym')]
            VR[f] = float(sum(vals) - law_c1[f]) if vals else float(-law_c1[f])
        for f in ['HK', 'UG', 'DIST', 'KE', 'N2']:
            VR[f] = float(-law_c1[f])
        out['VR'][name] = VR
        # law-level V^R ~ sum r (laws with >=1 valid cell)
        lf = [f for f in ['EC', 'PC', 'LC', 'N3'] if any(c['law'] in (f, f + '_sym') for c in cells)]
        if len(lf) >= 2:
            x = np.array([sum(r_of_cell(byid[c['sid']], c['law']) for c in cells if c['law'] in (f, f + '_sym')) for f in lf], float)
            yv = np.array([VR[f] for f in lf])
            X = np.column_stack([np.ones_like(x), x])
            b, r2 = insample(X, yv)
            out['lawlevel'][name] = dict(laws=lf, sum_r=x.tolist(), VR=yv.tolist(), coef=b.tolist(), r2=r2)
    out['fired'] = sum(1 for n in codes if out['percell'][n]['r2'] > 0.9) >= 3
    return out


# ---------------------------------------------------------------- K2 collapse
def classify(vmid, vplus, tau):
    if vmid <= 0:
        return 'no-value'
    if vplus < tau or vplus < 0.2 * vmid:
        return 'abbreviation'
    if vplus >= tau and vplus >= 0.5 * vmid:
        return 'constraint'
    return 'mixed'


def majority(labels):
    from collections import Counter
    c = Counter(labels)
    lab, n = c.most_common(1)[0]
    return lab if n >= 3 else 'unstable'


def vplus_from_miner(Lon, charged_dec):
    """Lon: {mask: L+}.  Returns {law: dict(add, loo, mid)} with optional EC charge."""
    full = (1 << 9) - 1
    ecb = 1 << LAW9.index('EC')

    def L(m):
        return Lon[m] + (charged_dec if (m & ecb) else 0.0)
    out = {}
    for i, f in enumerate(LAW9):
        b = 1 << i
        add = L(0) - L(b)
        loo = L(full ^ b) - L(full)
        out[f] = dict(add=add, loo=loo, mid=0.5 * (add + loo))
    return out


def classify_per_use(vmid, vplus, uses, pointer):
    """R12 (reported alternative): threshold per use -- surplus per use vs the per-use pointer cost."""
    if vmid <= 0:
        return 'no-value'
    pu = vplus / max(uses, 1)
    if pu < pointer or vplus < 0.2 * vmid:
        return 'abbreviation'
    if pu >= pointer and vplus >= 0.5 * vmid:
        return 'constraint'
    return 'mixed'


def collapse_setting(vals_off, vals_on, law_single, codes=PRIMARY, pos=None, uses=None, pointer=None):
    """vals_off/vals_on: code -> {law: {mid}}; law_single: code -> {law: L_c(f)}.
    uses: {law: total uses U_f}; pointer: code -> per-use pointer cost (R12, reported only)."""
    out = dict(per_code={}, tau={}, kappa={})
    for c in codes:
        tau = 2 * statistics.median([law_single[c][f] for f in LAW9])
        out['tau'][c] = tau
        out['per_code'][c] = {}
        out['kappa'][c] = {}
        for f in LAW9:
            vm, vp = vals_off[c][f]['mid'], vals_on[c][f]['mid']
            out['per_code'][c][f] = dict(V_mid=vm, Vplus_mid=vp, label=classify(vm, vp, tau))
            if uses is not None:
                out['per_code'][c][f]['label_per_use'] = classify_per_use(vm, vp, uses[f], pointer[c])
            out['kappa'][c][f] = (vp / vm) if vm > 0 else float('-inf')
    out['labels'] = {f: majority([out['per_code'][c][f]['label'] for c in codes]) for f in LAW9}
    if uses is not None:
        out['labels_per_use'] = {f: majority([out['per_code'][c][f]['label_per_use'] for c in codes]) for f in LAW9}
        out['pointer'] = pointer
        out['uses'] = uses
    if pos is not None:
        tau = out['tau']['C4']
        out['POS'] = dict(V_mid=pos['V_mid'], Vplus_mid=pos['Vplus_mid'],
                          label=classify(pos['V_mid'], pos['Vplus_mid'], tau), detail=pos)
        if uses is not None:
            out['POS']['label_per_use'] = classify_per_use(pos['V_mid'], pos['Vplus_mid'], pos['n_sign_off']['empty'],
                                                           pointer['C4'])
    return out


def hypothesis_iii(setting, codes=PRIMARY):
    kmed = {}
    for f in LAW9:
        ks = [setting['kappa'][c][f] for c in codes]
        kmed[f] = float(np.median(ks))
    rho_code = {}
    for c in codes:
        xs = [setting['kappa'][c][f] for f in LAW9]
        xs = [x if x != float('-inf') else -1e9 for x in xs]
        rho_code[c] = spearman(xs, [HYP_ORDER[f] for f in LAW9])
    kk = [kmed[f] if kmed[f] != float('-inf') else -1e9 for f in LAW9]
    rho_med = spearman(kk, [HYP_ORDER[f] for f in LAW9])
    pc_lc_abbrev = [f for f in ('PC', 'LC') if setting['labels'][f] == 'abbreviation']
    defs_max = max(kmed['DIST'], kmed['KE'])
    cons_min = min(kmed['EC'], kmed['PC'], kmed['LC'])
    outranks = defs_max > cons_min
    return dict(kappa_median=kmed, spearman_per_code=rho_code, spearman_median=rho_med,
                PC_or_LC_abbreviation=pc_lc_abbrev, definition_outranks_conservation=bool(outranks),
                refuted=bool(pc_lc_abbrev or outranks))



# ---------------------------------------------------------------- repair r1 additions (reported)
def k1_loso(phi_sys, phi_ls, codes, laws=LAW9):
    """R13: drop each system in turn (per-system decomposition, miner off) and re-run the K1 test."""
    sids = sorted(next(iter(phi_sys.values())).keys())
    out = {}
    for s in sids:
        Phi = {c: dict(zip(LAW9, np.sum([phi_sys[c][t] for t in sids if t != s], axis=0) + np.array(phi_ls[c])))
               for c in codes}
        k = k1_test(Phi, codes, laws=laws)
        out[s] = dict(tops=k['tops'], fired=k['fired'], min_rho=k['min_rho'], n_flips=len(k['flips']))
    return dict(per_drop=out, n_fired=sum(1 for v in out.values() if v['fired']),
                fired_drops=[s for s, v in out.items() if v['fired']])


def phi_normalized(phi_sys, phi_ls, L0k):
    """R13: per-system normalised Shapley: sum_k phi(f,k)/L_k(empty) + phi_ls(f)/mean_k L_k(empty)."""
    sids = sorted(phi_sys)
    mean0 = float(np.mean([L0k[s] for s in sids]))
    v = np.zeros(len(LAW9))
    for s in sids:
        v += np.array(phi_sys[s]) / L0k[s]
    v += np.array(phi_ls) / mean0
    return dict(zip(LAW9, [float(x) for x in v]))


def fit_models_73(points, dedup=()):
    """7.3 regressions: spec model and (R14, reported) family-aware models; LOSO CV-R2."""
    sids = sorted(s for s in points if s not in dedup)
    y = [points[s]['surplus_on'] for s in sids]
    G = [1.0 if points[s]['fam'] == 'G' else 0.0 for s in sids]
    n = [points[s]['n'] for s in sids]
    models = {
        'spec [1, n, pairs, P, famG]': [[1.0, points[s]['n'], points[s]['pairs'], points[s]['P'], g] for s, g in zip(sids, G)],
        'famP, famG*n^3': [[1 - g, g * k ** 3] for g, k in zip(G, n)],
        'famP, famG, famG*n^3': [[1 - g, g, g * k ** 3] for g, k in zip(G, n)],
    }
    out = {}
    for name, X in models.items():
        b, r2 = insample(X, y)
        out[name] = dict(cv_r2=cv_r2(X, y, sids), r2=r2, coef=[float(x) for x in b], n=len(y))
    return out

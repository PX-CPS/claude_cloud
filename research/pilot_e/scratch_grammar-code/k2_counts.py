import json, itertools, numpy as np
import systems as Y, pilot as P
P.fit_freq(Y.corpus(True))
for tag, incG in (('withG', True), ('noG', False)):
    corp = Y.corpus(incG)
    res = json.load(open('results_%s.json' % tag))
    u, r = {}, {}
    for f in Y.LAWS:
        u[f] = 0; r[f] = 0
        for s in corp:
            full = frozenset(Y.LAWS)
            if f in s.relevant(full - {'EC'}) or (f == 'EC' and s.conservative) or (f=='KE' and s.conservative):
                u[f] += 1
                if s.family == 'G':
                    nq = len(s.q)
                    r[f] += {'EC': 2*nq - 1, 'KE': len(s.kin_terms)}.get(f, 0)
                    continue
                npair = sum(1 for it in s.inter if it[0] in Y.PAIR)
                ncen = sum(1 for it in s.inter if it[0] in ('spring2','grav','fixgrav','coul'))
                r[f] += {'N2': s.n, 'N3': s.d*npair, 'PC': s.d, 'LC': (s.d-1)*ncen,
                         'EC': 2*s.n*s.d - 1, 'KE': s.n,
                         'HK': sum(1 for it in s.inter if it[0].startswith('spring')),
                         'UG': sum(1 for it in s.inter if it[0] in ('grav','fixgrav')),
                         'DIST': ncen}[f]
    X = np.array([[1, u[f], r[f]] for f in Y.LAWS], float)
    print(tag, 'u', u); print(tag, 'r', r)
    for code in P.CODES:
        for meas in ('shap', 'add', 'loo'):
            yv = np.array([res[code][meas][f] for f in Y.LAWS])
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ beta
            r2 = 1 - ((yv-pred)**2).sum()/((yv-yv.mean())**2).sum()
            # leave-one-out CV R^2
            cv = []
            for i in range(len(yv)):
                m = np.arange(len(yv)) != i
                b, *_ = np.linalg.lstsq(X[m], yv[m], rcond=None)
                cv.append(yv[i] - X[i] @ b)
            cv = np.array(cv); r2cv = 1 - (cv**2).sum()/((yv-yv.mean())**2).sum()
            print('%s %s %-4s R2=%.2f  R2_LOOCV=%.2f' % (tag, code, meas, r2, r2cv))

# per-system pooled test: add-one savings of f on system k vs count r_{f,k}
print()
corp = Y.corpus(True)
for code in ('C1','C2','C3','C4'):
    ys, rs, us, labs = [], [], [], []
    for f in Y.LAWS:
        for s in corp:
            base = P.sys_cost(s, frozenset(), code)[0]
            v = base - P.sys_cost(s, frozenset([f]), code)[0]
            if abs(v) < 1e-9: continue
            if s.family == 'G':
                rr = {'EC': 2*len(s.q)-1, 'KE': len(s.kin_terms)}.get(f, 0)
            else:
                npair = sum(1 for it in s.inter if it[0] in Y.PAIR)
                ncen = sum(1 for it in s.inter if it[0] in ('spring2','grav','fixgrav','coul'))
                rr = {'N2': s.n, 'N3': s.d*npair, 'PC': s.d, 'LC': (s.d-1)*ncen, 'EC': 2*s.n*s.d-1, 'KE': s.n,
                      'HK': sum(1 for it in s.inter if it[0].startswith('spring')),
                      'UG': sum(1 for it in s.inter if it[0] in ('grav','fixgrav')), 'DIST': ncen}[f]
            ys.append(v); rs.append(rr); labs.append((f, s.name, round(v,1), rr))
    ys, rs = np.array(ys), np.array(rs, float)
    Xp = np.column_stack([np.ones_like(rs), rs])
    b, *_ = np.linalg.lstsq(Xp, ys, rcond=None); pr = Xp @ b
    r2 = 1 - ((ys-pr)**2).sum()/((ys-ys.mean())**2).sum()
    # law-specific slopes (V = a_f * r): per-law fixed effect on slope
    laws = sorted(set(l[0] for l in labs))
    Z = np.column_stack([np.array([rr if l[0]==f else 0 for l,rr in zip(labs, rs)]) for f in laws] + [np.array([1.0 if l[0]==f else 0 for l in labs]) for f in laws])
    b2, *_ = np.linalg.lstsq(Z, ys, rcond=None); pr2 = Z @ b2
    r2b = 1 - ((ys-pr2)**2).sum()/((ys-ys.mean())**2).sum()
    print(code, 'pooled (law,system) pairs n=%d: R2(common slope on DOF count)=%.2f  R2(law-specific affine in count)=%.2f' % (len(ys), r2, r2b))
    if code == 'C1':
        for l in labs: print('   ', l)

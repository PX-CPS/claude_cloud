"""Track R: restriction semantics in parameter space (spec 1.7, 4, 7.4).

Model class per eligible system (displacement coordinates about the listed equilibrium):
    q'' = A q + C q' + e (+ u cos(w t) for the driven P04),   theta = (vec A, vec C, e[, u]).
Known-mass variant: M = diag(masses) (family P) or the generalized mass matrix M(q0) (family G).
Laws (linear equations on theta):
  EC: (MA) symmetric, C = 0, u = 0
  PC: sum_i m_i A[(i,a), c] = 0 for each spatial comp a and column c   (isolated systems)
  LC: symmetric part of (M (x) J^T) A = 0                               (planar systems)
  N3: 1D, K = -MA a weighted graph Laplacian: K symmetric (+ zero row sums if isolated; a wall is
      a grounded node, so wall systems only get the symmetry equations)
A cell (f, k) enters only if the linearized ground truth satisfies f's equations (checked
numerically); violated cells are reported and excluded.
"""
import itertools, math, random, zlib
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from common import SEED
import corpus as CP

EPS_BITS = 8
LAWS_R = ['EC', 'PC', 'LC', 'N3']


def loguni(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


class RSys:
    """Linearized model of one eligible system."""
    def __init__(self, sysm, truth):
        self.sysm = sysm
        s = sysm
        self.sid = s.sid
        if s.fam == 'P':
            self.q, self.v = s.coords, s.vels
        else:
            self.q, self.v = s.q, s.v
        self.N = len(self.q)
        acc = truth[s.sid]['acc']
        self.params = list(s.params)
        q0 = self.equilibrium()
        sub0 = dict(zip(self.q, q0))
        sub0.update({v: 0 for v in self.v})
        drive = s.fam == 'P' and s.driven
        cw = sp.Symbol('_cw')
        if drive:
            acc = [a.subs(sp.cos(CP.wdr * CP.t), cw) for a in acc]
        self.A_sym = sp.Matrix(self.N, self.N, lambda i, j: sp.simplify(sp.diff(acc[i], self.q[j]).subs(sub0)))
        self.C_sym = sp.Matrix(self.N, self.N, lambda i, j: sp.simplify(sp.diff(acc[i], self.v[j]).subs(sub0)))
        self.u_sym = sp.Matrix(self.N, 1, lambda i, _: sp.simplify(sp.diff(acc[i], cw))) if drive else None
        e = [a.subs(sub0) for a in acc]
        if drive:
            e = [x.subs(cw, 0) for x in e]
        self.e_sym = sp.Matrix(self.N, 1, lambda i, _: sp.simplify(e[i]))
        self.drive = drive
        if s.fam == 'P':
            self.M_sym = sp.diag(*[s.M(i) for i in s.bodies for _ in range(s.d)])
        else:
            Mm = truth[s.sid]['M']
            self.M_sym = sp.Matrix(self.N, self.N, lambda i, j: sp.simplify(Mm[i][j].subs(sub0)))
        self.slots = list(self.A_sym) + list(self.C_sym) + list(self.e_sym) + (list(self.u_sym) if drive else [])
        self.P = len(self.slots)

    def equilibrium(self):
        s = self.sysm
        sid = s.sid
        if sid == 'P12':
            l = CP.lP
            return [0, 0, l, 0, l / 2, sp.sqrt(3) * l / 2]
        if sid == 'P13':
            return [0, -(CP.lP + CP.mA * CP.g / CP.kA)]
        if sid == 'G06':
            Rh = [p for p in s.params if p.name == 'R'][0]
            Om = [p for p in s.params if p.name == 'Om'][0]
            self._g06 = (Rh, Om)
            return [sp.acos(CP.g / (Rh * Om ** 2))]      # off-axis branch; validity checked at draw time
        return [0] * len(self.q)

    # ------------------------------------------------------------ numeric instances
    def values(self, pvals):
        sub = {p: pvals[p] for p in self.params}
        A = np.array(self.A_sym.subs(sub).evalf(), dtype=float)
        C = np.array(self.C_sym.subs(sub).evalf(), dtype=float)
        e = np.array(self.e_sym.subs(sub).evalf(), dtype=float).ravel()
        u = np.array(self.u_sym.subs(sub).evalf(), dtype=float).ravel() if self.drive else None
        M = np.array(self.M_sym.subs(sub).evalf(), dtype=float)
        theta = np.concatenate([A.ravel(), C.ravel(), e] + ([u] if self.drive else []))
        return dict(A=A, C=C, e=e, u=u, M=M, theta=theta)

    def rows(self, law, M):
        """Constraint rows (list of length-P vectors) for law given the (numeric or sympy) mass matrix."""
        N, P = self.N, self.P
        s = self.sysm
        rows = []

        def zero():
            return [0] * P

        def Aidx(i, j):
            return i * N + j
        if law == 'EC':
            for i in range(N):
                for j in range(i + 1, N):
                    r = zero()
                    for k in range(N):
                        r[Aidx(k, j)] += M[i, k]
                        r[Aidx(k, i)] -= M[j, k]
                    rows.append(r)
            for i in range(N * N):
                r = zero()
                r[N * N + i] = 1
                rows.append(r)
            if self.drive:
                for i in range(N):
                    r = zero()
                    r[2 * N * N + N + i] = 1
                    rows.append(r)
        elif law == 'PC':
            d = s.d
            for a in range(d):
                for col in range(N):
                    r = zero()
                    for bi in range(s.n):
                        r[Aidx(bi * d + a, col)] += M[bi * d + a, bi * d + a]
                    rows.append(r)
        elif law == 'LC':
            J = [[0, -1], [1, 0]]
            W = [[0] * N for _ in range(N)]
            for bi in range(s.n):
                for a in range(2):
                    for b in range(2):
                        W[2 * bi + a][2 * bi + b] = M[2 * bi + a, 2 * bi + a] * J[b][a]   # m_i J^T
            for i in range(N):
                for j in range(i, N):
                    r = zero()
                    for k in range(N):
                        r[Aidx(k, j)] += W[i][k]
                        r[Aidx(k, i)] += W[j][k]
                    rows.append(r)
        elif law == 'N3':
            for i in range(N):
                for j in range(i + 1, N):
                    r = zero()
                    for k in range(N):
                        r[Aidx(k, j)] += M[i, k]
                        r[Aidx(k, i)] -= M[j, k]
                    rows.append(r)
            if s.isolated:
                for i in range(N):
                    r = zero()
                    for j in range(N):
                        for k in range(N):
                            r[Aidx(k, j)] += M[i, k]
                    rows.append(r)
        return rows


def applicable_R(sysm, law):
    A = sysm.applicable
    if law == 'EC':
        return 'EC' in A
    if law == 'PC':
        return 'PC' in A and sysm.isolated
    if law == 'LC':
        return 'LC' in A and sysm.fam == 'P' and sysm.d == 2
    if law == 'N3':
        return 'N3' in A and sysm.fam == 'P' and sysm.d == 1
    return False


def rank(rows, tol=1e-9):
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.array(rows, dtype=float), tol=tol))


def orth_rowspace(rows):
    Rm = np.array(rows, dtype=float)
    u, s, vt = np.linalg.svd(Rm, full_matrices=True)
    r = int(np.sum(s > 1e-9 * max(s.max(), 1)))
    return vt[:r], vt[r:]          # row-space basis (r x P), null-space basis ((P-r) x P)


# ---------------------------------------------------------------- codes
def I_R1(r, b):
    return r * b + 0.5 * r * math.log2(2 * math.pi)


def I_R2(rows, sig):
    Rb, _ = orth_rowspace(rows)
    r = Rb.shape[0]
    S = Rb @ np.diag(sig ** 2) @ Rb.T
    sign, logdet = np.linalg.slogdet(2 * math.pi * S)
    return r * EPS_BITS + 0.5 * logdet / math.log(2)


def I_R3(rows_int):
    """r*log2(201) + log2(index); index = covolume of the saturated row lattice
    = sqrt(det(R_b R_b^T)) / prod(invariant factors of R_b), via Smith normal form."""
    from sympy.matrices.normalforms import smith_normal_form
    Rm = sp.Matrix(rows_int)
    # independent rows
    _, piv = Rm.T.rref()
    Rb = Rm.extract(list(piv), list(range(Rm.shape[1])))
    r = Rb.shape[0]
    # clear denominators row-wise
    rowsI = []
    for i in range(r):
        row = list(Rb.row(i))
        den = sp.ilcm(*[sp.fraction(sp.nsimplify(x))[1] for x in row]) if row else 1
        rowsI.append([int(sp.nsimplify(x) * den) for x in row])
    Rz = sp.Matrix(rowsI)
    # prune zero columns for speed
    cols = [j for j in range(Rz.shape[1]) if any(Rz[i, j] != 0 for i in range(r))]
    Rz = Rz.extract(list(range(r)), cols)
    snf = smith_normal_form(Rz, domain=sp.ZZ)
    inv = [abs(snf[i, i]) for i in range(min(snf.shape)) if snf[i, i] != 0]
    det = (Rz * Rz.T).det()
    log_index = 0.5 * math.log2(int(det)) - sum(math.log2(int(x)) for x in inv)
    return r * math.log2(201) + log_index, r, log_index


def log_evidence(Phi, y, sig):
    n, p = Phi.shape
    if p == 0:
        return -0.5 * (n * math.log(2 * math.pi * sig ** 2) + y @ y / sig ** 2)
    Amat = np.eye(p) + Phi.T @ Phi / sig ** 2
    b = Phi.T @ y / sig ** 2
    L = np.linalg.cholesky(Amat)
    sol = np.linalg.solve(Amat, b)
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (n * math.log(2 * math.pi * sig ** 2) + y @ y / sig ** 2 - b @ sol + logdet)


def simulate(rsys, val, Nsamp, rng, sig=1e-3, ntraj=10):
    """Trajectories of the linear model from random initial conditions; returns design Phi, y."""
    N = rsys.N
    A, C, e, u = val['A'], val['C'], val['e'], val['u']
    w = float(val.get('w', 1.0))
    per = Nsamp // ntraj
    rows, ys = [], []
    for tr in range(ntraj):
        x0 = rng.normal(0, 1, 2 * N)

        def f(t, z):
            q, v = z[:N], z[N:]
            a = A @ q + C @ v + e + (u * math.cos(w * t) if u is not None else 0)
            return np.concatenate([v, a])
        ts = np.linspace(0, 10, per)
        sol = solve_ivp(f, (0, 10), x0, t_eval=ts, rtol=1e-10, atol=1e-12)
        for k, t in enumerate(sol.t):
            q, v = sol.y[:N, k], sol.y[N:, k]
            a = A @ q + C @ v + e + (u * math.cos(w * t) if u is not None else 0)
            for i in range(N):
                row = np.zeros(rsys.P)
                row[i * N:(i + 1) * N] = q
                row[N * N + i * N:N * N + (i + 1) * N] = v
                row[2 * N * N + i] = 1.0
                if u is not None:
                    row[2 * N * N + N + i] = math.cos(w * t)
                rows.append(row)
                ys.append(a[i] + rng.normal(0, sig))
    return np.array(rows), np.array(ys)


def I_R4(rsys, rows, val, Nsamp, rng, sig=1e-3):
    Phi, y = simulate(rsys, val, Nsamp, rng, sig)
    _, Nb = orth_rowspace(rows)
    lz_full = log_evidence(Phi, y, sig)
    lz_f = log_evidence(Phi @ Nb.T, y, sig)
    return (lz_f - lz_full) / math.log(2)


def monomial_tying(rows_rat):
    """RREF parametrization: is every slot a rational multiple of a single free parameter?"""
    Rm = sp.Matrix(rows_rat)
    rref, piv = Rm.rref()
    P = Rm.shape[1]
    free = [j for j in range(P) if j not in piv]
    mono = 0
    for j in range(P):
        if j in free:
            mono += 1
            continue
        i = piv.index(j)
        nz = sum(1 for f in free if rref[i, f] != 0)
        if nz <= 1:
            mono += 1
    return mono, P


# ---------------------------------------------------------------- driver
def run_trackR(truth, law_cost_c1, log=print):
    rng_scales = random.Random(SEED + 1)
    allp = sorted({p for s in CP.corpus() for p in s.params}, key=lambda p: (p.name, bool(p.is_real)))
    scale = {p: loguni(rng_scales, 0.1, 10) for p in allp}
    out = dict(cells=[], excluded=[], codim_table=[], monomial=[], pos=[], mach={}, reported_only=[])
    rsyss = {}
    for s in CP.corpus():
        if not s.trackR and s.sid != 'P07':
            continue
        rsyss[s.sid] = RSys(s, truth)
    for sid, rs in rsyss.items():
        s = rs.sysm
        rng = random.Random(SEED + zlib.crc32(sid.encode()))
        # parameter draw (R1, R2, R4): log-uniform [0.5, 2]; G06 re-drawn until the off-axis eq. exists
        while True:
            pv = {p: loguni(rng, 0.5, 2.0) for p in rs.params}
            if sid != 'G06':
                break
            Rh, Om = rs._g06
            if pv[CP.g] < pv[Rh] * pv[Om] ** 2:
                break
        val = rs.values(pv)
        if rs.drive:
            val['w'] = pv[CP.wdr]
        # integer draw for R3 / monomial check
        rngi = random.Random(SEED + 7 + zlib.crc32(sid.encode()))
        while True:
            pint = {p: sp.Integer(rngi.randint(1, 4)) for p in rs.params}
            if sid != 'G06':
                break
            Rh, Om = rs._g06
            if pint[CP.g] < pint[Rh] * pint[Om] ** 2:
                break
        Mint = rs.M_sym.subs(pint)
        sig2 = np.array([math.prod([scale[p] for p in sorted(sl.free_symbols, key=str) if p in scale]) if sl != 0 else 1.0
                         for sl in rs.slots])
        for law in LAWS_R:
            if not applicable_R(s, law):
                continue
            rows = rs.rows(law, val['M'])
            resid = float(np.max(np.abs(np.array(rows) @ val['theta']))) if rows else 0.0
            r = rank(rows)
            cell = dict(sid=sid, law=law, codim=r, N=rs.N, P=rs.P, resid=resid, fam=s.fam)
            if sid == 'P07':
                cell['note'] = 'reported only (affine, A=0)'
                out['reported_only'].append(cell)
                continue
            if resid > 1e-8:
                cell['note'] = "linearized ground truth violates the law's linear equations; excluded"
                out['excluded'].append(cell)
                log('TrackR excluded %s %s resid %.2e' % (sid, law, resid))
                continue
            cell['I_R1'] = {b: I_R1(r, b) for b in (4, 8, 16)}
            cell['I_R2'] = I_R2(rows, sig2)
            rows_int = rs.rows(law, Mint)
            cell['I_R3'], r3, cell['log2_index'] = I_R3(rows_int)
            assert r3 == r, (sid, law, r3, r)
            rng4 = np.random.default_rng(SEED + zlib.crc32((sid + law).encode()))
            cell['I_R4'] = {Ns: I_R4(rs, rows, val, Ns, rng4) for Ns in (100, 1000)}
            mono, P = monomial_tying(rows_int)
            cell['monomial_slots'] = mono
            out['monomial'].append(dict(sid=sid, law=law, monomial_slots=mono, P=P, all_monomial=(mono == P)))
            out['cells'].append(cell)
        # POS (reported): K = -MA positive definite on EC-restricted isotropic prior, known masses
        if 'EC' in s.applicable and sid != 'P07':
            M = val['M']
            K = -M @ val['A']
            pd_truth = bool(np.all(np.linalg.eigvalsh(0.5 * (K + K.T)) > 1e-9))
            if pd_truth:
                rows = rs.rows('EC', M)
                _, Nb = orth_rowspace(rows)
                rgp = np.random.default_rng(SEED + 11 + zlib.crc32(sid.encode()))
                th = rgp.standard_normal((100000, Nb.shape[0])) @ Nb
                N = rs.N
                As = th[:, :N * N].reshape(-1, N, N)
                Ks = -np.einsum('ij,sjk->sik', M, As)
                Ks = 0.5 * (Ks + np.transpose(Ks, (0, 2, 1)))
                ok = np.all(np.linalg.eigvalsh(Ks) > 0, axis=1)
                p = ok.mean()
                out['pos'].append(dict(sid=sid, frac=float(p), bits=(-math.log2(p) if p > 0 else None),
                                       hits=int(ok.sum())))
            else:
                out['pos'].append(dict(sid=sid, note='K not positive-definite at the linearized truth (zero mode); POS n/a'))
    # Mach block (N2, corpus level, unknown masses): c_bs = k_s/m_b rank 1 over bodies {A,B} x springs {kA,kC}
    rngm = random.Random(SEED + 99)
    pv = {p: loguni(rngm, 0.5, 2.0) for p in (CP.mA, CP.mB, CP.kA, CP.kC)}
    c = np.array([[pv[CP.kA] / pv[CP.mA], pv[CP.kC] / pv[CP.mA]], [pv[CP.kA] / pv[CP.mB], pv[CP.kC] / pv[CP.mB]]])
    grad = np.array([c[1, 1], -c[1, 0], -c[0, 1], c[0, 0]])
    nrm = grad / np.linalg.norm(grad)
    sl = np.array([scale[CP.kA] * scale[CP.mA], scale[CP.kC] * scale[CP.mA], scale[CP.kA] * scale[CP.mB], scale[CP.kC] * scale[CP.mB]])
    out['mach'] = dict(codim=1, det=float(np.linalg.det(c)), I_R1={b: I_R1(1, b) for b in (4, 8, 16)},
                       I_R2=EPS_BITS + 0.5 * math.log2(2 * math.pi * float(nrm @ np.diag(sl ** 2) @ nrm)),
                       formula_value_b8=(2 - 1) * (2 - 1) * 8)
    return out


# ---------------------------------------------------------------- unit tests U7 / U8
def codim_formalization_B(n, d, rng):
    """linear_codims.py class q'' = A q (A only), random masses."""
    N = n * d
    m = rng.uniform(1, 3, n)
    Mf = np.kron(np.diag(m), np.eye(d))

    def E():
        rows = []
        for i in range(N):
            for j in range(i + 1, N):
                r = np.zeros((N, N))
                r[i, j] += Mf[i, i]
                r[j, i] -= Mf[j, j]
                rows.append(r.ravel())
        return rows

    def Pm():
        rows = []
        for a in range(d):
            for col in range(N):
                r = np.zeros((N, N))
                for i in range(n):
                    r[i * d + a, col] = m[i]
                rows.append(r.ravel())
        return rows

    def Tr():
        rows = []
        for a in range(d):
            v = np.zeros(N)
            v[a::d] = 1.0
            for row in range(N):
                r = np.zeros((N, N))
                r[row, :] = v
                rows.append(r.ravel())
        return rows

    def L():
        J2 = np.array([[0., -1.], [1., 0.]])
        W = np.kron(np.diag(m), J2.T)
        rows = []
        for i in range(N):
            for j in range(i, N):
                r = np.zeros((N, N))
                for k in range(N):
                    r[k, j] += W[i, k]
                    r[k, i] += W[j, k]
                rows.append(r.ravel())
        return rows

    def N3():
        rows = E()
        for i in range(N):
            r = np.zeros((N, N))
            for j in range(N):
                r[:, j] += Mf[i, :]
            rows.append(r.ravel())
        return rows
    out = dict(E=rank(E()), P=rank(Pm()), EP=rank(E() + Pm()), ET=rank(E() + Tr()))
    if d == 1:
        out['N3'] = rank(N3())
    if d == 2:
        out['L'] = rank(L())
    return out


def U7():
    rng = np.random.default_rng(SEED)
    ok = True
    table = []
    for d in (1, 2):
        for n in range(1, 6):
            N = n * d
            c = codim_formalization_B(n, d, rng)
            exp = dict(E=N * (N - 1) // 2, P=d * N)
            good = c['E'] == exp['E'] and c['P'] == exp['P'] and c['EP'] == c['ET']
            if d == 1:
                good = good and c['N3'] == n * (n + 1) // 2
            if d == 2:
                good = good and c['L'] == N * (N + 1) // 2
            table.append(dict(n=n, d=d, **c, expected=exp, ok=good))
            ok = ok and good
    return ok, table


def U8():
    rng = np.random.default_rng(SEED)
    S = 100000
    A = rng.standard_normal((S, 2, 2))
    in_class = A[:, 0, 1] * A[:, 1, 0] > 0
    m1 = np.abs(A[:, 1, 0])
    m2 = np.abs(A[:, 0, 1])
    K11 = -m1 * A[:, 0, 0]
    K22 = -m2 * A[:, 1, 1]
    K12 = -m1 * A[:, 0, 1]
    pd = (K11 > 0) & (K11 * K22 - K12 ** 2 > 0)
    b1 = -math.log2(in_class.mean())
    b2 = -math.log2((in_class & pd).mean())
    return (abs(b1 - 1) < 0.05 and abs(b2 - 4) < 0.1), dict(bits_class=b1, bits_class_Kpos=b2)

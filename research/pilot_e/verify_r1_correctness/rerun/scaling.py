"""Out-of-corpus scaling series (spec 2 / 7.3): 2D N-body gravity N=2..6, free 1D spring chain
n=2..10 (shared k), N-link planar pendulum N=1..4.  Used only for the EC scaling-degeneracy test.
Truth for N-link pendulums: EL + LU solve + sp.simplify for N <= 3.  For N = 4 no symbolic
inverse is formed: the numeric check solves M(q) z = f numerically and the explicit-EOM candidate
of the fair min is omitted (it is dominated by the implicit form; see DEVIATIONS.txt)."""
import itertools, os, pickle
import sympy as sp
import corpus as CP
from common import HERE

SC_PKL = os.path.join(HERE, 'truth_scaling.pkl')


def systems():
    out = []
    ms = sp.symbols('m1:11')
    nb = {'N2', 'N3', 'PC', 'LC', 'EC', 'UG', 'DIST', 'KE'}
    for N in range(2, 7):
        s = CP.PSys('SNB%d' % N, 'N-body N=%d' % N, 2, list(ms[:N]),
                    [('grav', i, j) for i, j in itertools.combinations(range(1, N + 1), 2)], nb, False)
        out.append(s)
    for n in range(2, 11):
        s = CP.PSys('SCH%d' % n, 'chain n=%d' % n, 1, list(ms[:n]),
                    [('spring1', i, i + 1, CP.k) for i in range(1, n)], {'N2', 'N3', 'PC', 'EC', 'HK', 'KE'}, False)
        out.append(s)
    g = CP.g
    mR = sp.symbols('m1:5', real=True)
    lR = sp.symbols('l1:5', real=True)
    for N in range(1, 5):
        def Tf(q, v, N=N):
            T = 0
            vx = vy = 0
            for i in range(N):
                vx = vx + lR[i] * v[i] * sp.cos(q[i])
                vy = vy + lR[i] * v[i] * sp.sin(q[i])
                T += mR[i] * sp.trigsimp(sp.expand(vx ** 2 + vy ** 2)) / 2
            return T

        def Vf(q, N=N):
            y, V_ = 0, 0
            for i in range(N):
                y = y - lR[i] * sp.cos(q[i])
                V_ += mR[i] * g * y
            return V_

        def kin(q, v, N=N):
            out_ = []
            vx = vy = 0
            for i in range(N):
                vx = vx + lR[i] * v[i] * sp.cos(q[i])
                vy = vy + lR[i] * v[i] * sp.sin(q[i])
                out_.append((mR[i], sp.trigsimp(sp.expand(vx ** 2 + vy ** 2))))
            return out_
        s = CP.GSys('SNL%d' % N, '%d-link pendulum' % N, N, list(mR[:N]) + list(lR[:N]) + [g], Tf, Vf, kin,
                    {'EC', 'KE'}, False)
        out.append(s)
    return out


def truth(force=False):
    if os.path.exists(SC_PKL) and not force:
        with open(SC_PKL, 'rb') as fh:
            return pickle.load(fh)
    out = {}
    for s in systems():
        if s.fam == 'P':
            out[s.sid] = dict(acc=s.truth())
        else:
            Mm, f = s.implicit()
            sol = sp.Matrix(Mm).LUsolve(sp.Matrix(f)) if s.n <= 3 else None
            # N = 4: no symbolic inverse (explicit candidate skipped; see DEVIATIONS.txt)
            acc = [sp.simplify(e) for e in sol] if s.n <= 3 else None
            out[s.sid] = dict(acc=acc, M=Mm, f=f)
        print('scaling truth', s.sid, flush=True)
    with open(SC_PKL, 'wb') as fh:
        pickle.dump(out, fh)
    return out

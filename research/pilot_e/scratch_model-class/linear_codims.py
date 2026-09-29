"""Codimensions of law-defined subspaces in linear model classes q'' = A q.

Baseline class H: all real A (dimension (d n)^2 for n particles in d spatial dims).
Masses are given (system-independent, known from a shared mass table).
Each law is a set of linear equations on vec(A); codim = rank of that set.
Conjunctions: stack equations. Shapley attribution over laws is also computed.
"""
import itertools, math
import numpy as np

rng = np.random.default_rng(0)


def J2():
    return np.array([[0., -1.], [1., 0.]])


def constraints(n, d, m):
    """Return dict law -> function giving list of linear functionals (rows) on vec(A)."""
    N = n * d
    Mfull = np.kron(np.diag(m), np.eye(d))  # mass matrix

    def E():
        # energy/Hamiltonian (T + V(q), V quadratic): M A symmetric
        rows = []
        for i in range(N):
            for j in range(i + 1, N):
                r = np.zeros((N, N))
                # (M A)_{ij} - (M A)_{ji} = M_ii A_ij - M_jj A_ji
                r[i, j] += Mfull[i, i]
                r[j, i] -= Mfull[j, j]
                rows.append(r.ravel())
        return rows

    def P():
        # linear momentum conservation: sum_i m_i (A q)_i^{(a)} = 0 for each spatial comp a, all q
        rows = []
        for a in range(d):
            for col in range(N):
                r = np.zeros((N, N))
                for i in range(n):
                    r[i * d + a, col] = m[i]
                rows.append(r.ravel())
        return rows

    def Tr():
        # translation invariance of the force field: A (1_n (x) e_a) = 0 (forces depend on differences)
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
        # angular momentum conservation (d=2): dL/dt = sum_i m_i q_i x (A q)_i = 0 for all q
        # i.e. the quadratic form q^T S q with S = (M (x) J)-weighted A has zero symmetric part.
        assert d == 2
        W = np.kron(np.diag(m), J2().T)  # q_i x a_i = q_i^T J^T a_i ... sign irrelevant for rank
        rows = []
        for i in range(N):
            for j in range(i, N):
                r = np.zeros((N, N))
                # (W A)_{ij} + (W A)_{ji}; (W A)_{ij} = sum_k W_ik A_kj
                for k in range(N):
                    r[k, j] += W[i, k]
                    r[k, i] += W[j, k]
                rows.append(r.ravel())
        return rows

    def R():
        # rotation equivariance (d=2): A commutes with I_n (x) J
        assert d == 2
        G = np.kron(np.eye(n), J2())
        rows = []
        for i in range(N):
            for j in range(N):
                r = np.zeros((N, N))
                # (A G - G A)_{ij} = sum_k A_ik G_kj - G_ik A_kj
                for k in range(N):
                    r[i, k] += G[k, j]
                    r[k, j] -= G[i, k]
                rows.append(r.ravel())
        return rows

    laws = {"E": E, "P": P, "T": Tr}
    if d == 2:
        laws.update({"L": L, "R": R})
    return laws


def codim(rows):
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.array(rows), tol=1e-9))


def shapley(names, val):
    n = len(names)
    out = {}
    for x in names:
        s = 0.0
        others = [y for y in names if y != x]
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                s += w * (val(set(S) | {x}) - val(set(S)))
        out[x] = s
    return out


for d in (1, 2):
    print(f"==== spatial dim d={d} ====")
    for n in (1, 2, 3, 4, 5):
        m = rng.uniform(1, 3, n)
        laws = constraints(n, d, m)
        cache = {}

        def val(S):
            key = frozenset(S)
            if key not in cache:
                rows = []
                for s in S:
                    rows += laws[s]()
                cache[key] = codim(rows)
            return cache[key]

        names = list(laws)
        single = {k: val({k}) for k in names}
        pairs = {a + "&" + b: val({a, b}) for a, b in itertools.combinations(names, 2)}
        allc = val(set(names))
        sh = shapley(names, val)
        print(f"n={n} P={(n*d)**2} single={single} all={allc}")
        print("   pairs:", pairs)
        print("   shapley:", {k: round(v, 3) for k, v in sh.items()})

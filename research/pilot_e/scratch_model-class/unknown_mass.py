"""Newton-2 / energy with UNKNOWN (per-system) masses: class A = -M^{-1} K, M>0 diagonal, K symmetric.
(1) generic rank of the parametrisation map (m,K) -> A  => codim (n-1)(n-2)/2 predicted.
(2) n=2: codim 0, so V_c is pure log-volume: P_Gauss(A in class) and P(A in class with K>0).
(3) momentum conservation with unknown masses: exists m>0 with m^T A = 0.
"""
import numpy as np
import sympy as sp

for n in range(1, 7):
    m = sp.symbols(f"m0:{n}", positive=True)
    ks = {}
    K = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            ks[(i, j)] = sp.Symbol(f"k{i}{j}")
            K[i, j] = K[j, i] = ks[(i, j)]
    A = -sp.diag(*m).inv() * K
    params = list(m) + list(ks.values())
    Jac = sp.Matrix([list(A)]).jacobian(params)
    rng = np.random.default_rng(n)
    sub = {p: sp.Rational(int(rng.integers(1, 97)), int(rng.integers(1, 13))) for p in params}
    r = Jac.subs(sub).rank()
    print(f"n={n}: dim image={r}, n^2={n*n}, codim={n*n-r}, predicted (n-1)(n-2)/2={(n-1)*(n-2)//2}")

rng = np.random.default_rng(1)
S = 400000
A = rng.standard_normal((S, 2, 2))
in_class = A[:, 0, 1] * A[:, 1, 0] > 0
# with stability (K = -M A positive definite): need m1,m2>0 with M A symmetric and -MA PD
# choose m = (a21, a12) scaled (sign positive when both same sign and >0 ... handle both signs)
m1 = np.abs(A[:, 1, 0]); m2 = np.abs(A[:, 0, 1])
K11 = -m1 * A[:, 0, 0]; K22 = -m2 * A[:, 1, 1]; K12 = -m1 * A[:, 0, 1]
pd = (K11 > 0) & (K11 * K22 - K12 ** 2 > 0)
print("n=2 Gaussian prior: P(A = -M^-1 K) =", in_class.mean(), " bits:", -np.log2(in_class.mean()))
print("n=2 with K>0 (stable, bounded energy): P =", (in_class & pd).mean(), " bits:", -np.log2((in_class & pd).mean()))

# momentum with unknown masses, 1D: exists m>0, m^T A = 0  <=> A singular with positive left null vector
for n in (2, 3):
    A = rng.standard_normal((20000, n, n))
    # project to singular: codim 1 (det=0) -> then check positivity of left null vector
    cnt = 0
    for a in A:
        u, s, vt = np.linalg.svd(a)
        a0 = a - s[-1] * np.outer(u[:, -1], vt[-1])
        w = u[:, -1]
        if np.all(w > 0) or np.all(w < 0):
            cnt += 1
    print(f"momentum, unknown masses, n={n}: codim 1 (det A=0) + P(positive left null vec | singular) ~ {cnt/len(A):.3f} -> extra bits {-np.log2(cnt/len(A)):.2f}")

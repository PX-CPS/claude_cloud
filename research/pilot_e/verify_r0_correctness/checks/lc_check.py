import sys
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import numpy as np, sympy as sp
import corpus as CP, trackr as TRK
truth = CP.build_truth()
s = CP.by_id()['P12']
rs = TRK.RSys(s, truth)
rng = np.random.default_rng(0)
pv = {p: float(np.exp(rng.uniform(np.log(.5), np.log(2)))) for p in rs.params}
val = rs.values(pv)
A, M = val['A'], val['M']
q0 = [float(sp.sympify(x).subs(pv)) for x in rs.equilibrium()]
N = rs.N
# implemented LC rows
rows = np.array(rs.rows('LC', M), float)
print('implemented LC residual', np.abs(rows @ val['theta']).max())
# correct first-order condition: sum_i m_i r0_i x (A delta)_i = 0 for all delta
w = np.zeros(N)
for bi in range(3):
    x0, y0 = q0[2*bi], q0[2*bi+1]; m = M[2*bi, 2*bi]
    w[2*bi] = -m*y0; w[2*bi+1] = m*x0
print('correct LC residual w^T A', np.abs(w @ A).max(), ' scale', np.abs(A).max())

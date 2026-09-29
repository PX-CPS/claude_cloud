import sympy as sp, time
import scaling as S
def npend_noexp(N):
    th = sp.symbols('th1:%d' % (N + 1)); om = sp.symbols('om1:%d' % (N + 1)); al = sp.symbols('al1:%d' % (N + 1))
    m = sp.symbols('m1:%d' % (N + 1)); l = sp.symbols('l1:%d' % (N + 1)); g = sp.Symbol('g')
    y = 0; vx = vy = 0; T = 0; Vp = 0
    for i in range(N):
        y = y - l[i]*sp.cos(th[i]); vx = vx + l[i]*om[i]*sp.cos(th[i]); vy = vy + l[i]*om[i]*sp.sin(th[i])
        T += m[i]*sp.expand(vx**2+vy**2)/2; Vp += m[i]*g*y
    T = sp.trigsimp(sp.expand(T)); Lg = T - Vp
    eqs = []
    for i in range(N):
        dLdv = sp.diff(Lg, om[i])
        dt = sum(sp.diff(dLdv, th[j])*om[j] + sp.diff(dLdv, om[j])*al[j] for j in range(N))
        eqs.append(sp.expand(dt - sp.diff(Lg, th[i])))
    Mm = sp.Matrix(N, N, lambda i, j: sp.trigsimp(eqs[i].coeff(al[j])))
    fv = [sp.trigsimp(-(eqs[i] - sum(eqs[i].coeff(al[j])*al[j] for j in range(N)))) for i in range(N)]
    ns = 3*N+1
    return dict(N=N, lag=S.cse_c1([Lg], ns), imp=S.cse_c1([sp.factor(e) for e in list(Mm)+fv if e != 0], ns))
for N in (1,2,3):
    t0=time.time(); print(npend_noexp(N), '%.1fs'%(time.time()-t0), flush=True)

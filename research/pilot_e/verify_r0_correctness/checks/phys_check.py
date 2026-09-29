import sys, math, random
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import sympy as sp, numpy as np
import corpus as CP
tr = CP.build_truth(); byid = CP.by_id()
rnd = random.Random(5)
def ev(sid):
    s = byid[sid]; syms = list(s.state) + list(s.params)
    pt = {x: rnd.uniform(0.5, 2) for x in syms}
    val = [float(e.subs(pt)) for e in tr[sid]['acc']]
    P = {x.name: pt[x] for x in syms}
    return P, val
worst = 0
for _ in range(3):
    P, a = ev('G01'); ref = [-P['g']*math.sin(P['q1'])/P['l']]; worst = max(worst, abs(a[0]-ref[0]))
    P, a = ev('G02'); ref = [-P['g']*math.sin(P['q1'])/P['l'] - P['b']*P['v1']/(P['m']*P['l']**2)]; worst = max(worst, abs(a[0]-ref[0]))
    P, a = ev('G05'); ref = [(P['m1']-P['m2'])*P['g']/(P['m1']+P['m2'])]; worst = max(worst, abs(a[0]-ref[0]))
    P, a = ev('G06'); th = P['q1']; ref = [math.sin(th)*(P['Om']**2*math.cos(th) - P['g']/P['R'])]; worst = max(worst, abs(a[0]-ref[0]))
    # double pendulum (standard textbook form)
    P, a = ev('G03'); m1,m2,l1,l2,g=P['m1'],P['m2'],P['l1'],P['l2'],P['g']; t1,t2,w1,w2=P['q1'],P['q2'],P['v1'],P['v2']; d=t1-t2
    den = 2*m1+m2-m2*math.cos(2*d)
    a1 = (-g*(2*m1+m2)*math.sin(t1) - m2*g*math.sin(t1-2*t2) - 2*math.sin(d)*m2*(w2**2*l2 + w1**2*l1*math.cos(d)))/(l1*den)
    a2 = (2*math.sin(d)*(w1**2*l1*(m1+m2) + g*(m1+m2)*math.cos(t1) + w2**2*l2*m2*math.cos(d)))/(l2*den)
    worst = max(worst, abs(a[0]-a1), abs(a[1]-a2))
    # cart-pole: (M+m) x'' + m l cos th th'' - m l sin th th'^2 = 0 ; m l cos th x'' + m l^2 th'' + m g l sin th = 0
    P, a = ev('G07'); M,m,l,g=P['M'],P['m'],P['l'],P['g']; th,w=P['q2'],P['v2']
    A = np.array([[M+m, m*l*math.cos(th)],[m*l*math.cos(th), m*l*l]]); f = np.array([m*l*math.sin(th)*w*w, -m*g*l*math.sin(th)])
    ref = np.linalg.solve(A, f); worst = max(worst, abs(a[0]-ref[0]), abs(a[1]-ref[1]))
print('max abs deviation from independent textbook formulas:', worst)

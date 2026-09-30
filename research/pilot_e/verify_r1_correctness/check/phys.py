import os,sys,math,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP
import sympy as sp, numpy as np
tr=CP.build_truth(); B=CP.by_id()
rng=random.Random(5)
def ev(sid, vals):
    s=B[sid]; syms=list(s.state)+list(s.params)
    f=sp.lambdify(syms, tr[sid]['acc'],'math')
    return np.array(f(*[vals[str(x)] for x in syms]),float)
def rv(): return rng.uniform(0.5,2)
worst={}
for trial in range(5):
    # G03
    v={k:rv() for k in ['q1','q2','v1','v2','m1','m2','l1','l2','g']}
    t1,t2,w1,w2,m1,m2,L1,L2,g=[v[k] for k in ['q1','q2','v1','v2','m1','m2','l1','l2','g']]
    den=2*m1+m2-m2*math.cos(2*t1-2*t2)
    a1=(-g*(2*m1+m2)*math.sin(t1)-m2*g*math.sin(t1-2*t2)-2*math.sin(t1-t2)*m2*(w2**2*L2+w1**2*L1*math.cos(t1-t2)))/(L1*den)
    a2=(2*math.sin(t1-t2)*(w1**2*L1*(m1+m2)+g*(m1+m2)*math.cos(t1)+w2**2*L2*m2*math.cos(t1-t2)))/(L2*den)
    worst['G03']=max(worst.get('G03',0),np.max(np.abs(ev('G03',v)-[a1,a2])))
    # G07
    v={k:rv() for k in ['q1','q2','v1','v2','M','m','l','g']}
    th,w,M,m,l,g=v['q2'],v['v2'],v['M'],v['m'],v['l'],v['g']
    A=np.array([[M+m, m*l*math.cos(th)],[m*l*math.cos(th), m*l*l]]); b=np.array([m*l*w*w*math.sin(th), -m*g*l*math.sin(th)])
    worst['G07']=max(worst.get('G07',0),np.max(np.abs(ev('G07',v)-np.linalg.solve(A,b))))
    # G06
    v={k:rv() for k in ['q1','v1','m','R','Om','g']}
    a=math.sin(v['q1'])*(v['Om']**2*math.cos(v['q1'])-v['g']/v['R'])
    worst['G06']=max(worst.get('G06',0),abs(ev('G06',v)[0]-a))
    v={k:rv() for k in ['q1','v1','m1','m2','g']}
    worst['G05']=max(worst.get('G05',0),abs(ev('G05',v)[0]-(v['m1']-v['m2'])*v['g']/(v['m1']+v['m2'])))
    v={k:rv() for k in ['q1','v1','m','l','g','b']}
    a=-v['g']/v['l']*math.sin(v['q1'])-v['b']*v['v1']/(v['m']*v['l']**2)
    worst['G02']=max(worst.get('G02',0),abs(ev('G02',v)[0]-a))
    # G04 triple pendulum: independent Cartesian construction, numeric EL
    names=['q1','q2','q3','v1','v2','v3','m1','m2','m3','l1','l2','l3','g']
    v={k:rv() for k in names}
    q=sp.symbols('Q1:4'); qd=sp.symbols('W1:4'); ms=sp.symbols('M1:4'); ls=sp.symbols('L1:4'); gg=sp.Symbol('gg')
    x=y=0; T=0; Vp=0
    for i in range(3):
        x=x+ls[i]*sp.sin(q[i]); y=y-ls[i]*sp.cos(q[i])
        xd=sum(sp.diff(x,q[j])*qd[j] for j in range(3)); yd=sum(sp.diff(y,q[j])*qd[j] for j in range(3))
        T+=ms[i]*(xd**2+yd**2)/2; Vp+=ms[i]*gg*y
    Lg=T-Vp
    sub={**{q[i]:v['q%d'%(i+1)] for i in range(3)},**{qd[i]:v['v%d'%(i+1)] for i in range(3)},**{ms[i]:v['m%d'%(i+1)] for i in range(3)},**{ls[i]:v['l%d'%(i+1)] for i in range(3)},gg:v['g']}
    Mm=np.array([[float(sp.diff(Lg,qd[i],qd[j]).subs(sub)) for j in range(3)] for i in range(3)])
    rhs=np.array([float((sp.diff(Lg,q[i])-sum(sp.diff(Lg,qd[i],q[j])*qd[j] for j in range(3))).subs(sub)) for i in range(3)])
    worst['G04']=max(worst.get('G04',0),np.max(np.abs(ev('G04',v)-np.linalg.solve(Mm,rhs))))
print(worst)

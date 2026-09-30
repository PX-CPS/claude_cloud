import os,sys,math,random,itertools
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP
import sympy as sp, numpy as np
tr=CP.build_truth(); B=CP.by_id(); rng=random.Random(7)
def ev(sid, vals):
    s=B[sid]; syms=list(s.state)+list(s.params)
    return np.array(sp.lambdify(syms, tr[sid]['acc'],'math')(*[vals[str(x)] for x in syms]),float)
def rnd(sid):
    s=B[sid]; return {str(x):rng.uniform(0.5,2) for x in list(s.state)+list(s.params)}
W={}
for _ in range(4):
    v=rnd('P14'); a=np.array([v['q']*v['Bz']*v['w1'], -v['q']*v['Bz']*v['u1']])/v['mH']; W['P14']=max(W.get('P14',0),np.abs(ev('P14',v)-a).max())
    v=rnd('P13'); r=np.array([v['x1'],v['y1']]); R=np.linalg.norm(r); a=(-v['kA']*(R-v['l'])*r/R + np.array([0,-v['mG']*v['g']]))/v['mG']; W['P13']=max(W.get('P13',0),np.abs(ev('P13',v)-a).max())
    for sid,ms,kind in [('P12',['mD','mE','mF'],'spr'),('P15',['mP','mQ','mR'],'coul'),('P11',['m1','m2','m3','m4','m5'],'grav'),('P10',['mA','mB','mC'],'grav')]:
        v=rnd(sid); n=len(ms); pos=[np.array([v['x%d'%(i+1)],v['y%d'%(i+1)]]) for i in range(n)]; F=[np.zeros(2) for _ in range(n)]
        ks={(0,1):'k1',(1,2):'k2',(0,2):'k3'}; qs={0:'qA',1:'qB',2:'qC'}
        for i,j in itertools.combinations(range(n),2):
            r=pos[i]-pos[j]; R=np.linalg.norm(r)
            if kind=='spr': f=-v[ks[(i,j)]]*(R-v['l'])*r/R
            elif kind=='coul': f=v['ke']*v[qs[i]]*v[qs[j]]*r/R**3
            else: f=-v['G']*v[ms[i]]*v[ms[j]]*r/R**3
            F[i]+=f; F[j]-=f
        a=np.concatenate([F[i]/v[ms[i]] for i in range(n)]); W[sid]=max(W.get(sid,0),np.abs(ev(sid,v)-a).max()/np.abs(a).max())
    v=rnd('P04'); a=(-v['kA']*v['x1']-v['b']*v['u1']+v['F0']*math.cos(v['w']*v['t']))/v['mA']; W['P04']=max(W.get('P04',0),abs(ev('P04',v)[0]-a))
    v=rnd('P06'); x=[v['x%d'%i] for i in range(1,5)]; m=[v['m%d'%i] for i in range(6,10)]; k=v['k']
    a=[k*(x[1]-x[0])/m[0], (k*(x[0]-x[1])+k*(x[2]-x[1]))/m[1], (k*(x[1]-x[2])+k*(x[3]-x[2]))/m[2], k*(x[2]-x[3])/m[3]]
    W['P06']=max(W.get('P06',0),np.abs(ev('P06',v)-a).max())
print(W)

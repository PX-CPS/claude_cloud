import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, reps as R, trees as TR, lattice as LT, semantics as SM
import sympy as sp
B=CP.by_id(); tr=CP.build_truth()
def rowscaled(s):
    Mm,f=tr[s.sid]['M'],tr[s.sid]['f']
    outs=[]
    st=set(s.state)|set(s.z)
    for i in range(s.n):
        row=sp.expand(sp.Add(*[Mm[i][j]*s.z[j] for j in range(s.n)])-f[i])
        # content: gcd of coefficients over monomials in state-like generators
        terms=sp.Add.make_args(row)
        g=terms[0]
        for t in terms[1:]: g=sp.gcd(g,t)
        c,_=sp.factor_terms(g).as_independent(*st, as_Add=False)
        r2=sp.factor_terms(sp.expand(row/c))
        outs.append(r2)
    return [(None,o) for o in outs], dict(form='implicit', variant='rowscaled')
for code in ['C1','C2','C3','C4']:
  for sid in ['G03','G04','G07','G06']:
    s=B[sid]
    j=LT.Job(code,'D1',check=False,corp=[s])
    c0,rep0,fl=j.sys_best(s,frozenset())
    it,meta=rowscaled(s)
    rep=R.realize(s,it,meta,frozenset(),'D1',code,16)
    err=SM.Decoder(s,tr).check(rep)
    cE,repE,flE=j.sys_best(s,frozenset({'EC'}))
    print(code,sid,'baseline',round(c0-fl,1),rep0.meta['variant'],'rowscaled',round(rep.cost,1),'err %.1e'%err,'EC-form',round(cE-flE,1),repE.form)

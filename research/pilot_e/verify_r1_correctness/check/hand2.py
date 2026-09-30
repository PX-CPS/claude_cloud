import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, reps as R, trees as TR, lattice as LT
import sympy as sp
B=CP.by_id()
def show(code, level, sid, T):
    j=LT.Job(code, level, check=False, corp=[B[sid]])
    s=B[sid]
    c,rep,flag=j.sys_best(s, frozenset(T)&s.applicable)
    print('==',code,level,sid,sorted(T),'cost',round(c,3),'form',rep.form,rep.meta['variant'],'flag',flag,'side',round(rep.side_bits,3))
    for lab,tid in rep.items:
        print('   ',lab, TR.to_sympy_print(tid))
    alts=j.alts[(sid,rep.form,R.form_subset(s,rep.form,frozenset(T)&s.applicable))]
    print('   alts', [(a.meta['variant'], round(a.cost,2)) for a in alts])
for T in [('N2',),('N2','LC')]:
    show('C1','D1','P08',T)
for T in [('N2',),('N2','LC'),('N2','N3'),('N2','N3','LC')]:
    show('C1','D1','P09',T)

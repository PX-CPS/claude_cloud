import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, reps as R, trees as TR, lattice as LT, laws as LW
from lattice import mask_of
import sympy as sp
tr=CP.build_truth()
B=CP.by_id()
def show(code, level, sid, T):
    j=LT.Job(code, level, check=False, corp=[B[sid]])
    s=B[sid]
    c,rep,flag=j.sys_best(s, frozenset(T)&s.applicable)
    print('==',code,level,sid,sorted(T),'cost',round(c,3),'form',rep.form,rep.meta['variant'],'flag',flag,'side',rep.side_bits)
    for lab,tid in rep.items:
        print('   ',lab, TR.to_sympy_print(tid), '| C1prof', TR.profile(tid)[:3])
    return c,rep
for T in [(),('N2',)]:
    show('C1','D1','P02',T)
    show('C3','D1','P02',T)
    show('C4','D1','P02',T)
for T in [(),('EC',),('EC','KE')]:
    show('C1','D1','G01',T)
for T in [(),('N2',),('N2','N3')]:
    show('C1','D1','P05',T)
for T in [(),('N2','UG'),('N2','UG','DIST'), ('N2','LC')]:
    show('C1','D1','P08',T)

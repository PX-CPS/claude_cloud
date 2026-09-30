import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, reps as R, trees as TR, lattice as LT
B=CP.by_id()
def show(code, level, sid, T):
    j=LT.Job(code, level, check=False, corp=[B[sid]])
    s=B[sid]
    c,rep,flag=j.sys_best(s, frozenset(T)&s.applicable)
    print('==',code,level,sid,sorted(T),'cost',round(c,3),'form',rep.form,rep.meta['variant'],'flag',round(flag,3))
    for lab,tid in rep.items:
        print('   ',lab, TR.to_sympy_print(tid), TR.profile(tid)[:3])
    for f in R.formulations(s,level):
        k=(sid,f,R.form_subset(s,f,frozenset(T)&s.applicable))
        if k in j.alts: print('   ',f,[(a.meta['variant'],round(a.cost,2)) for a in j.alts[k]])
for lv in ['D1','D2']:
    for T in [(),('EC',)]:
        show('C1',lv,'G03',T)
show('C1','D1','P07',())
show('C1','D1','P07',('N2',))

import os,sys,json,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import corpus as CP, reps as R, trees as TR, lattice as LT
B=CP.by_id()
j=LT.Job('C2','D1',check=False,corp=[B['P02'],B['G01']])
F=TR.FREQ
print('kind',{k:round(v,4) for k,v in F.kind.items()}); print('role',F.role); print('arity',{k:round(v,4) for k,v in F.arity.items() if v>0.01})
J=json.load(open('/home/user/claude_cloud/research/pilot_e/pilot_v1/results/job_C2-D1-B16.json'))
print('freq equal to job file:', all(abs(J['freq']['kind'][k]-F.kind[k])<1e-12 for k in F.kind))
c,rep,flag=j.sys_best(B['P02'],frozenset())
# manual: Mul(4 args) + Int(-1) + kA + x1 + Pow + mA + Int(-1)
kb=lambda k:-math.log2(F.kind[k])
man = kb('Mul')-math.log2(F.arity[4]) + 2*(kb('Int')+ (2*math.floor(math.log2(2))+1)+1) + kb('Pow')
# leaves: kA(param), x1(state), mA(param); scope state=2 params=2
for role in ['param','state','param']:
    man += kb('Sym') - math.log2(F.role[role]) + math.log2(2)
man += 3  # items header gamma(2)
print('C2 P02 empty: code', c, 'manual', man)

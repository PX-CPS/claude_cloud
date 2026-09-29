import sys, math
sys.path.insert(0, '/home/user/claude_cloud/research/pilot_e/pilot_v1')
import corpus as CP, lattice as LT, trees as TR
truth = CP.build_truth()
LT.fit_freq(truth, 'D1')
F = TR.FREQ
print({k: round(v, 4) for k, v in F.kind.items()})
print({k: round(v, 4) for k, v in F.role.items()})
print({k: round(v, 4) for k, v in F.arity.items() if k <= 5})
# hand: P02 empty: Mul(-1, kA, x1, Pow(mA,-1))  -> kinds: Mul, Int, Sym, Sym, Pow, Sym, Int ; arity 4
kb = lambda k: -math.log2(F.kind[k])
ab = lambda n: -math.log2(F.arity[n])
rb = lambda r, size: -math.log2(F.role[r]) + (math.log2(size) if size > 1 else 0)
ic = 3 + 1  # int_cost(-1) = gamma(2)+1 = 4
tot = kb('Mul') + ab(4) + 2 * (kb('Int') + 4) + kb('Pow') + 3 * kb('Sym') + 2 * rb('param', 2) + rb('state', 2)
print('hand C2 P02 empty (tree only)', tot, ' +gamma(2)=3 ->', tot + 3)
tot2 = kb('Mul') + ab(3) + (kb('Int') + 4) + 2 * kb('Sym') + rb('param', 2) + rb('state', 2)
print('hand C2 P02 N2', tot2 + 3)

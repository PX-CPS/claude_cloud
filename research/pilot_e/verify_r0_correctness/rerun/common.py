"""Common constants for pilot E-v1."""
import math, os

SEED = 20260929
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

LAW9 = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG', 'DIST', 'KE']
PHYS7 = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG']
PRIMARY = ['C1', 'C2', 'C3', 'C4']
LEVELS = ['D0', 'D1', 'D2']
KINDS = ['Add', 'Mul', 'Pow', 'Func', 'Call', 'Sym', 'Int', 'Rat']
BASE_BUILTINS = ['sin', 'cos', 'tan', 'cot', 'exp', 'log']
LOG95 = math.log2(95)


def builtins_for(level):
    return BASE_BUILTINS + (['Dq', 'Dt'] if level == 'D2' else [])


def gamma_len(n):
    n = int(n)
    assert n >= 1, n
    return 2 * int(math.floor(math.log2(n))) + 1


def int_cost(v):
    return gamma_len(abs(int(v)) + 1) + 1


def rat_cost(p, q):
    return int_cost(p) + gamma_len(q)

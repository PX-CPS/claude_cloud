"""Unit tests U1-U8 (spec section 9).  U2-U6 are evaluated on the job outputs in run_all.py;
this module provides the static ones and the helpers."""
import math
import sympy as sp
from common import gamma_len
import trees as TR
import trackr as TRR


def U1():
    g = [gamma_len(n) for n in range(1, 9)]
    ok_g = g == [1, 3, 3, 5, 5, 5, 5, 7]
    x0, k, m1 = sp.symbols('x0 k m1')
    e = x0 * k - m1
    tid = TR.from_sympy(e, 'U1')
    got = TR.tree_cost(tid, 'C1', {'param': 3}, 1, 6)
    # hand count: Add(2 args): 3+gamma(1)=4; Mul(k,x0): 4; leaves k, x0, m1: 3+log2(3) each;
    # Mul(-1, m1): 4; Int -1: 3 + gamma(2)+1 = 7.   Total = 19 + 3*(3+log2 3)
    hand = 4 + 4 + 4 + 7 + 3 * (3 + math.log2(3))
    return bool(ok_g and abs(got - hand) < 1e-9), dict(gamma=g, c1_cost=got, hand=hand, tree=sp.srepr(e))


def U7():
    return TRR.U7()


def U8():
    return TRR.U8()

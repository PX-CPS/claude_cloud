"""Common constants for pilot E-v1."""
import math, os

SEED = 20260929
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)

LAW9 = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG', 'DIST', 'KE']
PHYS7 = ['N2', 'N3', 'PC', 'LC', 'EC', 'HK', 'UG']
PRIMARY = ['C1', 'C2', 'C3', 'C4']
EXTRA_CODES = ['C6', 'C7', 'C9']          # repair round r1: structurally different codes (reported)
LEVELS = ['D0', 'D1', 'D2']
KINDS = ['Add', 'Mul', 'Pow', 'Func', 'Call', 'Sym', 'Int', 'Rat']
BASE_BUILTINS = ['sin', 'cos', 'tan', 'cot', 'exp', 'log']
LOG95 = math.log2(95)

# ---------------------------------------------------------------------------------------------
# Repair round r1 switches (see DEVIATIONS.txt, section R).  The defaults are the repaired
# pipeline; run_all.py also runs a variant with the added baseline alternatives switched off
# ('n2_factored', 'lc_vec') so that both are reported.
CFG = dict(
    n2_factored=True,      # R1: no-N2 baseline may write (sum of forces)/m_i per body (pure algebra)
    lc_vec=True,           # R6: no-LC baseline may write a central term as s*V(dx,dy) (base grammar)
    stmt_cse=True,         # R4: law statements coded with the same CSE + greedy inlining as models
    side_info=True,        # R2: N2 / PC decoders pay for the body -> mass-parameter pointers
    gradres_scope=True,    # R5: gradient-residual forms do not put z1..zn in the leaf scope
)


def set_cfg(d):
    CFG.update(d or {})


def builtins_for(level):
    if level == 'D2':
        return BASE_BUILTINS + ['Dq', 'Dt']
    if level == 'D2M':                      # R7 (added machine variant): D2 + 'for each coordinate' map
        return BASE_BUILTINS + ['Dq', 'Dt', 'Map']
    return list(BASE_BUILTINS)


def diff_level(level):
    return level in ('D2', 'D2M')


def gamma_len(n):
    n = int(n)
    assert n >= 1, n
    return 2 * int(math.floor(math.log2(n))) + 1


def int_cost(v):
    return gamma_len(abs(int(v)) + 1) + 1


def rat_cost(p, q):
    return int_cost(p) + gamma_len(q)

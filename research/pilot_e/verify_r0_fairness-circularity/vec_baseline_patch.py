import sympy as sp
import reps as R, corpus as CP
from trees import V
_tf = R.template_force
def template_force(sysm, it, i, S):
    t = _tf(sysm, it, i, S)
    if t is None and it[0] in CP.CENTRAL and sysm.d == 2:
        _, cen = R.explicit_force(sysm, it, i, S)
        s, dx, dy = cen
        return s * V(dx, dy)
    return t
R.template_force = template_force

"""Track G lattice: per (code, machine level, B) job.

L_c(S|T) = sum_{f in T} L_c(f|T) + [dec_c(EC) if charged and EC in T] + CC_c(T) + sum_k L*_c(M_k|T)
L*_c(M_k|T) = min_{formulations licensed} cost(rep) + log2(#formulations licensed)
CC_c(T) (C4 only) = B * #distinct law-licensed corpus constants referenced (masses under N2, spring
constants under HK, G under UG, g and ke always).  For the per-system decomposition each corpus
constant's B bits are split equally among the systems that reference it.
"""
import itertools, math, json, time, os, lzma
import numpy as np
import sympy as sp
from common import LAW9, SEED, gamma_len, LOG95
import corpus as CP
import trees as TR
import reps as R
import laws as LW
import semantics as SM
import miner as MN

NL = len(LAW9)
ALLMASKS = list(range(1 << NL))


def mask_of(T):
    return sum(1 << LAW9.index(f) for f in T)


def set_of(mask):
    return frozenset(LAW9[i] for i in range(NL) if mask >> i & 1)


def shapley_from_table(v):
    """v: array over 512 masks (cost). Returns saving-Shapley phi[f] = sum w [v(U) - v(U+f)]."""
    v = np.asarray(v, dtype=float)
    fac = math.factorial
    w = [fac(s) * fac(NL - s - 1) / fac(NL) for s in range(NL)]
    pc = np.array([bin(m).count('1') for m in ALLMASKS])
    phi = np.zeros(NL)
    for fi in range(NL):
        bit = 1 << fi
        U = np.array([m for m in ALLMASKS if not m & bit])
        phi[fi] = np.sum(np.array([w[pc[m]] for m in U]) * (v[U] - v[U | bit]))
    return phi


def fit_freq(truth, level):
    """C2: fit kind/arity/role frequencies on the all-explicit empty-theory reps (C1-CSE'd)."""
    trees = []
    for s in CP.corpus():
        form = 'newton' if s.fam == 'P' else 'explicit'
        rep = R.realize(s, *(R.build_newton(s, frozenset(), lambda e: 0) if s.fam == 'P'
                             else R.build_explicit_G(s, truth)), frozenset(), level, 'C1', 16)
        trees += [tid for _, tid in rep.items]
    TR.FREQ.fit(trees)
    return TR.FREQ.to_json()


class Job:
    def __init__(self, code, level, B=16, check=True, log=print, corp=None, truth=None):
        self.code, self.level, self.B, self.check = code, level, B, check
        self.log = log
        self.truth = truth if truth is not None else CP.build_truth()
        self.corp = corp if corp is not None else CP.corpus()
        if code == 'C2':
            self.freq = fit_freq(CP.build_truth(), level)
        self.reps = {}          # (sid, form, S) -> best Rep
        self.alts = {}          # (sid, form, S) -> list of Rep (all alternatives)
        self.errors = []        # (sid, form, S, variant, err)
        self.decoders = {s.sid: SM.Decoder(s, self.truth) for s in self.corp} if check else {}

    # ------------------------------------------------------------ reps
    def rep(self, s, form, S):
        key = (s.sid, form, S)
        if key not in self.reps:
            alts = R.build_all(s, form, S, self.level, self.code, self.B, self.truth)
            if self.check:
                for a in alts:
                    err = self.decoders[s.sid].check(a)
                    self.errors.append((s.sid, form, sorted(S), a.meta['variant'], err))
            self.alts[key] = alts
            self.reps[key] = min(alts, key=lambda r: r.cost) if alts else None
        return self.reps[key]

    def sys_best(self, s, Tk):
        """L*_k for theory restricted to A_k; returns (cost, rep)."""
        forms = [f for f in R.formulations(s, self.level) if f != 'el' or 'EC' in Tk]
        flag = math.log2(len(forms)) if len(forms) > 1 else 0.0
        best = None
        for f in forms:
            r = self.rep(s, f, R.form_subset(s, f, Tk))
            if r is None:
                continue
            if best is None or r.cost < best.cost - 1e-12:
                best = r
        return best.cost + flag, best, flag

    # ------------------------------------------------------------ full lattice
    def run(self):
        t0 = time.time()
        self.Lk = {}           # sid -> np.array over 512 masks
        self.choice = {}       # (sid, mask) -> (rep, flag)
        for s in self.corp:
            arr = np.zeros(len(ALLMASKS))
            cache = {}
            for m in ALLMASKS:
                Tk = set_of(m) & s.applicable
                if Tk not in cache:
                    cache[Tk] = self.sys_best(s, Tk)
                c, rep, flag = cache[Tk]
                arr[m] = c
                self.choice[(s.sid, m)] = (rep, flag)
            self.Lk[s.sid] = arr
        self.log('[%s %s B%d] reps done (%d reps) %.0fs' % (self.code, self.level, self.B, len(self.reps), time.time() - t0))
        # law statements
        lawc = {}
        self.LS = np.zeros(len(ALLMASKS))
        self.CC = np.zeros(len(ALLMASKS))
        self.CCshare = {s.sid: np.zeros(len(ALLMASKS)) for s in self.corp}
        self.used_templates = {}
        for m in ALLMASKS:
            T = set_of(m)
            used = set()
            refs = {}
            for s in self.corp:
                rep, _ = self.choice[(s.sid, m)]
                used |= rep.templates
                for sym in rep.licensed_refs:
                    refs.setdefault(sym, []).append(s.sid)
            used = frozenset(u for u in used if isinstance(u, str))
            self.used_templates[m] = used
            tot = 0.0
            for f in T:
                k2 = (f, 'DIST' in T, len(R.lib_names(T)), used if f == 'HK' else None)
                if k2 not in lawc:
                    lawc[k2] = LW.law_cost(f, T, used, self.code, self.level)
                tot += lawc[k2]
            self.LS[m] = tot
            if self.code == 'C4':
                self.CC[m] = self.B * len(refs)
                for sym, users in refs.items():
                    for sid in users:
                        self.CCshare[sid][m] += self.B / len(users)
        self.dec = LW.dec_cost(self.code, self.level)
        self.L = self.LS + self.CC + sum(self.Lk.values())
        self.law_single = {f: LW.law_cost(f, frozenset([f]), self.used_templates[mask_of([f])], self.code, self.level)
                           for f in LAW9}
        self.law_single['POS'] = LW.law_cost('POS', frozenset(), set(), 'C1' if self.code == 'C4' else self.code, self.level)
        self.log('[%s %s B%d] lattice done %.0fs' % (self.code, self.level, self.B, time.time() - t0))
        return self

    # ------------------------------------------------------------ values
    def values(self, charged=False):
        L = self.L + (self.dec * np.array([1.0 if m >> LAW9.index('EC') & 1 else 0.0 for m in ALLMASKS])
                      if charged else 0.0)
        full = (1 << NL) - 1
        out = {}
        phi_sys = {}
        for s in self.corp:
            v = self.Lk[s.sid] + self.CCshare[s.sid]
            phi_sys[s.sid] = shapley_from_table(v)
        LSv = self.LS + (L - self.L)
        phi_ls = shapley_from_table(LSv)
        Phi = sum(phi_sys.values()) + phi_ls
        for fi, f in enumerate(LAW9):
            bit = 1 << fi
            add = L[0] - L[bit]
            loo = L[full ^ bit] - L[full]
            out[f] = dict(Phi=float(Phi[fi]), add=float(add), loo=float(loo), mid=0.5 * float(add + loo))
        return dict(L0=float(L[0]), Lfull=float(L[full]), values=out,
                    phi_sys={sid: [float(x) for x in p] for sid, p in phi_sys.items()},
                    phi_ls=[float(x) for x in phi_ls],
                    efficiency_gap=float(sum(Phi) - (L[0] - L[full])),
                    L_all=[float(x) for x in L])

    # ------------------------------------------------------------ miner
    def mine(self, mask, check=True, per_system=None, max_iter=64):
        sysreps = []
        for s in self.corp:
            if per_system is not None and s.sid != per_system:
                continue
            rep, flag = self.choice[(s.sid, mask)]
            sysreps.append((s, rep, flag))
        extra = 0.0 if per_system is not None else float(self.LS[mask] + self.CC[mask])
        mn = MN.Miner(self.code, self.B, self.level, sysreps, extra, log=self.log, max_iter=max_iter)
        t0 = mn.total()
        if per_system is None:
            assert abs(t0 - self.L[mask]) < 1e-6, (t0, self.L[mask])
        total = mn.run()
        out = dict(mask=mask, theory=sorted(set_of(mask)), L_off=float(t0), L_on=float(total),
                   ntemplates=len(mn.templates), ndict=len(mn.dict_vals), history=mn.history)
        if self.code == 'C4':
            refs = set()
            for s, rep, _ in sysreps:
                refs |= rep.licensed_refs
            out['pos_sign_known'] = mn.sign_known_count() + sum(1 for sym in refs if R.sign_known(sym))
        if check:
            out['U6_roundtrip'] = mn.roundtrip_ok()
            tenv = mn.template_env()
            worst = 0.0
            for m in mn.systems:
                prep = R.Rep()
                prep.sid, prep.items, prep.consts, prep.meta = m.sid, m.items, dict(m.consts), m.rep.meta
                prep.form = m.rep.form
                worst = max(worst, self.decoders[m.sid].check(prep, templates=tenv))
            out['U5_maxerr'] = worst
        return out

    # ------------------------------------------------------------ EC scaling (spec 7.3)
    def ec_surplus(self, s, check=True):
        """s_k(EC) = L+(M_k | T_EC-) - L+(M_k | T_EC+), per-system miner (the system alone)."""
        res = {}
        for tag, Tk in (('plus', s.applicable), ('minus', s.applicable - {'EC'})):
            cost, rep, flag = self.sys_best(s, frozenset(Tk))
            extra = self.B * len(rep.licensed_refs) if self.code == 'C4' else 0.0
            mn = MN.Miner(self.code, self.B, self.level, [(s, rep, flag)], extra, log=self.log)
            t0 = mn.total()
            assert abs(t0 - (cost + extra)) < 1e-6, (t0, cost, extra)
            tot = mn.run()
            ok = mn.roundtrip_ok() if check else None
            res[tag] = dict(L_off=float(cost + extra), L_on=float(tot), form=rep.form, variant=rep.meta['variant'],
                            ntemplates=len(mn.templates), U6=ok)
        res['surplus_on'] = res['minus']['L_on'] - res['plus']['L_on']
        res['surplus_off'] = res['minus']['L_off'] - res['plus']['L_off']
        return res

    # ------------------------------------------------------------ C4 POS counts
    def pos_counts(self, mask):
        """#sign-known constants stored (local lumped + corpus-licensed masses/springs) under theory mask."""
        n = 0
        refs = set()
        for s in self.corp:
            rep, _ = self.choice[(s.sid, mask)]
            n += len(rep.sign_known)
            refs |= rep.licensed_refs
        n += sum(1 for sym in refs if R.sign_known(sym))
        return n

    # ------------------------------------------------------------ C8 corpus text
    def corpus_text(self, mask):
        parts = []
        for s in self.corp:
            rep, _ = self.choice[(s.sid, mask)]
            for lab, tid in rep.items:
                parts.append(sp.srepr(TR.to_sympy_print(tid)))
        T = set_of(mask)
        for f in sorted(T):
            for e, fm in LW.law_items(f, 'DIST' in T, self.used_templates[mask]):
                parts.append(sp.srepr(e))
        return '\n'.join(parts)

    def summary(self):
        return dict(code=self.code, level=self.level, B=self.B, nreps=len(self.reps),
                    max_err=max((e[-1] for e in self.errors), default=0.0),
                    law_single=self.law_single, dec=self.dec)

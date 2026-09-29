"""Generic corpus-level greedy tree-grammar miner (spec 1.6): the K2-collapse instrument.

Candidates: (a) exact subtrees (3-30 nodes) occurring >= 2 times corpus-wide; (b) Plotkin LGG of
pairs of distinct subtrees in the same (head kind, arity) group (<= 50,000 seeded random pairs per
group, all pairs if fewer), >= 2 non-hole nodes, <= 4 holes (nonlinear holes allowed);
(c) C4 only: corpus dictionary of lumped constants identical in >= 2 systems.
Scoring: savings = sum_matches [L(match) - L(call) - sum_h L(arg_h)] - L(def) - dLib, greedy
non-overlapping preorder ORDERED matching.  Loop: rank by count*size, score top 200 exactly, add the
best if savings > 0 (and, implementation guard, only if the exact corpus length decreases), rewrite
all matches; <= 64 iterations.

Template definitions (implementation choice): gamma(#holes+1) + body tree whose leaves are holes or
global symbols (C1: log2(#holes + #global symbols)); calls to earlier templates allowed.
Holes and system-local leaves (temps, output refs, local C4 constants) are never concrete template
leaves (LGG maps them to holes).
"""
import math, random, itertools, collections, time
import sympy as sp
from common import gamma_len, LOG95, SEED
import trees as TR
import reps as R

MAXH = 4
MINS, MAXS = 3, 30


def is_local(label):
    return label[0] in ('t', 'o', 'c', 'h')


class MSys:
    __slots__ = ('sid', 'items', 'ctx', 'nlib0', 'flag', 'consts', 'nlic', 'rep', 'dmap')


class Miner:
    def __init__(self, code, B, level, sysreps, extra_cost, seed=SEED, log=None, max_iter=64, topk=200):
        """sysreps: list of (sysm, rep, flag_bits). extra_cost: law statements + CC (constant)."""
        self.code, self.B, self.level = code, B, level
        self.extra = extra_cost
        self.rng = random.Random(seed)
        self.max_iter, self.topk = max_iter, topk
        self.log = log or (lambda *a: None)
        self.templates = []                 # list of (nholes, body_id)
        self.dict_vals = []                 # C4 dictionary constant values
        self.systems = []
        for sysm, rep, flag in sysreps:
            m = MSys()
            m.sid, m.items, m.flag, m.rep = sysm.sid, list(rep.items), flag, rep
            m.ctx = R.SysCtx(sysm, rep.form, rep.S, level, code)
            m.nlib0 = m.ctx.nlib
            m.consts = dict(rep.consts)
            lic = R.licensed_symbols(sysm, rep.S) if code == 'C4' else set()
            m.nlic = sum(1 for s in sysm.params if s in lic)
            m.dmap = {}
            self.systems.append(m)
        self.input_items = {m.sid: list(m.items) for m in self.systems}
        glob = set()
        for m in self.systems:
            for _, tid in m.items:
                for lab in TR.leaves(tid):
                    if lab[0] == 's':
                        glob.add(lab)
        self.n_glob_state = sum(1 for l in glob if TR.leaf_role(l) == 'state')
        self.n_glob_param = sum(1 for l in glob if TR.leaf_role(l) == 'param')

    # ------------------------------------------------------------ exact cost
    def nlib(self, m):
        return m.nlib0 + len(self.templates)

    def sys_cost(self, m):
        ctx = m.ctx
        ctx.nlib = self.nlib(m)
        if self.code == 'C4':
            nloc = sum(1 for l in m.consts if l[0] == 'c')
            c = R.items_cost(m.items, 'C4', ctx, n_param=self.scope_consts(m) + m.nlic + len(self.dict_vals))
            c += self.B * nloc
        else:
            c = R.items_cost(m.items, self.code, ctx)
        ctx.nlib = m.nlib0
        return c + m.flag

    @staticmethod
    def scope_consts(m):
        """local constants that are tree leaves (PC decoder constants, index >= 1000, are not)."""
        return sum(1 for l in m.consts if l[0] == 'c' and l[2] < 1000)

    def def_cost(self, nh, body, j):
        if self.code == 'C3':
            return (3 + TR.c3_len(body)) * LOG95
        tc = 'C1' if self.code in ('C4', 'C5') else self.code
        nlib = max(m.nlib0 for m in self.systems) + j
        sizes = {'hole': nh, 'state': self.n_glob_state, 'param': self.n_glob_param + len(self.dict_vals)}
        from common import builtins_for
        return gamma_len(nh + 1) + TR.tree_cost(body, tc, sizes, nlib, len(builtins_for(self.level)))

    def total(self):
        t = self.extra + sum(self.sys_cost(m) for m in self.systems)
        t += sum(self.def_cost(nh, b, j) for j, (nh, b) in enumerate(self.templates))
        t += self.B * len(self.dict_vals) if self.code == 'C4' else 0.0
        return t

    # ------------------------------------------------------------ C4 dictionary
    def run_dictionary(self):
        groups = collections.defaultdict(list)
        for m in self.systems:
            for lab, v in m.consts.items():
                if lab[0] == 'c':
                    groups[v].append((m, lab))
        cand = [(v, users) for v, users in groups.items() if len({u[0].sid for u in users}) >= 2]
        cand.sort(key=lambda x: (-len(x[1]), sp.srepr(x[0])))
        base = self.total()
        for v, users in cand:
            gid = len(self.dict_vals)
            saved = [(m, list(m.items), dict(m.consts), dict(m.dmap)) for m, _ in users]
            self.dict_vals.append(v)
            dl = TR.leaf(('d', gid))
            for m, lab in users:
                cl = TR.leaf(lab)
                memo = {}
                m.items = [(l2, TR.subst(t2, {cl: dl}, memo)) for l2, t2 in m.items]
                del m.consts[lab]
                m.consts[('d', gid)] = v
                m.dmap[('d', gid)] = lab
            new = self.total()
            if new < base - 1e-9:
                base = new
            else:
                self.dict_vals.pop()
                for m, it, cs, dm in saved:
                    m.items, m.consts, m.dmap = it, cs, dm
        return base

    # ------------------------------------------------------------ candidates
    def all_roots(self):
        for m in self.systems:
            for _, tid in m.items:
                yield m, tid

    def position_counts(self):
        cnt = collections.Counter()
        stack = [tid for _, tid in self.all_roots()]
        while stack:
            j = stack.pop()
            cnt[j] += 1
            stack.extend(TR.NODES[j][2])
        return cnt

    def has_local(self, i, memo={}):
        r = memo.get(i)
        if r is None:
            kind, label, ch = TR.NODES[i]
            r = (kind == 'Sym' and is_local(label)) or any(self.has_local(c) for c in ch)
            memo[i] = r
        return r

    def lgg(self, a, b, holes):
        if a == b and not self.has_local(a):
            return a
        ka, la, ca = TR.NODES[a]
        kb, lb, cb = TR.NODES[b]
        if ka == kb and la == lb and len(ca) == len(cb) and ca:
            return TR.mk(ka, la, tuple(self.lgg(x, y, holes) for x, y in zip(ca, cb)))
        key = (a, b)
        h = holes.get(key)
        if h is None:
            h = len(holes)
            holes[key] = h
            if h >= MAXH:
                raise OverflowError
        return TR.leaf(('h', h))

    def nonhole(self, p, memo={}):
        r = memo.get(p)
        if r is None:
            kind, label, ch = TR.NODES[p]
            r = 0 if (kind == 'Sym' and label[0] == 'h') else 1 + sum(self.nonhole(c) for c in ch)
            memo[p] = r
        return r

    def nholes(self, p):
        return len({l for l in TR.leaves(p) if l[0] == 'h'})

    def candidates(self):
        cnt = self.position_counts()
        cands = {}
        distinct = [j for j in cnt if MINS <= TR.SIZE[j] <= MAXS]
        distinct.sort()
        for j in distinct:
            if cnt[j] >= 2 and not self.has_local(j):
                cands[j] = cnt[j] * TR.SIZE[j]
        groups = collections.defaultdict(list)
        for j in distinct:
            k, l, ch = TR.NODES[j]
            groups[(k, len(ch))].append(j)
        gen = collections.defaultdict(set)
        for key in sorted(groups, key=str):
            g = groups[key]
            n = len(g)
            npairs = n * (n - 1) // 2
            if npairs <= 50000:
                pairs = itertools.combinations(g, 2)
            else:
                pairs = set()
                while len(pairs) < 50000:
                    a, b = self.rng.sample(g, 2)
                    pairs.add((min(a, b), max(a, b)))
                pairs = sorted(pairs)
            for a, b in pairs:
                try:
                    p = self.lgg(a, b, {})
                except OverflowError:
                    continue
                if self.nonhole(p) < 2 or p in (a, b):
                    continue
                gen[p].add(a)
                gen[p].add(b)
        for p, subs in gen.items():
            h = sum(cnt[s] for s in subs) * self.nonhole(p)
            cands[p] = max(cands.get(p, 0), h)
        ranked = sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))
        return [p for p, _ in ranked[:self.topk]]

    # ------------------------------------------------------------ matching
    def match(self, p, t, bind):
        kp, lp, chp = TR.NODES[p]
        if kp == 'Sym' and lp[0] == 'h':
            h = lp[1]
            if h in bind:
                return bind[h] == t
            bind[h] = t
            return True
        if p == t and not chp:
            return True
        kt, lt, cht = TR.NODES[t]
        if kp != kt or lp != lt or len(chp) != len(cht):
            return False
        for a, b in zip(chp, cht):
            if not self.match(a, b, bind):
                return False
        return True

    def find(self, p, t, nh, memo):
        """Greedy preorder non-overlapping matches in subtree t: Counter of (t_match, args)."""
        r = memo.get(t)
        if r is not None:
            return r
        bind = {}
        if self.match(p, t, bind):
            args = tuple(bind[h] for h in range(nh))
            r = collections.Counter({(t, args): 1})
            for a in args:
                r.update(self.find(p, a, nh, memo))
        else:
            r = collections.Counter()
            for c in TR.NODES[t][2]:
                r.update(self.find(p, c, nh, memo))
        memo[t] = r
        return r

    def rewrite(self, p, t, nh, label, memo):
        r = memo.get(t)
        if r is not None:
            return r
        bind = {}
        if self.match(p, t, bind):
            r = TR.mk('Call', label, tuple(self.rewrite(p, bind[h], nh, label, memo) for h in range(nh)))
        else:
            kind, lab, ch = TR.NODES[t]
            r = TR.mk(kind, lab, tuple(self.rewrite(p, c, nh, label, memo) for c in ch)) if ch else t
        memo[t] = r
        return r

    # ------------------------------------------------------------ scoring
    def item_sizes(self, m):
        """yield (tid, sizes) per item (scope as in reps.items_cost)."""
        ntemps = sum(1 for l, _ in m.items if l is not None)
        nparam = m.ctx.n_param
        if self.code == 'C4':
            nparam = self.scope_consts(m) + m.nlic + len(self.dict_vals)
        ti = oi = 0
        for lab, tid in m.items:
            if lab is not None:
                sizes = {'state': m.ctx.n_state, 'param': nparam, 'temp': ti, 'out': 0}
                ti += 1
            else:
                sizes = {'state': m.ctx.n_state, 'param': nparam, 'temp': ntemps, 'out': oi}
                oi += 1
            yield tid, sizes

    def tcost(self, t, sizes, nlib, nb):
        if self.code == 'C3':
            return TR.c3_len(t) * LOG95
        tc = 'C1' if self.code in ('C4', 'C5') else self.code
        return TR.tree_cost(t, tc, sizes, nlib, nb)

    def call_cost(self, nh, nlib):
        if self.code == 'C3':
            return (2 + 2 + max(nh - 1, 0)) * LOG95
        kind = 3.0 if self.code != 'C2' else TR.FREQ.kind_bits('Call')
        return kind + math.log2(nlib + 1)

    def score(self, p):
        nh = self.nholes(p)
        memo = {}
        from common import builtins_for
        nb = len(builtins_for(self.level))
        sav = 0.0
        nmatch = 0
        dlib = 0.0
        for m in self.systems:
            nlib = self.nlib(m)
            ncalls_existing = 0
            for tid, sizes in self.item_sizes(m):
                cnt = self.find(p, tid, nh, memo)
                for (t, args), c in cnt.items():
                    s = self.tcost(t, sizes, nlib, nb) - self.call_cost(nh, nlib) - \
                        sum(self.tcost(a, sizes, nlib, nb) for a in args)
                    sav += c * s
                    nmatch += c
                ncalls_existing += TR.profile(tid)[4]
            if self.code != 'C3':
                dlib += ncalls_existing * (math.log2(nlib + 1) - math.log2(max(nlib, 1)))
        sav -= self.def_cost(nh, p, len(self.templates)) + dlib
        return sav, nmatch

    # ------------------------------------------------------------ main loop
    def run(self):
        t0 = time.time()
        if self.code == 'C4':
            self.run_dictionary()
        base = self.total()
        self.start_total = base
        self.history = []
        rejected = set()
        for it in range(self.max_iter):
            cands = [p for p in self.candidates() if p not in rejected]
            scored = []
            for p in cands:
                s, nm = self.score(p)
                if s > 1e-9 and nm > 0:
                    scored.append((s, p))
            scored.sort(key=lambda x: (-x[0], x[1]))
            accepted = False
            for s, p in scored:
                nh = self.nholes(p)
                label = ('M', len(self.templates))
                saved = [(m, list(m.items)) for m in self.systems]
                memo = {}
                for m in self.systems:
                    m.items = [(l2, self.rewrite(p, t2, nh, label, memo)) for l2, t2 in m.items]
                self.templates.append((nh, p))
                new = self.total()
                if new < base - 1e-9:
                    self.history.append(dict(it=it, pattern=TR.SIZE[p], holes=nh, est=s, exact=base - new))
                    base = new
                    accepted = True
                    break
                self.templates.pop()
                for m, items in saved:
                    m.items = items
                rejected.add(p)
            if not accepted:
                break
        self.final_total = base
        self.log('miner %s %s: %d templates, %d dict, %.1f -> %.1f (%.1fs)' % (
            self.code, self.level, len(self.templates), len(self.dict_vals), self.start_total, base, time.time() - t0))
        return base

    # ------------------------------------------------------------ U6 / U5 support
    def expand(self, t, memo=None):
        if memo is None:
            memo = {}
        r = memo.get(t)
        if r is not None:
            return r
        kind, label, ch = TR.NODES[t]
        if kind == 'Call' and isinstance(label, tuple):
            nh, body = self.templates[label[1]]
            args = [self.expand(c, memo) for c in ch]
            mapping = {TR.leaf(('h', h)): args[h] for h in range(nh)}
            r = self.expand(TR.subst(body, mapping, {}), memo)
        elif ch:
            r = TR.mk(kind, label, tuple(self.expand(c, memo) for c in ch))
        else:
            r = t
        memo[t] = r
        return r

    def roundtrip_ok(self):
        memo = {}
        for m in self.systems:
            inv = {TR.leaf(d): TR.leaf(c) for d, c in m.dmap.items()}
            got = [(l, TR.subst(self.expand(t, memo), inv, {})) for l, t in m.items]
            if got != self.input_items[m.sid]:
                return False
        return True

    def template_env(self):
        return {('M', j): (nh, b) for j, (nh, b) in enumerate(self.templates)}

    def sign_known_count(self):
        """POS with miner: sign-known distinct stored constants (local + dictionary)."""
        n = 0
        for m in self.systems:
            n += sum(1 for l, v in m.consts.items() if l[0] == 'c' and R.sign_known(v))
        n += sum(1 for v in self.dict_vals if R.sign_known(v))
        return n

"""Design evidence for PROTOCOL_DRAFT v0.3 (round-2 response). Not the frozen fixture code (that is code/fixtures.py, §14).

Simulates D1 for three O2 window-value rules on synthetic S1 windows (10 per draw), stdlib only:
  PEAK (v0.3 primary): per size k in {2, 3}, the adjacent group with the largest m(G) (ties: smallest slot tuple);
       window value = mean d over the matched peaks; P = testable windows with >= 1 matched peak.
  ALL  (v0.2 rule, v0.3 secondary): mean over k of the mean d over every matched adjacent group.
  EVID (round-2 option a): as ALL, restricted to groups containing an adjacent pair with e > 0.25 in both slots.
T: >= 2 singles with e > 0.25. O3: emergent flips only, fragile windows (>= 3/8 single flips) excluded.
D1 as in v0.3 §9: |T| < 6 -> INCONCLUSIVE_T; |P| < 6 -> INSEPARABLE; SUPPORTED: exact one-sided Wilcoxon p < 0.05,
HL >= 0.25 and O3 not reversed; NOT SUPPORTED: one-sided 95 % upper bound of HL < 0.25 and O3 not positive-dominant;
otherwise INCONCLUSIVE. (For ALL/EVID the same D1 is applied with their own P.)

Scorers: S(G) = A*tanh((E - sum_{i in G} x_i - off)/w) (evidence-only), L = [S > 0]; or the convex null g(sum x).
Planted synergies (+syn nat added to Delta(G)):
  event    - G contains two consecutive slots of the planted event;
  evidence - G contains two consecutive slots that both have e > 0.25.
Matching (v0.3/v0.4): D <= 0.25 + 1e-9 and m(H) >= m(G) - SLACK - 1e-9 with SLACK = 0 (v0.2 used 0.05);
G* = argmin (D, |m(H) - m(G)|, tuple). "Above resolution" is e > 0.25 + 1e-9 everywhere; D1 thresholds are exact.
Seeds (normative for experiment03A fixtures): random.Random(f"experiment03-designsim:{mix}:{A}:{w}:{syn}:{mode}:{margin}")
for tanh cells (Python str() of each value) and f"experiment03-designsim:convex:{mix}:{a}" for convex cells.
Usage: python3 -I -B round2_design_sim.py [draws] [order] [slack] [variant]
  order "D" (default) or "sum" (argmin (|m(H) - m(G)|, D, tuple)); slack 0 (default, v0.3) or 0.05 (v0.2);
  variant "none" (default), "dom" (elementwise dominance: every sorted e of H in [e of G, e of G + 0.25]),
  "reltol15" / "reltol25" (composition tolerance max(0.25, r * max e(G)), r = 0.15 / 0.25).
"""
import itertools, math, random, statistics, sys
from functools import lru_cache

NA = {k: [t for t in itertools.combinations(range(8), k) if all(t[j + 1] - t[j] > 1 for j in range(k - 1))] for k in (2, 3)}
ADJ = {2: [(i, i + 1) for i in range(7)], 3: [(i, i + 1, i + 2) for i in range(6)]}
GROUPS = ADJ[2] + ADJ[3]
TOL = 1e-9
ORDER = "D"
SLACK = 0.0
VARIANT = "none"

def match(e, G):
    k = len(G); sG = sorted((e[i] for i in G), reverse=True); mG = sum(sG); best = None
    for H in NA[k]:
        sH = sorted((e[i] for i in H), reverse=True); mH = sum(sH)
        D = max(abs(a - b) for a, b in zip(sG, sH))
        if VARIANT == "dom":
            ok = all(g - TOL <= x <= g + 0.25 + TOL for x, g in zip(sH, sG))
        else:
            tol = max(0.25, {"reltol15": 0.15, "reltol25": 0.25}.get(VARIANT, 0.0) * max(sG))
            ok = D <= tol + TOL and mH >= mG - SLACK - TOL
        if ok:
            key = (D, abs(mH - mG), H) if ORDER == "D" else (abs(mH - mG), D, H)
            if best is None or key < best: best = key
    return best[-1] if best else None

@lru_cache(maxsize=None)
def wcounts(n):
    M = n * (n + 1) // 2; cnt = [0] * (M + 1); cnt[0] = 1
    for r in range(1, n + 1):
        new = cnt[:]
        for s in range(M - r + 1):
            if cnt[s]: new[s + r] += cnt[s]
        cnt = new
    return cnt

def ranks_abs(v):
    n = len(v); order = sorted(range(n), key=lambda i: abs(v[i])); r = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(v[order[j + 1]]) == abs(v[order[i]]): j += 1
        for t in range(i, j + 1): r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r

def wilcoxon_p(v):
    v = [a for a in v if a != 0]; n = len(v)
    if n == 0: return 1.0
    r = ranks_abs(v); W = sum(x for x, a in zip(r, v) if a > 0)
    if len(set(r)) == n:
        return sum(wcounts(n)[int(round(W)):]) / 2 ** n
    return sum(sum(x for x, b in zip(r, s) if b) >= W - 1e-12 for s in itertools.product((0, 1), repeat=n)) / 2 ** n

def hl(v):
    v = [a for a in v if a != 0]; n = len(v)
    if n == 0: return 0.0, -math.inf, math.inf
    wa = sorted((v[i] + v[j]) / 2 for i in range(n) for j in range(i, n)); M = len(wa)
    cnt = wcounts(n); cum = 0; k = 0
    for s in range(len(cnt)):
        if (cum + cnt[s]) / 2 ** n <= 0.05: cum += cnt[s]; k = s + 1
        else: break
    return statistics.median(wa), (wa[k - 1] if k else -math.inf), (wa[M - k] if k else math.inf)

def window(x, delta0, label_from_S, S0, syn, mode, event):
    """x: per-slot evidence; delta0(G): evidence-only Delta; returns per-rule (value, o3) or None if not in P."""
    e = [delta0((i,)) for i in range(8)]
    ev = {i for i in range(8) if e[i] > 0.25 + TOL}          # above resolution (protocol §2)
    def bonus(G):
        if not syn: return 0.0
        pairs = [(i, i + 1) for i in range(7) if i in G and i + 1 in G]
        if mode == "event": return syn if any(a in event and b in event for a, b in pairs) else 0.0
        return syn if any(a in ev and b in ev for a, b in pairs) else 0.0
    delta = lambda G: delta0(G) + bonus(G)
    flip = lambda G: not label_from_S(S0 - delta(G))
    if sum(v > 0.25 + TOL for v in e) < 2: return {"T": False}          # testable: >= 2 singles with e > 0.25 (signed)
    m = {G: sum(e[i] for i in G) for G in GROUPS}
    H = {G: match(e, G) for G in GROUPS}
    sflip = [flip((i,)) for i in range(8)]
    fragile = sum(sflip) >= 3
    def agg(groups):
        dk, ok = [], []
        for k in (2, 3):
            gs = [G for G in groups if len(G) == k and H[G] is not None]
            if gs: dk.append(statistics.mean(delta(G) - delta(H[G]) for G in gs))
            # O3 decision role (v0.3): emergent flips only (no single in G or G* flips); fragile windows excluded
            o = [] if fragile else [int(flip(G)) - int(flip(H[G])) for G in gs if not any(sflip[i] for i in G) and not any(sflip[i] for i in H[G])]
            if o: ok.append(statistics.mean(o))
        return (statistics.mean(dk) if dk else None), (statistics.mean(ok) if ok else None)
    peaks = [min(ADJ[k], key=lambda G: (-m[G], G)) for k in (2, 3)]
    evid = [G for G in GROUPS if any(i in ev and i + 1 in ev for i in G if i + 1 in G)]
    return {"T": True, "PEAK": agg(peaks), "ALL": agg(GROUPS), "EVID": agg(evid)}

def d1(ws, rule):
    T = [w for w in ws if w is not None and w["T"]]
    if len(T) < 6: return "INCONCLUSIVE_T", None
    P = [w[rule] for w in T if w[rule][0] is not None]
    if len(P) < 6: return "INSEPARABLE", None
    v = [a for a, _ in P]; o3 = [b for _, b in P if b is not None]
    pos, neg = sum(b > 0 for b in o3), sum(b < 0 for b in o3)
    p = wilcoxon_p(v); h, lo, hi = hl(v); med = statistics.median(v)
    if p < 0.05 and h >= 0.25 and pos >= neg: return "SUPPORTED", med
    if hi < 0.25 and pos <= neg: return "NOT_SUPPORTED", med
    return "INCONCLUSIVE", med

def gen(rng, kind, s):
    """Evidence layouts, in units of the tanh width s. Returns x and the planted event slots."""
    if kind == "contig":            # localized event, 2-3 consecutive strong slots, near-zero background
        x = [rng.uniform(0, 0.05 * s) for _ in range(8)]; L = rng.choice([2, 3]); a = rng.randrange(0, 9 - L)
        ev = set(range(a, a + L))
    elif kind == "scatter":         # 2-3 nonadjacent strong slots
        x = [rng.uniform(0, 0.05 * s) for _ in range(8)]; ev = set(rng.choice(NA[rng.choice([2, 3])]))
    elif kind == "spread":          # full coverage, no event
        return [rng.uniform(0.05, 0.5) * s for _ in range(8)], set()
    elif kind == "event3":          # 3 consecutive event slots on an informative background
        x = [rng.uniform(0.1, 0.5) * s for _ in range(8)]; a = rng.randrange(0, 6); ev = {a, a + 1, a + 2}
        for i in ev: x[i] = rng.uniform(0.3, 0.6) * s
        return x, ev
    elif kind == "plateau3":        # 3 consecutive near-equal slots on a weak background (round-2 completeness null)
        x = [rng.uniform(0, 0.05 * s) for _ in range(8)]; a = rng.randrange(0, 6); v = rng.uniform(0.4, 1.2) * s
        for i in (a, a + 1, a + 2): x[i] = v * rng.uniform(0.97, 1.03)
        return x, {a, a + 1, a + 2}
    elif kind == "plateau_bg":      # 3 consecutive near-equal slots on an informative background
        x = [rng.uniform(0.1, 0.5) * s for _ in range(8)]; a = rng.randrange(0, 6); v = rng.uniform(0.5, 1.0) * s
        for i in (a, a + 1, a + 2): x[i] = v * rng.uniform(0.97, 1.03)
        return x, {a, a + 1, a + 2}
    elif kind == "plateau_adv":     # adversarial: strong, deeply saturated near-equal plateau (used with high margins)
        x = [rng.uniform(0, 0.02 * s) for _ in range(8)]; a = rng.randrange(0, 6); v = rng.uniform(1.0, 1.6) * s
        for i in (a, a + 1, a + 2): x[i] = v * rng.uniform(0.97, 1.03)
        return x, {a, a + 1, a + 2}
    elif kind == "raccident":       # experiment02 R-accident blur profile (slots 1, 3, 4, 5), jittered and placed anywhere
        x = [rng.uniform(0, 0.02 * s) for _ in range(8)]; shift = rng.randrange(-1, 3)
        for i, v in ((1, 0.15), (3, 0.40), (4, 0.90), (5, 0.30)):
            x[(i + shift) % 8] = v * s * rng.uniform(0.8, 1.2)
        return x, {(i + shift) % 8 for i in (3, 4, 5)}
    elif kind == "heterog":         # 4-6 strong slots of heterogeneous strength, >= 1 consecutive pair among them, weak background
        x = [rng.uniform(0, 0.05 * s) for _ in range(8)]; a = rng.randrange(0, 7)
        ev = {a, a + 1} | set(rng.sample([i for i in range(8) if i not in (a, a + 1)], rng.choice([2, 3, 4])))
        for i in ev: x[i] = rng.uniform(0.3, 1.0) * s
        return x, ev
    elif kind == "twopairs":        # two separated consecutive strong pairs on a weak background (round-2 skeptic)
        x = [rng.uniform(0, 0.1 * s) for _ in range(8)]; ev = {1, 2, 5, 6}
        for i in ev: x[i] = rng.uniform(0.3, 0.6) * s
        return x, ev
    for i in ev: x[i] = rng.uniform(0.3, 1.0) * s
    return x, ev

MIXES = {
    "S1like": ["contig"] * 3 + ["event3"] * 4 + ["spread"] * 3,
    "event3": ["event3"] * 10,
    "twopairs": ["twopairs"] * 10,
    "spread": ["spread"] * 10,
    "scatter": ["scatter"] * 10,
    "contig_heavy": ["contig"] * 6 + ["event3"] * 4,
    "plateau3": ["plateau3"] * 10,
    "plateau_bg": ["plateau_bg"] * 10,
    "plateau_mix": ["plateau3"] * 4 + ["event3"] * 3 + ["spread"] * 3,
    "plateau_adv": ["plateau_adv"] * 10,
    "raccident": ["raccident"] * 10,
    "heterog": ["heterog"] * 10,
}

def run_tanh(mix, A, w, syn, mode, draws, margin, rules=("PEAK", "ALL", "EVID")):
    rng = random.Random(f"experiment03-designsim:{mix}:{A}:{w}:{syn}:{mode}:{margin}")
    out = {r: {} for r in rules}; meds = {r: [] for r in rules}
    for _ in range(draws):
        ws = []
        for kind in MIXES[mix]:
            x, ev = gen(rng, kind, w); E = sum(x); off = E - rng.uniform(*margin) * w
            S = lambda G, x=x, E=E, off=off: A * math.tanh((E - sum(x[i] for i in G) - off) / w)
            S0 = S(())
            if not S0 > 0: ws.append(None); continue
            ws.append(window(x, lambda G, S=S, S0=S0: S0 - S(G), lambda s: s > 0, S0, syn, mode, ev))
        for r in rules:
            v, med = d1(ws, r); out[r][v] = out[r].get(v, 0) + 1
            if med is not None: meds[r].append(med)
    return out, meds

def run_convex(mix, a, draws, rules=("PEAK", "ALL", "EVID")):
    rng = random.Random(f"experiment03-designsim:convex:{mix}:{a}")
    out = {r: {} for r in rules}; meds = {r: [] for r in rules}
    for _ in range(draws):
        ws = []
        for kind in MIXES[mix]:
            x, ev = gen(rng, kind, 1.0)
            g = lambda G, x=x: (lambda s: s + a * s * s)(sum(x[i] for i in G))
            S0 = 1.5   # label = 1 while removed support < 1.5 nat
            ws.append(window(x, g, lambda s: s > 0, S0, 0.0, "event", ev))
        for r in rules:
            v, med = d1(ws, r); out[r][v] = out[r].get(v, 0) + 1
            if med is not None: meds[r].append(med)
    return out, meds

def fmt(o, n):
    ev = n - o.get("INSEPARABLE", 0) - o.get("INCONCLUSIVE_T", 0)
    s = " ".join(f"{lab}={o.get(k, 0)}" for k, lab in (("SUPPORTED", "SUPPORT"), ("NOT_SUPPORTED", "NOT_SUP"), ("INCONCLUSIVE", "INCONCL"),
                                                     ("INSEPARABLE", "INSEPAR"), ("INCONCLUSIVE_T", "INCONCL_T")))
    if ev: s += f" | SUP/evaluable={o.get('SUPPORTED', 0) / ev:.2f}"
    return s

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ORDER = sys.argv[2] if len(sys.argv) > 2 else "D"
    SLACK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    VARIANT = sys.argv[4] if len(sys.argv) > 4 else "none"
    grid = [(A, w) for A in (4, 6, 8) for w in (1.5, 2, 3)]
    print(f"# draws per cell = {n}; tie-break order {ORDER}; caliper slack {SLACK}; variant {VARIANT}; 10 windows per draw; margin u = (E - off)/w ~ U(lo, hi)")
    print("## NULL (no adjacency term)")
    for margin in ((0.2, 1.5), (0.05, 0.3), (1.5, 3.0)):
        for mix in ("S1like", "event3", "twopairs", "spread", "scatter", "contig_heavy", "plateau3", "plateau_bg", "plateau_mix", "raccident", "heterog"):
            for A, w in grid:
                o, meds = run_tanh(mix, A, w, 0.0, "event", n, margin)
                print(f"null tanh margin={margin} {mix:12s} A={A} w={w}: " + " || ".join(
                    f"{r}: {fmt(o[r], n)} medO2={statistics.median(meds[r]) if meds[r] else float('nan'):+.3f} maxmed={max(meds[r]) if meds[r] else float('nan'):+.3f}" for r in o), flush=True)
    for margin in ((1.5, 3.0), (2.5, 4.0)):
        for A in (4, 8, 12):
            for w in (1.0, 1.5, 3.0):
                o, meds = run_tanh("plateau_adv", A, w, 0.0, "event", n, margin)
                print(f"null tanh-adversarial margin={margin} plateau_adv A={A} w={w}: " + " || ".join(
                    f"{r}: {fmt(o[r], n)} medO2={statistics.median(meds[r]) if meds[r] else float('nan'):+.3f} maxmed={max(meds[r]) if meds[r] else float('nan'):+.3f}" for r in o), flush=True)
    for a in (-0.1, 0.1, 0.3, 1.0):
        for mix in ("S1like", "event3", "spread"):
            o, meds = run_convex(mix, a, n)
            print(f"null convex a={a:+.1f} {mix:12s}: " + " || ".join(
                f"{r}: {fmt(o[r], n)} medO2={statistics.median(meds[r]) if meds[r] else float('nan'):+.3f} maxmed={max(meds[r]) if meds[r] else float('nan'):+.3f}" for r in o), flush=True)
    print("## POWER (+0.5 and +1.0 nat)")
    for syn in (0.5, 1.0):
        for mode, mixes in (("event", ("event3", "twopairs", "S1like", "plateau_bg", "raccident", "heterog")), ("evidence", ("spread", "event3", "twopairs", "S1like", "plateau_bg", "raccident", "heterog"))):
            for mix in mixes:
                for A, w in grid:
                    o, meds = run_tanh(mix, A, w, syn, mode, n, (0.2, 1.5))
                    print(f"power syn={syn} {mode:8s} {mix:12s} A={A} w={w}: " + " || ".join(
                        f"{r}: {fmt(o[r], n)} medO2={statistics.median(meds[r]) if meds[r] else float('nan'):+.3f}" for r in o), flush=True)

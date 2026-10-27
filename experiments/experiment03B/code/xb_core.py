"""experiment03B core: masks, effects, fixed/adaptive rankings, ties, primary statistic, video-cluster bootstrap.

Pure Python (standard library only). No model, no I/O. PROTOCOL.md §4-§7 and RECORDS.md are the specification:
  D(G | R) = q * [S(x_R) - S(x_(R u G))], with q = +1 if L0 = 1, -1 if L0 = 0, locked to the baseline label.
Masks are tuples of distinct slots sorted ascending; () is the unmasked input. Slots are 0..7.
"""
import itertools
import math
import random

N_SLOTS = 8
SLOTS = tuple(range(N_SLOTS))
BUDGETS = (1, 2, 3, 4)
DEVELOPMENT_VIDEOS = ("RoadAccidents127_x264", "Normal_Videos_018_x264")


def norm_mask(m):
    m = tuple(sorted(int(s) for s in m))
    if len(set(m)) != len(m) or any(s not in SLOTS for s in m):
        raise ValueError(f"invalid mask {m}")
    return m


def all_pairs():
    return list(itertools.combinations(SLOTS, 2))


def q_of(L0):
    if L0 not in (0, 1):
        raise ValueError(f"q needs a parsed baseline label, got {L0!r}")
    return 1 if L0 == 1 else -1


def effect(q, s_state, s_masked):
    """q * (S(state) - S(state with the extra group replaced)). None if either score is missing or non-finite."""
    if s_state is None or s_masked is None or not (math.isfinite(s_state) and math.isfinite(s_masked)):
        return None
    return q * (s_state - s_masked)


# ---------------------------------------------------------------- fixed ranking

def ranking(q, s0, singles):
    """Fixed singleton ranking (RECORDS.md RANKING). singles: {slot: score of x_{slot}} on one family/readout."""
    missing = [s for s in SLOTS if singles.get(s) is None]
    if s0 is None or missing:
        return {"status": "incomplete", "missing": missing if s0 is not None else list(SLOTS)}
    eff = {s: effect(q, s0, singles[s]) for s in SLOTS}
    if any(v is None for v in eff.values()):
        return {"status": "incomplete", "missing": [s for s, v in eff.items() if v is None]}
    order = sorted(SLOTS, key=lambda s: (-eff[s], s))
    by_val = {}
    for s in SLOTS:
        by_val.setdefault(eff[s], []).append(s)
    ties = sorted(sorted(g) for g in by_val.values() if len(g) > 1)
    return {"status": "complete",
            "effects": {str(s): eff[s] for s in SLOTS},
            "order": order,
            "ties": ties,
            "tie_spans_cut": {str(k): eff[order[k - 1]] == eff[order[k]] for k in BUDGETS},
            "top2_margin": eff[order[1]] - eff[order[2]],
            "all_tied": len(by_val) == 1,
            "all_nonpositive": all(v <= 0 for v in eff.values())}


def prefixes(order, budgets=BUDGETS):
    return {str(k): sorted(order[:k]) for k in budgets}


# ---------------------------------------------------------------- adaptive ranking

def adaptive_trajectory(q, score_of, max_budget=4):
    """Adaptive deletion order on one family/readout.

    score_of(mask_tuple) -> score or None. Called for () and for every candidate mask; the caller decides whether a
    call is cached or new. At each state R every remaining slot j gets effect q * (S(x_R) - S(x_(R u {j}))); the
    largest effect is selected (exact ties -> lower slot). Returns (record, complete). On a missing score the
    trajectory stops and complete is False (the state lists the missing candidates).
    """
    masked = ()
    states, order = [], []
    for step in range(1, max_budget + 1):
        state_score = score_of(masked)
        cands, missing = [], []
        for j in SLOTS:
            if j in masked:
                continue
            m = norm_mask(masked + (j,))
            sc = score_of(m)
            e = effect(q, state_score, sc)
            if e is None:
                missing.append(j)
            cands.append({"slot": j, "mask": list(m), "score": sc, "effect": e})
        if state_score is None or missing:
            states.append({"step": step, "masked": list(masked), "state_score": state_score, "candidates": cands,
                           "selected": None, "tied_max": [], "missing_candidates": missing})
            return {"states": states, "order": order}, False
        best = max(c["effect"] for c in cands)
        tied = sorted(c["slot"] for c in cands if c["effect"] == best)
        sel = tied[0]
        states.append({"step": step, "masked": list(masked), "state_score": state_score, "candidates": cands,
                       "selected": sel, "tied_max": tied[1:]})
        order.append(sel)
        masked = norm_mask(masked + (sel,))
    return {"states": states, "order": order}, True


def adaptive_planned_masks(order_so_far):
    """The candidate masks the next adaptive state needs (for planning and dry-run counts)."""
    m = norm_mask(order_so_far)
    return [norm_mask(m + (j,)) for j in SLOTS if j not in m]


# ---------------------------------------------------------------- statistics

def kendall_tau_b(x, y):
    """Kendall tau-b for paired sequences (standard tie correction). None if undefined (a constant sequence)."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return None if denom == 0 else (conc - disc) / denom


def flip_1_to_0(label):
    """F = 1 if the original label on the replaced input is 0, 0 if it is 1, None if missing."""
    return None if label is None else (1 if label == 0 else 0)


def d_w(top2, pair_labels):
    """Primary per-input statistic. pair_labels: {(i, j): label or None} for all 28 pairs.
    Returns (d, F_top2, F_random_mean, n_present)."""
    flips = {p: flip_1_to_0(pair_labels.get(p)) for p in all_pairs()}
    n_present = sum(v is not None for v in flips.values())
    ft = flips[tuple(top2)]
    if ft is None or n_present < 28:
        return None, ft, None if n_present < 28 else sum(flips.values()) / 28, n_present
    fr = sum(flips.values()) / 28
    return ft - fr, ft, fr, n_present


def d_w_bounds(top2, pair_labels):
    """Worst-case bounds: missing selected flip -> 0 (lower) / 1 (upper); missing comparator flips -> 1 / 0."""
    flips = {p: flip_1_to_0(pair_labels.get(p)) for p in all_pairs()}
    ft = flips[tuple(top2)]
    lo = (0 if ft is None else ft) - sum(1 if v is None else v for v in flips.values()) / 28
    hi = (1 if ft is None else ft) - sum(0 if v is None else v for v in flips.values()) / 28
    return lo, hi


def video_means(values_by_input, video_of):
    """Average non-missing per-input values within each video. Returns {video_id: mean} for videos with >= 1 value."""
    acc = {}
    for uid, v in values_by_input.items():
        if v is None:
            continue
        acc.setdefault(video_of[uid], []).append(v)
    return {vid: sum(vs) / len(vs) for vid, vs in acc.items()}


def _percentile_linear(sorted_vals, q):
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def cluster_bootstrap(vmeans, seed, B=10000, alpha=0.05):
    """Protocol §7.1: videos sorted by id; rng = random.Random(seed); B resamples of V videos with replacement via
    rng.randrange(V); statistic = mean of the chosen video means; percentile interval by linear interpolation at
    position (B - 1) * q. Returns (point, (lo, hi)) or (None, None) without videos."""
    vids = sorted(vmeans)
    V = len(vids)
    if V == 0:
        return None, None
    vals = [vmeans[v] for v in vids]
    point = sum(vals) / V
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        stats.append(sum(vals[rng.randrange(V)] for _ in range(V)) / V)
    stats.sort()
    return point, (_percentile_linear(stats, alpha / 2), _percentile_linear(stats, 1 - alpha / 2))


def scorer_class_native(S):
    """Deployed-output class of the auxiliary scorer: S > 0 -> 1, S < 0 -> 0, S == 0 -> undecided (None)."""
    if S is None or not math.isfinite(S):
        return None
    return 1 if S > 0 else (0 if S < 0 else None)


def scorer_class_hi(S_hi):
    if S_hi is None or not math.isfinite(S_hi):
        return None
    return 1 if S_hi > 0 else (0 if S_hi < 0 else None)


# ---------------------------------------------------------------- per-input request plan

def static_plan():
    """The input-independent requests of Phases 0-1 (PROTOCOL §6): (phase, purpose, target, family, mask)."""
    out = [("baseline", "baseline", "label_only_score", "none", ()),
           ("baseline", "baseline", "original_generation", "none", ()),
           ("baseline", "nosound", "nosound_generation", "none", ())]
    for fam in ("blur", "grey"):
        out += [("fixed", "single", "label_only_score", fam, (s,)) for s in SLOTS]
    for fam in ("blur", "grey"):
        out += [("fixed", "pair", "label_only_score", fam, p) for p in all_pairs()]
    out += [("fixed", "pair", "original_generation", "grey", p) for p in all_pairs()]
    return out


def upper_bound_counts():
    """Per-input upper bound of unique requests and transformed inputs (no cache overlap)."""
    st = static_plan()
    scores = sum(1 for r in st if r[2] == "label_only_score") + 2 + 11 + 2
    gens = sum(1 for r in st if r[2] != "label_only_score") + 3 + 2
    transformed = 8 + 28 + 11 + 8 + 28 + 2 + 2
    return {"label_only_score": scores, "generation": gens, "transformed_inputs": transformed}

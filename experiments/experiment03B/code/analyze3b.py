#!/usr/bin/env python3
"""experiment03B analyzer: everything is recomputed from raw request records (RECORDS.md); stored derived files are
not trusted. Writes results/ (inputs_metrics.csv, pairs.csv, trajectories.csv, primary.json, secondary.json,
table_a1.json). Standard library + xb_core only. PROTOCOL.md §5-§8 define every quantity.

Native margin (protocol v1.1): S_native = logit_1 - logit_0, the exact BF16 logit difference. It equals the stored
S = log P('1') - log P('0') up to ~1e-7 log-softmax rounding, which would otherwise hide exact native ties.

    python analyze3b.py --run-dir experiments/experiment03B/runs/<run_id> [--seeds 0 1 2] [--partial]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import xb_core as xc  # noqa: E402
from io3b import read_jsonl, write_json  # noqa: E402

SUCCESS = ("fresh", "cached", "prior_cache")
N_REMAINING_AT = {3: 6, 4: 5}  # candidates at adaptive states 3 and 4 (8 slots minus the 2 or 3 already masked)
CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
              "Shooting", "Shoplifting", "Stealing", "Vandalism"]
RESULT_FIELDS = ("S", "S_hi", "logit_0", "logit_1", "hi_logit_0", "hi_logit_1", "label", "parse_status", "raw_response",
                 "slot_tensor_sha256")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


class Records:
    """Authoritative values per (window_uid, target, family, mask, repeat): the latest successful line (request files
    in name order, then line order)."""

    def __init__(self, rd):
        self.val, self.by_key, self.conflicts = {}, {}, []
        self.status_counts = {}
        self.question_sha = {}
        for p in sorted((rd / "requests").glob("*.jsonl")):
            for r in read_jsonl(p):
                self.status_counts[r["cache_status"]] = self.status_counts.get(r["cache_status"], 0) + 1
                if r["cache_status"] not in SUCCESS:
                    continue
                if r.get("question_sha256"):
                    self.question_sha.setdefault(r["target"], set()).add(r["question_sha256"])
                k = (r["window_uid"], r["target"], r["family"], tuple(r["mask"]), r["repeat"])
                self.val[k] = r
                prev = self.by_key.get(r["result_key"])
                if prev is not None and {f: prev.get(f) for f in RESULT_FIELDS} != {f: r.get(f) for f in RESULT_FIELDS}:
                    self.conflicts.append(r["result_key"])
                self.by_key[r["result_key"]] = r

    def score(self, uid, family, mask, field="S_hi"):
        r = self.val.get((uid, "label_only_score", family if mask else "none", tuple(mask), 0))
        if r is None:
            return None
        if field == "S_native":
            l0, l1 = _num(r.get("logit_0")), _num(r.get("logit_1"))
            return None if l0 is None or l1 is None else l1 - l0
        return _num(r.get(field))

    def rec(self, uid, target, family, mask, repeat=0):
        return self.val.get((uid, target, family if mask else "none", tuple(mask), repeat))

    def label(self, uid, family, mask, target="original_generation", repeats=2):
        for rep in range(0, repeats + 1):
            r = self.rec(uid, target, family, mask, rep)
            if r is not None and r.get("label") in (0, 1) and r.get("parse_status") == "ok":
                return r["label"]
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def vw_mean(values, video_of):
    """Video-weighted mean: within-video means first, then the mean over videos."""
    vm = xc.video_means(values, video_of)
    return (sum(vm.values()) / len(vm)) if vm else None, len(vm)


def analyse_input(inp, R, cfg):
    uid = inp["window_uid"]
    reps = cfg["retries"]["parse_failure_repeats"]
    row = {"window_uid": uid, "video_id": inp["video_id"], "category": inp["category"], "anchor": inp["anchor"],
           "strata": "+".join(inp.get("strata") or []), "development": bool(inp.get("development")),
           "status": inp.get("status"), "metadata_flags": "+".join(inp.get("metadata_flags") or []),
           "clipped": inp.get("clipped"), "annotated_slots_n": len(inp.get("annotated_slots") or [])}
    L0 = R.label(uid, "none", (), repeats=reps)
    ns = R.label(uid, "none", (), target="nosound_generation", repeats=reps)
    S0, Sh0 = R.score(uid, "none", (), "S_native"), R.score(uid, "none", (), "S_hi")
    base = R.rec(uid, "label_only_score", "none", ())
    row.update({"L0": L0, "nosound_label": ns, "nosound_changed": None if (L0 is None or ns is None) else int(ns != L0),
                "S_native0": S0, "S_hi0": Sh0, "candidate_mass0": None if base is None else base.get("candidate_mass"),
                "scorer_class_native0": xc.scorer_class_native(S0), "scorer_class_hi0": xc.scorer_class_hi(Sh0)})
    row["scorer_agrees_native0"] = None if L0 is None or row["scorer_class_native0"] is None else int(row["scorer_class_native0"] == L0)
    out = {"row": row, "pairs": [], "traj": []}
    if L0 is None:
        row["attribution"] = "unavailable_L0_missing"
        return out
    q = xc.q_of(L0)
    row["q"] = q
    rk = {}
    for name, fam, field in (("blur", "blur", "S_hi"), ("grey", "grey", "S_hi"), ("blur_native", "blur", "S_native")):
        rk[name] = xc.ranking(q, R.score(uid, "none", (), field), {s: R.score(uid, fam, (s,), field) for s in xc.SLOTS})
    blur_ok = rk["blur"]["status"] == "complete"
    row["blur_ranking"] = rk["blur"]["status"]
    if blur_ok:
        b = rk["blur"]
        row.update({"order_blur": " ".join(map(str, b["order"])), "tie_cut2": b["tie_spans_cut"]["2"],
                    "any_tie": bool(b["ties"]), "all_tied": b["all_tied"], "all_nonpositive": b["all_nonpositive"],
                    "top2_margin": b["top2_margin"]})
    T = sorted(rk["blur"]["order"][:2]) if blur_ok else None
    row["T_w"] = None if T is None else f"{T[0]} {T[1]}"
    eff1 = {fam: {s: xc.effect(q, Sh0, R.score(uid, fam, (s,))) for s in xc.SLOTS} for fam in ("blur", "grey")}
    pair_labels, grey_pair_eff, blur_pair_eff = {}, {}, {}
    n_agree = n_cmp = 0
    for fam in ("blur", "grey"):
        for p in xc.all_pairs():
            sh, s = R.score(uid, fam, p), R.score(uid, fam, p, "S_native")
            D = xc.effect(q, Sh0, sh)
            J = None if D is None or eff1[fam][p[0]] is None or eff1[fam][p[1]] is None else D - eff1[fam][p[0]] - eff1[fam][p[1]]
            pr = {"window_uid": uid, "video_id": inp["video_id"], "family": fam, "pair": f"{p[0]} {p[1]}",
                  "adjacent": int(p[1] - p[0] == 1), "S_hi": sh, "S_native": s, "D": D,
                  "D_native": xc.effect(q, S0, s), "J": J}
            if fam == "grey":
                lab = R.label(uid, "grey", p, repeats=reps)
                pair_labels[p] = lab
                pr.update({"label": lab, "flip": None if lab is None else int(lab != L0)})
                sc = xc.scorer_class_native(s)
                if lab is not None and sc is not None:
                    n_cmp += 1
                    n_agree += int(sc == lab)
                grey_pair_eff[p] = D
            else:
                blur_pair_eff[p] = D
            out["pairs"].append(pr)
    row["n_pair_labels"] = sum(v is not None for v in pair_labels.values())
    row["grey_pairs_changed"] = sum(1 for v in pair_labels.values() if v is not None and v != L0)
    row["scorer_grey_pairs_agree"], row["scorer_grey_pairs_compared"] = n_agree, n_cmp
    row["complete_28"] = int(row["n_pair_labels"] == 28)
    if L0 == 1:
        if T is not None:
            d, ft, fr, _ = xc.d_w(T, pair_labels)
            lo, hi = xc.d_w_bounds(T, pair_labels)
        else:  # undefined T_w: the selected flip is bounded by 0 and 1
            d, ft = None, None
            flips = [xc.flip_1_to_0(v) for v in pair_labels.values()]
            fr = sum(flips) / 28 if all(f is not None for f in flips) else None
            lo = 0 - sum(1 if f is None else f for f in flips) / 28
            hi = 1 - sum(0 if f is None else f for f in flips) / 28
        row.update({"d_w": d, "F_top2": ft, "F_random_mean": fr, "d_w_lo": lo, "d_w_hi": hi})
    if T is not None and all(v is not None for v in grey_pair_eff.values()):
        row["e_w"] = grey_pair_eff[tuple(T)] - sum(grey_pair_eff.values()) / 28
    if blur_ok and rk["grey"]["status"] == "complete":
        g = rk["grey"]
        row["overlap_blur_grey"] = len(set(T) & set(g["order"][:2]))
        row["tau_b_blur_grey"] = xc.kendall_tau_b([rk["blur"]["effects"][str(s)] for s in xc.SLOTS],
                                                  [g["effects"][str(s)] for s in xc.SLOTS])
        Tg = sorted(g["order"][:2])
        if all(v is not None for v in blur_pair_eff.values()):
            row["reciprocal_blur_effect_of_grey_top2"] = blur_pair_eff[tuple(Tg)] - sum(blur_pair_eff.values()) / 28
    if blur_ok and rk["blur_native"]["status"] == "complete":
        n_ = rk["blur_native"]
        row["overlap_native_hi"] = len(set(T) & set(n_["order"][:2]))
        row["tau_b_native_hi"] = xc.kendall_tau_b([rk["blur"]["effects"][str(s)] for s in xc.SLOTS],
                                                  [n_["effects"][str(s)] for s in xc.SLOTS])
        row["native_any_tie"] = bool(n_["ties"])

    def score_of(mask):
        return R.score(uid, "blur", mask) if mask else R.score(uid, "none", ())
    traj, complete = xc.adaptive_trajectory(q, score_of, cfg["ranking"]["adaptive_max_budget"])
    out["adaptive"] = {**traj, "complete": complete}
    row["adaptive_complete"] = complete
    row["order_adaptive"] = " ".join(map(str, traj["order"]))
    fixed_pref = xc.prefixes(rk["blur"]["order"]) if blur_ok else {}
    adapt_pref = xc.prefixes(traj["order"], budgets=range(1, len(traj["order"]) + 1)) if complete else {}
    static = {(fam, p) for fam in ("blur", "grey") for p in xc.all_pairs()} | {(fam, (s,)) for fam in ("blur", "grey") for s in xc.SLOTS}
    fixed_eval = {("grey", tuple(m)) for m in fixed_pref.values()}
    for mode, pref in (("fixed", fixed_pref), ("adaptive", adapt_pref)):
        for k in range(1, 5):
            m = tuple(pref.get(str(k), ()))
            if not m:
                out["traj"].append({"window_uid": uid, "video_id": inp["video_id"], "mode": mode, "k": k, "mask": None})
                continue
            lab = R.label(uid, "grey", m, repeats=reps)
            sh, s = R.score(uid, "grey", m), R.score(uid, "grey", m, "S_native")
            # scheduled model calls this mode needs beyond the shared Phase 0-1 plan; a property of the design
            if mode == "fixed":
                new = {1: 1, 2: 0}.get(k, 2)  # k=1: one generation (score shared); k=2: in the 28 pairs
            else:
                cand = N_REMAINING_AT[k] if k >= 3 else 0
                shared = k <= 2 or ("grey", m) in fixed_eval or ("grey", m) in static
                new = cand + (0 if shared else 2)
            out["traj"].append({"window_uid": uid, "video_id": inp["video_id"], "mode": mode, "k": k,
                                "mask": " ".join(map(str, m)), "label": lab,
                                "flip": None if lab is None else int(lab != L0),
                                "q_dS_hi": xc.effect(q, Sh0, sh), "q_dS_native": xc.effect(q, S0, s),
                                "attributable_new_calls": new})
    if inp.get("strata") and "normal" in inp["strata"] and L0 == 0:
        row["normal_0to1_grey_pairs"] = sum(1 for v in pair_labels.values() if v == 1)
    return out


def estimate(rows, key, seed, B, alpha, include, scale=100.0, unit="pp", missing_name="n_inputs_missing_dw"):
    """Video-cluster estimate of a per-input quantity. n_inputs = population size; n_videos = videos with at least
    one non-missing value (the bootstrap's V). Flip-rate differences are in percentage points (scale 100, unit pp);
    score margins in nats of S_hi (scale 1, unit nats)."""
    vals = {r["window_uid"]: r.get(key) for r in rows if include(r)}
    video_of = {r["window_uid"]: r["video_id"] for r in rows}
    vm = xc.video_means(vals, video_of)
    point, ci = xc.cluster_bootstrap(vm, seed, B, alpha)
    return {f"estimate_{unit}": None if point is None else scale * point,
            f"ci95_{unit}": None if ci is None else [scale * ci[0], scale * ci[1]],
            "n_videos": len(vm), "n_videos_population": len({video_of[u] for u in vals}),
            "n_inputs": len(vals), missing_name: sum(v is None for v in vals.values()),
            f"video_means_{unit}": {k: scale * v for k, v in sorted(vm.items())},
            "values": {k: v for k, v in sorted(vals.items())}}


def primary_pop(r, dev=False):
    return r.get("L0") == 1 and (dev or not r["development"]) and r["status"] == "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--partial", action="store_true", help="the runner found planned work without terminal records")
    a = ap.parse_args()
    rd = Path(a.run_dir)
    cfg = json.loads((rd / "frozen" / "config.json").read_text())
    inputs = json.loads((rd / "inputs.json").read_text())["inputs"]
    R = Records(rd)
    res = rd / "results"
    res.mkdir(exist_ok=True)
    outs = [analyse_input(i, R, cfg) for i in inputs]
    rows = [o["row"] for o in outs]
    B, alpha = cfg["analysis"]["bootstrap_B"], cfg["analysis"]["alpha"]
    offs = cfg["analysis"]["stability_seed_offsets"]
    code = {f: sha_file(CODE / f) for f in ("analyze3b.py", "xb_core.py", "io3b.py")}
    run_flags = {"partial": a.partial, "skipped_budget_lines": R.status_counts.get("skipped_budget", 0),
                 "error_lines": R.status_counts.get("error", 0), "record_conflicts": len(R.conflicts)}

    def primary_for(seed):
        p = estimate(rows, "d_w", seed, B, alpha, primary_pop)
        p["d_w"] = p.pop("values")
        p["bootstrap"] = {"seed": seed, "B": B, "method": "video-cluster percentile; random.Random(seed).randrange",
                          "stability": [{"seed": seed + o, "ci95_pp": estimate(rows, "d_w", seed + o, B, alpha, primary_pop)["ci95_pp"]}
                                        for o in offs]}
        lo = estimate(rows, "d_w_lo", seed, 1000, alpha, primary_pop)
        hi = estimate(rows, "d_w_hi", seed, 1000, alpha, primary_pop)
        p["bounds_pp"] = [lo["estimate_pp"], hi["estimate_pp"]]
        p["bounds_note"] = ("worst-case bounds on the estimate had every missing label been observed (missing selected "
                            "flips 0/1, missing comparator flips 1/0, then video averaging); the complete-case estimate "
                            "need not lie inside them")
        sens = {"tie_free": lambda r: primary_pop(r) and r.get("tie_cut2") is False,
                "scorer_agreement": lambda r: primary_pop(r) and r.get("scorer_agrees_native0") == 1,
                "with_development": lambda r: primary_pop(r, dev=True),
                "metadata_clean": lambda r: primary_pop(r) and not r.get("metadata_flags")}
        p["sensitivity"] = {}
        for name, f in sens.items():
            e = estimate(rows, "d_w", seed, B, alpha, f)
            e.pop("values")
            p["sensitivity"][name] = e
        pp = [r for r in rows if primary_pop(r)]
        p["counts"] = {"inputs": len(rows), "L0_parsed": sum(r.get("L0") is not None for r in rows),
                       "alerts_confirmation": len(pp), "all_tied": sum(bool(r.get("all_tied")) for r in pp),
                       "all_nonpositive": sum(bool(r.get("all_nonpositive")) for r in pp),
                       "tie_cut2": sum(bool(r.get("tie_cut2")) for r in pp),
                       "complete_28": sum(r.get("complete_28") == 1 for r in pp)}
        p["population"] = "unique inputs, not development, status ok, fresh original L0 = 1 (PROTOCOL §3.3)"
        p["run_flags"] = run_flags
        p["code_sha256"] = code
        return p

    primary = primary_for(a.seeds[0])
    write_json(res / "primary.json", primary)
    for s in a.seeds[1:]:
        write_json(res / "seeds" / f"primary_seed{s}.json", primary_for(s))

    video_of = {r["window_uid"]: r["video_id"] for r in rows}
    scope = cfg["analysis"].get("secondary_scope", "confirmation")
    label_avail = lambda r: (r.get("L0") is not None and r["status"] == "ok"  # noqa: E731
                             and (scope == "all" or not r["development"]))
    ok = [r for r in rows if label_avail(r)]
    avail = {r["window_uid"] for r in ok}
    sec = {"seed": a.seeds[0], "scope": scope, "n_inputs": len(ok), "n_videos": len({r["video_id"] for r in ok}),
           "scope_note": "confirmation = non-development inputs with a parsed L0 (protocol default); all = development "
                         "included (integration check only)", "run_flags": run_flags}
    e = estimate(rows, "e_w", a.seeds[0], B, alpha, label_avail, scale=1.0, unit="nats", missing_name="n_inputs_missing")
    e.pop("values")
    sec["1_grey_numeric_margin"] = e
    traj = [t for o in outs for t in o["traj"] if t["window_uid"] in avail]
    tr = {}
    for mode in ("fixed", "adaptive"):
        for k in range(1, 5):
            sub = [t for t in traj if t["mode"] == mode and t["k"] == k and t.get("mask") is not None]
            ch, n1 = vw_mean({t["window_uid"]: t.get("flip") for t in sub}, video_of)
            dq, _ = vw_mean({t["window_uid"]: t.get("q_dS_hi") for t in sub}, video_of)
            tr[f"{mode}_k{k}"] = {"label_change_rate_video_weighted": ch, "mean_q_dS_hi_video_weighted": dq,
                                  "n_inputs": len(sub), "n_videos": n1,
                                  "attributable_new_calls": sum(t.get("attributable_new_calls") or 0 for t in sub)}
    sec["2_fixed_vs_adaptive_grey_prefixes"] = tr
    diffs = {}
    for k in range(1, 5):
        for key in ("flip", "q_dS_hi"):
            d = {}
            for uid in sorted(avail):
                f = next((t for t in traj if t["window_uid"] == uid and t["mode"] == "fixed" and t["k"] == k), None)
                g = next((t for t in traj if t["window_uid"] == uid and t["mode"] == "adaptive" and t["k"] == k), None)
                if f and g and f.get(key) is not None and g.get(key) is not None:
                    d[uid] = g[key] - f[key]
            m, nv = vw_mean(d, video_of)
            diffs[f"k{k}_{key}"] = {"adaptive_minus_fixed_video_weighted": m, "n_inputs": len(d), "n_videos": nv}
    sec["2_adaptive_minus_fixed_within_video"] = diffs
    agree = sum(r.get("scorer_grey_pairs_agree") or 0 for r in ok)
    cmpn = sum(r.get("scorer_grey_pairs_compared") or 0 for r in ok)
    gvw, gnv = vw_mean({r["window_uid"]: r["scorer_grey_pairs_agree"] / r["scorer_grey_pairs_compared"]
                        for r in ok if r.get("scorer_grey_pairs_compared")}, video_of)
    sec["3_agreement"] = {
        "top2_overlap_blur_grey_mean_inputs": mean([r.get("overlap_blur_grey") for r in ok]),
        "tau_b_blur_grey_mean_inputs": mean([r.get("tau_b_blur_grey") for r in ok]),
        "top2_overlap_native_hi_mean_inputs": mean([r.get("overlap_native_hi") for r in ok]),
        "tau_b_native_hi_mean_inputs": mean([r.get("tau_b_native_hi") for r in ok]),
        "baseline_scorer_agrees_with_L0": f"{sum(r.get('scorer_agrees_native0') == 1 for r in ok)}/"
                                          f"{sum(r.get('scorer_agrees_native0') is not None for r in ok)}",
        "grey_pairs_scorer_agrees_with_label_pooled": f"{agree}/{cmpn}",
        "grey_pairs_scorer_agreement_video_weighted": gvw, "grey_pairs_agreement_n_videos": gnv,
        "blur_rankings": {"any_tie": sum(bool(r.get("any_tie")) for r in ok), "tie_cut2": sum(bool(r.get("tie_cut2")) for r in ok),
                          "all_tied": sum(bool(r.get("all_tied")) for r in ok),
                          "all_nonpositive": sum(bool(r.get("all_nonpositive")) for r in ok)},
        "native_ranking_any_tie": sum(bool(r.get("native_any_tie")) for r in ok),
        "candidate_mass0_mean": mean([r.get("candidate_mass0") for r in ok]),
        "note": "input-level means (unweighted) over label-available inputs; Table A1's overlap column is video-weighted "
                "over alert inputs"}
    pairs = [p for o in outs for p in o["pairs"]]
    sp = [p for p in pairs if p["window_uid"] in avail]
    sec["4_pair_nonadditivity"] = {fam: {"mean_J": mean([p["J"] for p in sp if p["family"] == fam]),
                                         "mean_abs_J": mean([abs(p["J"]) for p in sp if p["family"] == fam and p["J"] is not None]),
                                         "mean_J_adjacent": mean([p["J"] for p in sp if p["family"] == fam and p["adjacent"]]),
                                         "mean_J_nonadjacent": mean([p["J"] for p in sp if p["family"] == fam and not p["adjacent"]]),
                                         "n_pairs": sum(1 for p in sp if p["family"] == fam and p["J"] is not None),
                                         "n_inputs": len(avail),
                                         "note": "exploratory; not adjusted for singleton effects or position"}
                                   for fam in ("blur", "grey")}
    freq = [0] * 8
    for r in rows:
        if primary_pop(r) and r.get("T_w"):
            for s in map(int, r["T_w"].split()):
                freq[s] += 1
    sec["4_top2_slot_frequency_primary"] = freq
    nrm = [r for r in ok if "normal" in r["strata"].split("+") and r.get("L0") == 0]
    obs = sum(r["n_pair_labels"] for r in nrm)
    flips = sum(r.get("normal_0to1_grey_pairs", 0) for r in nrm)
    sec["5_normal_video_0to1_grey_pair_flips"] = {
        "inputs": len(nrm), "flips": flips, "pairs_scheduled": 28 * len(nrm), "labels_observed": obs,
        "labels_missing": 28 * len(nrm) - obs, "flip_rate_observed": flips / obs if obs else None,
        "flip_rate_upper_bound": (flips + 28 * len(nrm) - obs) / (28 * len(nrm)) if nrm else None}
    sec["6_by_stratum_and_category"] = {}
    for name, pred in [(s, lambda r, s=s: s in r["strata"].split("+")) for s in ("event", "negative", "normal")] + \
                      [(c, lambda r, c=c: r["category"] == c) for c in CATEGORIES]:
        sub = [r for r in ok if pred(r)]
        sec["6_by_stratum_and_category"][name] = {"inputs": len(sub), "videos": len({r["video_id"] for r in sub}),
                                                  "alerts": sum(r.get("L0") == 1 for r in sub),
                                                  "nosound_changed": sum(r.get("nosound_changed") == 1 for r in sub)}
    sec["7_nosound_label_change"] = f"{sum(r.get('nosound_changed') == 1 for r in ok)}/{sum(r.get('nosound_changed') is not None for r in ok)}"
    sec["8_annotation_coverage_event_inputs"] = {str(k): sum(1 for r in rows if "event" in r["strata"].split("+")
                                                         and r["window_uid"] in avail and r["annotated_slots_n"] == k)
                                                 for k in range(9)}
    e = estimate(rows, "reciprocal_blur_effect_of_grey_top2", a.seeds[0], B, alpha, label_avail, scale=1.0, unit="nats",
                 missing_name="n_inputs_missing")
    e.pop("values")
    sec["9_reciprocal_numeric_check"] = e
    sec["baseline_alert_rate"] = {"all_selected": f"{sum(r.get('L0') == 1 for r in rows)}/{len(rows)}",
                                  "L0_missing": sum(r.get("L0") is None for r in rows)}
    sec["record_conflicts"] = R.conflicts
    write_json(res / "secondary.json", sec)

    def row_for(name, pred, primary_row=False):
        pop = [r for r in rows if pred(r)]
        alerts = [r for r in pop if r.get("L0") == 1 and r["status"] == "ok"]
        comp = [r for r in alerts if r.get("d_w") is not None]
        out = {"population": name, "n_videos": len({r["video_id"] for r in pop}), "n_windows": len(pop),
               "n_alerts": len(alerts), "n_complete": sum(r.get("complete_28") == 1 for r in alerts),
               "n_with_d_w": len(comp)}
        if not comp:
            out.update({k: "NA" for k in ("top2_flip_pct", "random_pair_flip_pct", "paired_advantage_pp", "ci95_pp")})
            out["na_reason"] = ("no selected window" if not pop else
                                f"no original alert among {len(pop)} windows" if not alerts else
                                f"{out['n_complete']} of {len(alerts)} alert windows have all 28 grey-pair labels; "
                                f"none has a defined d_w")
        else:
            m = {r["window_uid"]: r for r in comp}
            ft, _ = vw_mean({u: r["F_top2"] for u, r in m.items()}, video_of)
            fr, _ = vw_mean({u: r["F_random_mean"] for u, r in m.items()}, video_of)
            dw, _ = vw_mean({u: r["d_w"] for u, r in m.items()}, video_of)
            out.update({"top2_flip_pct": 100 * ft, "random_pair_flip_pct": 100 * fr, "paired_advantage_pp": 100 * dw,
                        "ci95_pp": primary["ci95_pp"] if primary_row else "descriptive"})
        ov, _ = vw_mean({r["window_uid"]: r.get("overlap_blur_grey") for r in alerts}, video_of)
        out["mean_top2_overlap_blur_grey"] = ov if ov is not None else "NA"
        return out
    nd = lambda r: not r["development"]  # noqa: E731
    st = lambda s: (lambda r: nd(r) and s in r["strata"].split("+"))  # noqa: E731
    table = [row_for("All original alerts, confirmation set", nd, primary_row=True),
             row_for("Event-centered windows", st("event")),
             row_for("Annotation-free anomaly-video windows", st("negative")),
             row_for("Normal-video windows", st("normal"))]
    for c in CATEGORIES:
        table.append(row_for(c, lambda r, c=c: nd(r) and r["category"] == c))
    table.append(row_for("Sensitivity: including development videos", lambda r: True))
    qsha = hashlib.sha256("".join(q + "\n" for q in cfg["prompt"]["guiding_questions"]).encode()).hexdigest()
    by_target = {t: sorted(x[:12] for x in v) for t, v in sorted(R.question_sha.items())}
    note = {"model_revision": cfg["model"]["revision"], "source_head": cfg["source"]["expected_head"],
            "prompts": f"VERA learner instruct {cfg['prompt']['learner_instruct_file']} at {cfg['source']['expected_head'][:8]}; "
                       f"released guiding questions sha256 {qsha[:12]}; no-sound = question 5 removed; question sha256 "
                       f"by target {by_target}",
            "scorer": f"{cfg['scoring']['contract']}; rankings on {cfg['scoring']['ranking_readout']}; native margin = logit_1 - logit_0",
            "replacements": f"construction blur (PIL GaussianBlur radius "
                            f"{cfg['replacements']['blur_radius_fraction'] * cfg['replacements']['preprocessed_resolution']:.2f} "
                            f"at {cfg['replacements']['preprocessed_resolution']}x{cfg['replacements']['preprocessed_resolution']}); "
                            f"evaluation grey RGB{tuple(cfg['replacements']['grey_rgb'])}",
            "excluded_development_videos": list(xc.DEVELOPMENT_VIDEOS),
            "complete_case": f"{primary['n_inputs'] - primary['n_inputs_missing_dw']} of {primary['n_inputs']} confirmation "
                             f"alert inputs have d_w ({primary['n_videos']} videos); {primary['n_inputs_missing_dw']} missing",
            "missing_bounds_pp": primary["bounds_pp"],
            "bootstrap": f"seed {primary['bootstrap']['seed']}, {primary['bootstrap']['B']} video-cluster resamples",
            "weighting": "flip percentages and advantages are video-weighted (within-video mean, then mean over videos)",
            "engine": cfg["engine"], "run_id": cfg["run_id"], "run_flags": run_flags,
            "analyzer_sha256": code["analyze3b.py"][:12]}
    write_json(res / "table_a1.json", {"rows": table, "note": note})

    def dump(name, recs):
        cols = []
        for r in recs:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(res / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in recs:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    dump("inputs_metrics.csv", rows)
    dump("pairs.csv", pairs)
    dump("trajectories.csv", [t for o in outs for t in o["traj"]])
    print(json.dumps({"inputs": len(rows), "primary_estimate_pp": primary["estimate_pp"], "ci95_pp": primary["ci95_pp"],
                      "n_videos": primary["n_videos"], "conflicts": len(R.conflicts), "partial": a.partial}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

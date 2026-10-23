#!/usr/bin/env python3
"""experiment02 analyzer (CPU, no model calls). Reads the logged requests of a run and writes
temporal_effects.csv, frame_attribution.csv, evaluation.csv and summary.json into <run-dir>/results.

Targets stay separate: `label` columns are original-prompt (VERA) labels; `S`/`Delta` columns are the auxiliary
label-only scorer. Delta(G) = S(baseline) - S(G perturbed): positive = the replaced frames supported anomaly
preference under the scorer; negative = they supported normality. Interactions are non-additivity on that scale.
S is a difference of two bf16 logits: effects with |value| <= S_resolution are flagged below_resolution.
Usage: python experiments/experiment02/code/analyze_attr.py --run-dir experiments/experiment02/runs/run_001
"""
import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import yaml  # noqa: E402

import attr_lib as al  # noqa: E402

OK = ("fresh", "cached")


def fmt(v, nd=6):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mstr(m):
    return "-".join(map(str, m))


def scorer_class(S):
    return None if S is None else int(S >= 0)


def tau_b(x, y):
    """Kendall tau-b between two value vectors over the same slots (ties handled); returns (tau, tied_pairs)."""
    conc = disc = tx = ty = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            tx += 1
            ty += 1
        elif dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx * dy > 0:
            conc += 1
        else:
            disc += 1
    n0 = len(x) * (len(x) - 1) / 2
    den = ((n0 - tx) * (n0 - ty)) ** 0.5
    return (None if den == 0 else (conc - disc) / den), {"tied_in_first": tx, "tied_in_second": ty}


def compare(ra, rb, budgets):
    """Agreement of two frame-score dicts (already sign-adjusted to the ranking direction)."""
    if ra["status"] != "complete" or rb["status"] != "complete":
        return {"status": "not_rankable", "first": ra["status"], "second": rb["status"]}
    x = [ra["signed"][s] for s in range(al.N_SLOTS)]
    y = [rb["signed"][s] for s in range(al.N_SLOTS)]
    t, ties = tau_b(x, y)
    out = {"status": "compared", "kendall_tau_b": t, **ties}
    for b in budgets:
        if ra.get(f"tie_spans_cut_at_{b}") or rb.get(f"tie_spans_cut_at_{b}"):
            out[f"top{b}_overlap"] = None
            out[f"top{b}_note"] = "tie spans the cut"
        else:
            out[f"top{b}_overlap"] = len(set(ra["order"][:b]) & set(rb["order"][:b]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rd = al.ROOT / a.run_dir
    res = rd / "results"
    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in json.loads((rd / "cases.json").read_text(encoding="utf-8"))["cases"]}
    fams = cfg["stage_b"]["families"]
    sizes = tuple(cfg["stage_c"]["group_sizes"])
    budgets = cfg["stage_c"]["budgets"]
    mass_flag = cfg["stage_a"]["perturbed_candidate_mass_flag"]

    def latest_ok(path):
        d = {}
        for r in al.read_jsonl(path):
            if r["cache_status"] in OK:
                d[r["request_id"]] = r
        return d
    base = latest_ok(res / "baseline_responses.jsonl")
    abl = latest_ok(res / "prompt_ablation.jsonl")
    pert = list(latest_ok(res / "perturbation_responses.jsonl").values())
    all_logged = al.read_jsonl(res / "baseline_responses.jsonl") + al.read_jsonl(res / "prompt_ablation.jsonl") \
        + al.read_jsonl(res / "perturbation_responses.jsonl")
    cache = al.read_jsonl(res / "cache_records.jsonl")
    diag = json.loads((res / "scorer_diagnostics.json").read_text(encoding="utf-8"))

    def b0(cid, target, repeat=0):
        stage = "A1" if target == "original_generation" else "A2"
        return base.get(f"{stage}|{target}|{cid}|none||r{repeat}", {})

    idx = {}
    for r in pert:
        if r["stage"] == "B":
            idx[(r["case_id"], r["family"], tuple(r["mask"]), r["target"])] = r

    def get(cid, fam, m, target):
        return idx.get((cid, fam, tuple(m), target), {})

    # ------------------------------------------------ temporal_effects.csv (Stage B)
    masks = al.primary_masks() + al.control_masks(cfg)
    te_rows = []
    for cid in cfg["primary_cases"]:
        c = cases[cid]
        g0, s0 = b0(cid, "original_generation"), b0(cid, "label_only_score")
        L0, S0 = g0.get("label"), s0.get("S")
        cdir = None if S0 is None else (1 if S0 >= 0 else -1)
        base_claims = al.claim_flags(g0.get("raw_response"))
        for fam in fams:
            single_bin = {s: al.signed_binary_diff(L0, get(cid, fam, (s,), "original_generation").get("label"))
                          for s in range(al.N_SLOTS)}
            single_S = {}
            for s in range(al.N_SLOTS):
                Sp = get(cid, fam, (s,), "label_only_score").get("S")
                single_S[s] = None if S0 is None or Sp is None else S0 - Sp
            for mtype, m in masks:
                g = get(cid, fam, m, "original_generation")
                sc = get(cid, fam, m, "label_only_score")
                Lp, Sp = g.get("label"), sc.get("S")
                bd = al.signed_binary_diff(L0, Lp)
                D = None if S0 is None or Sp is None else S0 - Sp
                res_ = max(s0.get("S_resolution") or 0.0, sc.get("S_resolution") or 0.0) if sc else None
                inter = al.interaction(D, [single_S[s] for s in m]) if len(m) > 1 else None
                pc = al.claim_flags(g.get("raw_response")) if g else {}
                te_rows.append({
                    "case_id": cid, "family": fam, "mask_type": mtype, "mask_slots": mstr(m), "size": len(m),
                    "frame_ids": ";".join(str(c["frame_indices"][s]) for s in m),
                    "frame_times_s": ";".join(f"{c['frame_times_s'][s]:.2f}" for s in m),
                    "orig_label": L0, "pert_label": Lp, "pert_parse_status": g.get("parse_status"),
                    "binary_diff_orig_minus_pert": bd, "flipped": None if bd is None else int(bd != 0),
                    "binary_interaction": al.interaction(bd, [single_bin[s] for s in m]) if len(m) > 1 else None,
                    "S_baseline": S0, "S_perturbed": Sp, "Delta": D,
                    "Delta_per_frame": None if D is None else D / len(m),
                    "S_resolution": res_,
                    "Delta_below_resolution": None if D is None or res_ is None else abs(D) <= res_,
                    "sum_single_Delta": None if len(m) == 1 or any(single_S[s] is None for s in m) else sum(single_S[s] for s in m),
                    "interaction_Delta": inter,
                    "interaction_below_resolution": None if inter is None or res_ is None else abs(inter) <= res_,
                    "support_for_scorer_baseline_class": None if D is None else cdir * D,
                    "scorer_class_flipped": None if Sp is None or S0 is None else int(scorer_class(Sp) != scorer_class(S0)),
                    "perturbed_candidate_mass": sc.get("candidate_mass"),
                    "low_candidate_mass": None if sc.get("candidate_mass") is None else sc["candidate_mass"] < mass_flag,
                    "slot_identity_ok": (g.get("slot_identity_ok") is not False) and (sc.get("slot_identity_ok") is not False),
                    "claims_lost": ";".join(k for k, v in base_claims.items() if v and not pc.get(k)),
                    "claims_gained": ";".join(k for k, v in pc.items() if v and not base_claims.get(k)),
                    "generation_cache_status": g.get("cache_status"), "score_cache_status": sc.get("cache_status"),
                    "missing": ";".join(n for n, x in (("generation", g), ("score", sc)) if not x),
                })
    write_csv(res / "temporal_effects.csv", te_rows)

    # ------------------------------------------------ frame_attribution.csv + ranking comparisons
    fa_rows, rank_cmp = [], {}
    cov = al.coverage(sizes)
    for cid in cfg["primary_cases"]:
        c = cases[cid]
        L0 = b0(cid, "original_generation").get("label")
        S0 = b0(cid, "label_only_score").get("S")
        ndir = 1 if scorer_class(S0) in (1, None) else -1
        bdir = 1 if L0 in (1, None) else -1  # binary: support for the original-prompt baseline label
        ranks = {}
        for fam in fams:
            rows = [r for r in te_rows if r["case_id"] == cid and r["family"] == fam
                    and r["mask_type"] in ("single", "adjacent_pair", "adjacent_triple")]
            eff = {tuple(map(int, r["mask_slots"].split("-"))): r["Delta"] for r in rows if r["Delta"] is not None}
            binary = {tuple(map(int, r["mask_slots"].split("-"))): r["binary_diff_orig_minus_pert"] for r in rows
                      if r["binary_diff_orig_minus_pert"] is not None}
            vals = {"singleton": (al.singleton_scores(eff), ndir),
                    "group_pooled": (al.group_scores_pooled(eff, sizes), ndir),
                    "group_size_balanced": (al.group_scores_size_balanced(eff, sizes), ndir),
                    "binary_singleton": (al.singleton_scores(binary), bdir),
                    "binary_group_size_balanced": (al.group_scores_size_balanced(binary, sizes), bdir)}
            for name, (v, d) in vals.items():
                order, ties = al.rank_slots(v, d)
                st = al.ranking_status(v, budgets, d)
                ranks[f"{name}|{fam}"] = {"order": order, "ties": ties, "direction": d, **st,
                                          "values": v, "signed": {s: (None if x is None else d * x) for s, x in v.items()}}
            for s in range(al.N_SLOTS):
                rs, rg = ranks[f"singleton|{fam}"], ranks[f"group_size_balanced|{fam}"]
                fa_rows.append({"case_id": cid, "family": fam, "slot": s, "frame_id": c["frame_indices"][s],
                                "time_s": round(c["frame_times_s"][s], 3), "coverage_pairs": cov[s][sizes[0]],
                                "coverage_triples": cov[s][sizes[1]], "singleton_Delta": rs["values"][s],
                                "group_pooled": ranks[f"group_pooled|{fam}"]["values"][s],
                                "group_size_balanced": rg["values"][s],
                                "binary_singleton_diff": ranks[f"binary_singleton|{fam}"]["values"][s],
                                "binary_group_size_balanced": ranks[f"binary_group_size_balanced|{fam}"]["values"][s],
                                "numeric_direction": ndir, "binary_direction": bdir,
                                "rank_singleton": rs["order"].index(s) + 1 if s in rs["order"] else None,
                                "rank_group_size_balanced": rg["order"].index(s) + 1 if s in rg["order"] else None,
                                "singleton_ranking_status": rs["status"], "group_ranking_status": rg["status"]})
        cmp = {}
        for name in ("singleton", "group_size_balanced", "group_pooled", "binary_singleton"):
            cmp[f"{name}: {fams[0]} vs {fams[1]}"] = compare(ranks[f"{name}|{fams[0]}"], ranks[f"{name}|{fams[1]}"], budgets)
        for fam in fams:
            cmp[f"singleton vs group_size_balanced ({fam})"] = compare(ranks[f"singleton|{fam}"], ranks[f"group_size_balanced|{fam}"], budgets)
        rank_cmp[cid] = {"rankings": {k: {kk: vv for kk, vv in v.items() if kk not in ("values", "signed")} for k, v in ranks.items()},
                         "comparisons": cmp, "numeric_direction": ndir, "binary_direction": bdir}
    write_csv(res / "frame_attribution.csv", fa_rows)

    # ------------------------------------------------ evaluation.csv (Stage C + matched-cardinality controls)
    ev_rows = []
    cidx = {(r["case_id"], r["family"], r.get("request_suffix"), r["target"]): r for r in pert if r["stage"] == "C"}
    for (cid, fam, suffix, target), r in sorted(cidx.items(), key=lambda kv: str(kv[0])):
        if target != "original_generation":
            continue
        sc = cidx.get((cid, fam, suffix, "label_only_score"), {})
        L0 = b0(cid, "original_generation").get("label")
        s0 = b0(cid, "label_only_score")
        S0, Sp = s0.get("S"), sc.get("S")
        D = None if S0 is None or Sp is None else S0 - Sp
        cdir = None if S0 is None else (1 if S0 >= 0 else -1)
        res_ = max(s0.get("S_resolution") or 0.0, sc.get("S_resolution") or 0.0) if sc else None
        ev_rows.append({"section": "ranking_eval", "case_id": cid, "ranking": r["ranking"],
                        "ranking_source_family": r.get("ranking_source_family"), "eval_family": fam,
                        "budget": r["budget"], "mode": r["mode"], "replaced_slots": mstr(r["mask"]), "n_replaced": len(r["mask"]),
                        "ranking_order": mstr(r["ranking_order"]), "ranking_ties": json.dumps(r["ranking_ties"]),
                        "ranking_tie_spans_cut": r.get("ranking_tie_spans_cut"),
                        "orig_label": L0, "pert_label": r.get("label"), "pert_parse_status": r.get("parse_status"),
                        "original_prompt_class_preserved": None if r.get("label") is None or L0 is None else int(r["label"] == L0),
                        "S_baseline": S0, "S_perturbed": Sp, "Delta": D, "S_resolution": res_,
                        "support_change_for_scorer_class": None if D is None else cdir * D,
                        "scorer_class_preserved": None if Sp is None or S0 is None else int(scorer_class(Sp) == scorer_class(S0)),
                        "perturbed_candidate_mass": sc.get("candidate_mass"),
                        "generation_cache_status": r.get("cache_status"), "score_cache_status": sc.get("cache_status")})
    ctrl = al.control_masks(cfg)
    ctrl_cov = {k: sorted({s for t, m in ctrl if t == k for s in m}) for k in ("nonconsecutive_pair", "nonconsecutive_triple")}
    for cid in cfg["primary_cases"]:
        for fam in fams:
            for mtype in ("adjacent_pair", "nonconsecutive_pair", "adjacent_triple", "nonconsecutive_triple"):
                rows = [t for t in te_rows if t["case_id"] == cid and t["family"] == fam and t["mask_type"] == mtype]
                Ds = [t["Delta"] for t in rows if t["Delta"] is not None]
                Is = [t["interaction_Delta"] for t in rows if t["interaction_Delta"] is not None]
                Bi = [t["binary_interaction"] for t in rows if t["binary_interaction"] is not None]
                fl = [t["flipped"] for t in rows if t["flipped"] is not None]
                ev_rows.append({"section": "matched_cardinality", "case_id": cid, "ranking": mtype, "eval_family": fam,
                                "n_replaced": rows[0]["size"] if rows else None, "n_masks": len(rows),
                                "masks": ";".join(t["mask_slots"] for t in rows),
                                "slots_covered": ";".join(map(str, sorted({int(x) for t in rows for x in t["mask_slots"].split("-")}))),
                                "mean_Delta": statistics.mean(Ds) if Ds else None,
                                "mean_abs_Delta": statistics.mean(abs(x) for x in Ds) if Ds else None,
                                "max_abs_Delta": max((abs(x) for x in Ds), default=None),
                                "mean_interaction": statistics.mean(Is) if Is else None,
                                "mean_abs_interaction": statistics.mean(abs(x) for x in Is) if Is else None,
                                "n_interaction_above_resolution": sum(1 for t in rows if t["interaction_below_resolution"] is False),
                                "n_binary_interaction_nonzero": sum(1 for x in Bi if x != 0),
                                "n_original_label_flips": sum(fl) if fl else 0})
    write_csv(res / "evaluation.csv", ev_rows)

    # ------------------------------------------------ summary.json
    summ = {"scorer_status": diag["status"], "scorer_technical_gate": diag.get("technical_gate"), "scorer_checks": diag["checks"],
            "control_slot_coverage": ctrl_cov, "cases": {}, "rank_comparisons": rank_cmp,
            "stage_a": json.loads((res / "stage_a_summary.json").read_text(encoding="utf-8")) if (res / "stage_a_summary.json").exists() else None}
    for cid in cfg["primary_cases"]:
        per = {"baseline": {"original_label": b0(cid, "original_generation").get("label"),
                            "S": b0(cid, "label_only_score").get("S"),
                            "S_resolution": b0(cid, "label_only_score").get("S_resolution"),
                            "label_only_generation": b0(cid, "label_only_generation").get("label")}}
        for fam in fams:
            rows = [t for t in te_rows if t["case_id"] == cid and t["family"] == fam]
            by = {}
            for mtype in ("single", "adjacent_pair", "adjacent_triple", "all_eight", "nonconsecutive_pair", "nonconsecutive_triple"):
                rr = [t for t in rows if t["mask_type"] == mtype]
                Ds = [t["Delta"] for t in rr if t["Delta"] is not None]
                Is = [t["interaction_Delta"] for t in rr if t["interaction_Delta"] is not None]
                by[mtype] = {"n": len(rr), "label_flips": sum(t["flipped"] or 0 for t in rr),
                             "missing_labels": sum(t["pert_label"] is None for t in rr),
                             "flipped_masks": [t["mask_slots"] for t in rr if t["flipped"]],
                             "Delta_min": min(Ds) if Ds else None, "Delta_max": max(Ds) if Ds else None,
                             "mean_abs_Delta": statistics.mean(abs(x) for x in Ds) if Ds else None,
                             "n_Delta_above_resolution": sum(1 for t in rr if t["Delta_below_resolution"] is False),
                             "mean_interaction": statistics.mean(Is) if Is else None,
                             "mean_abs_interaction": statistics.mean(abs(x) for x in Is) if Is else None,
                             "scorer_class_flips": sum(t["scorer_class_flipped"] or 0 for t in rr),
                             "flipped_scorer_masks": [t["mask_slots"] for t in rr if t["scorer_class_flipped"]],
                             "low_candidate_mass": sum(1 for t in rr if t["low_candidate_mass"])}
            per[fam] = by
        per["stage_c"] = [e for e in ev_rows if e["section"] == "ranking_eval" and e["case_id"] == cid]
        summ["cases"][cid] = per
    gen_cache = [r for r in cache if "raw_response" in r]
    sc_cache = [r for r in cache if "S" in r]
    counts = {"unique_transformed_sequences": len({r["sequence_key"] for r in cache if r.get("transformed")}),
              "fresh_generation_calls": len(gen_cache), "fresh_score_calls": len(sc_cache),
              "cached_reuses": sum(1 for r in all_logged if r.get("cache_status") == "cached"),
              "errors_logged": sum(1 for r in all_logged if r.get("cache_status") == "error"),
              "skipped_budget": sum(1 for r in all_logged if r.get("cache_status") == "skipped_budget"),
              "slot_identity_failures": sum(1 for r in cache if r.get("slot_identity_ok") is False),
              "generation_seconds": round(sum(r.get("runtime_s", 0) for r in gen_cache), 1),
              "score_seconds": round(sum(r.get("runtime_s", 0) for r in sc_cache), 1),
              "peak_gpu_mem_allocated_gib_per_call_max": round(max((r.get("peak_gpu_mem_allocated_bytes", 0) for r in cache), default=0) / 2**30, 2)}
    man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
    counts["engine_counts_by_invocation"] = [s.get("engine_counts_cumulative_this_invocation") for st in man["stages"].values()
                                             for s in st if s.get("engine_counts_cumulative_this_invocation")]
    summ["counts"] = counts
    al.write_json(res / "summary.json", summ)
    print(json.dumps({"counts": {k: v for k, v in counts.items() if k != "engine_counts_by_invocation"},
                      "scorer_status": diag["status"]}, indent=1))


def write_csv(p, rows):
    if not rows:
        Path(p).write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


if __name__ == "__main__":
    main()

"""Erratum E1 check: S_hi changes below one bf16 step can still move the deployed bf16 score.

Reads an existing experiment02 effect table (results/temporal_effects.csv) and counts, per case and family, the rows
where the fp32 readout S_hi changed by less than one bf16 step while the native bf16 score S moved by a full step
(a rounding-boundary crossing), and the rows where S_hi changed by at least one step while S did not move. CPU only;
no model, no inference. Writes JSON to --out (never into the run directory).

    python3 experiments/experiment02/errata/precision_crossings.py \
        --effects experiments/experiment02/runs/run_002/results/temporal_effects.csv --out /tmp/precision_crossings.json
"""
import argparse, csv, json, math
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--effects", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.effects, encoding="utf-8", newline="")))
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else math.nan
scored = [r for r in rows if not math.isnan(f(r, "Delta")) and not math.isnan(f(r, "Delta_bf16"))]
by = Counter(); examples = []
for r in scored:
    d, db, res = f(r, "Delta"), f(r, "Delta_bf16"), f(r, "S_resolution")
    sub = abs(d) < res
    key = (r["case_id"], r["family"])
    by[key + ("rows",)] += 1
    if sub:
        by[key + ("sub_step",)] += 1
        if db != 0:
            by[key + ("sub_step_but_bf16_moved",)] += 1
            if len(examples) < 6:
                examples.append({k: r[k] for k in ("case_id", "family", "mask_type", "mask_slots", "S_baseline", "S_perturbed",
                                                   "Delta", "S_bf16_baseline", "S_bf16_perturbed", "Delta_bf16", "S_resolution")})
    elif db == 0:
        by[key + ("step_or_more_but_bf16_unchanged",)] += 1
    if abs(db) >= res and not sub:
        by[key + ("both_moved",)] += 1
    if d != 0 and db != 0 and (d > 0) != (db > 0):
        by[key + ("opposite_sign",)] += 1
    if (int(r["Delta_below_resolution"]) == 1) != sub:
        by[key + ("flag_disagrees_with_recomputation",)] += 1
    if r["mask_type"] != "all_eight":
        c = (r["case_id"], "non_all_eight")
        by[c + ("rows",)] += 1
        by[c + ("step_or_more_on_S_hi",)] += abs(d) >= res
        by[c + ("native_S_moved",)] += db != 0
        by[c + ("sub_step_but_bf16_moved",)] += sub and db != 0
        by[c + ("step_or_more_but_bf16_unchanged",)] += (not sub) and db == 0
        by[(r["case_id"], "max_abs_Delta_non_all_eight")] = max(by[(r["case_id"], "max_abs_Delta_non_all_eight")], abs(d))

cases = sorted({(k[0], k[1]) for k in by if len(k) == 3 and k[1] in ("blur", "constant")})
table = [{"case_id": c, "family": fam, **{m: by[(c, fam, m)] for m in (
    "rows", "sub_step", "sub_step_but_bf16_moved", "step_or_more_but_bf16_unchanged", "opposite_sign",
    "flag_disagrees_with_recomputation")}}
    for c, fam in cases]
tot = {m: sum(t[m] for t in table) for m in table[0] if m not in ("case_id", "family")}
per_case = [{"case_id": c, **{m: by[(c, "non_all_eight", m)] for m in (
    "rows", "step_or_more_on_S_hi", "native_S_moved", "sub_step_but_bf16_moved", "step_or_more_but_bf16_unchanged")},
    "max_abs_Delta": round(by[(c, "max_abs_Delta_non_all_eight")], 6)} for c in sorted({c for c, _ in cases})]
out = {"source": a.effects, "rows_total": len(rows), "rows_scored": len(scored), "totals": tot, "by_case_family": table,
       "non_all_eight_by_case": per_case, "examples_sub_step_but_bf16_moved": examples}
json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
print(json.dumps({"rows_scored": len(scored), **tot}))
for t in table + per_case: print(t)

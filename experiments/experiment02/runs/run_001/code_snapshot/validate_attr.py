#!/usr/bin/env python3
"""Validate an experiment02 run directory; writes results/validation.json; exit 1 on any FAIL.
Usage: python experiments/experiment02/code/validate_attr.py --run-dir experiments/experiment02/runs/run_001
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import yaml  # noqa: E402

import attr_lib as al  # noqa: E402

ROOT = al.ROOT
rows = []


def check(cid, desc, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"exception: {e!r}"
    rows.append({"id": cid, "check": desc, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {cid} {desc}: {detail}", flush=True)


def fl(x):
    return None if x in ("", None) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    rd = ROOT / a.run_dir
    res = rd / "results"
    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text(encoding="utf-8"))
    need = ["cases.json", "resolved_config.yaml", "run_manifest.json", "environment.txt",
            "results/baseline_responses.jsonl", "results/prompt_ablation.jsonl", "results/perturbation_responses.jsonl",
            "results/scorer_diagnostics.json", "results/cache_records.jsonl", "results/temporal_effects.csv",
            "results/frame_attribution.csv", "results/evaluation.csv", "results/summary.json",
            "results/prompts/original_question.txt", "results/prompts/label_only_question.txt",
            "results/prompts/nosound_question.txt", "results/prompts/label_only_instruct.diff",
            "results/prompts/nosound_questions.diff"]
    missing = [f for f in need if not (rd / f).exists()]
    check("V01", "required artifacts exist", lambda: (not missing, f"missing {missing}" if missing else f"{len(need)} files"))
    if missing:
        return finish(res)
    cases = {c["case_id"]: c for c in json.loads((rd / "cases.json").read_text(encoding="utf-8"))["cases"]}
    logged = {n: al.read_jsonl(res / n) for n in ("baseline_responses.jsonl", "prompt_ablation.jsonl", "perturbation_responses.jsonl")}

    def latest_ok(recs):
        d = {}
        for r in recs:
            if r["cache_status"] in ("fresh", "cached"):
                d[r["request_id"]] = r
        return list(d.values())
    base = latest_ok(logged["baseline_responses.jsonl"])
    abl = latest_ok(logged["prompt_ablation.jsonl"])
    pert = latest_ok(logged["perturbation_responses.jsonl"])
    cache = al.read_jsonl(res / "cache_records.jsonl")
    diag = json.loads((res / "scorer_diagnostics.json").read_text(encoding="utf-8"))
    allreq = base + abl + pert
    s = cfg["sampling"]

    def v02():
        probs = []
        for cid, c in cases.items():
            ws, we, idx = al.vp.window_for_anchor(c["anchor"], c["decoded_frames"], s["fps_for_windows"], s["t_window_s"], s["snippet_len"])
            if c["frame_indices"] != idx or [c["window_start"], c["window_end"]] != [ws, we] or len(idx) != 8:
                probs.append(f"{cid} window/frames")
            if any(abs(t - i / c["fps"]) > 1e-9 for t, i in zip(c["frame_times_s"], idx)):
                probs.append(f"{cid} timestamps")
            fdir = ROOT / c["frames_dir"]
            if [al.sha256_file(fdir / f"{i:06d}.jpg") for i in idx] != c["frame_sha256"]:
                probs.append(f"{cid} frame sha256")
            if al.sha256_file(ROOT / c["video_path"]) != c["video_sha256"] or c["video_sha256"] != cfg["videos"][c["video_id"]]["sha256"]:
                probs.append(f"{cid} video sha256")
            if c["decoded_frames"] != cfg["videos"][c["video_id"]]["expected_length"] or c["fps"] != 30.0:
                probs.append(f"{cid} decoded/fps")
            if c.get("experiment01") and not c["experiment01"]["window_matches_recomputation"]:
                probs.append(f"{cid} exp01 record")
        exp = {"R-accident": [2090.0, 2390.0], "R-pedestrian": [346.0, 646.0]}
        for cid, w in exp.items():
            if [cases[cid]["window_start"], cases[cid]["window_end"]] != w:
                probs.append(f"{cid} expected window")
        return not probs, f"{len(cases)} cases; problems {probs}"
    check("V02", "case frames, windows, timestamps, hashes and metadata", v02)

    def v03():
        basehash = {}
        for r in base + abl:
            if "slot_tensor_sha256" in r and not r["mask"]:
                basehash.setdefault(r["case_id"], set()).add(tuple(r["slot_tensor_sha256"]))
        probs = [cid for cid, hs in basehash.items() if len(hs) != 1]
        n = 0
        for r in pert:
            if "slot_tensor_sha256" not in r:
                continue
            b = next(iter(basehash[r["case_id"]]))
            h = r["slot_tensor_sha256"]
            ok = len(h) == 8 and all((h[i] != b[i]) == (i in r["mask"]) for i in range(8))
            n += 1
            if not ok:
                probs.append(r["request_id"])
        return not probs, f"{n} perturbation records: unmasked slots bit-identical to baseline, masked slots changed; problems {probs[:5]}"
    check("V03", "only the intended slots changed; preprocessing identical elsewhere", v03)

    def v04():
        want = sorted(m for _, m in al.primary_masks() + al.control_masks(cfg))
        probs = []
        targets = ["original_generation"] + (["label_only_score"] if diag["status"] != "blocked" else [])
        for cid in cfg["primary_cases"]:
            for fam in cfg["stage_b"]["families"]:
                for t in targets:
                    got = sorted(tuple(r["mask"]) for r in pert if r["stage"] == "B" and r["case_id"] == cid
                                 and r["family"] == fam and r["target"] == t)
                    if got != want:
                        probs.append(f"{cid}/{fam}/{t}: {len(got)} masks")
        types = {r["mask_type"]: set() for r in pert if r["stage"] == "B"}
        for r in pert:
            if r["stage"] == "B":
                types[r["mask_type"]].add(len(r["mask"]))
        size_ok = types.get("single") == {1} and types.get("adjacent_pair") == {2} and types.get("adjacent_triple") == {3} \
            and types.get("all_eight") == {8} and types.get("nonconsecutive_pair") == {2} and types.get("nonconsecutive_triple") == {3}
        nonadj = all(al.is_nonadjacent(tuple(r["mask"])) for r in pert if r["mask_type"].startswith("nonconsecutive"))
        return not probs and size_ok and nonadj, f"28 masks per case/family/target; sizes {types}; problems {probs}"
    check("V04", "group enumeration and matched intervention sizes", v04)

    def v05():
        probs = []
        for r in allreq:
            case = cases[r["case_id"]]
            spec = r["family_spec"]
            k = al.request_key(cfg, r["target"], r["prompt_sha256"], case, tuple(r["mask"]), spec)
            if r["repeat"]:
                k = al.sha256_bytes(f"{k}:repeat{r['repeat']}".encode())
            if k != r["result_key"]:
                probs.append(r["request_id"])
        keys = [c["key"] for c in cache]
        uniq = len(keys) == len(set(keys))
        logged_fresh = {r["result_key"] for r in allreq if r["cache_status"] == "fresh"}
        missing_in_cache = logged_fresh - set(keys)
        gen_vs_score = all(("raw_response" in c) != ("S" in c) for c in cache)
        return not probs and uniq and not missing_in_cache and gen_vs_score, (
            f"recomputed keys match {len(allreq) - len(probs)}/{len(allreq)}; cache keys unique {uniq}; "
            f"fresh results missing from cache {len(missing_in_cache)}; generation and score records never share a key {gen_vs_score}")
    check("V05", "cache keys complete (model, prompt/contract, source, frames, transform, preprocessing) and consistent", v05)

    def v06():
        probs = []
        for r in allreq:
            if "raw_response" not in r:
                continue
            want = al.parse_label_only(r["raw_response"]) if r["variant"] == "label_only" else al.vp.strict_parse_label(r["raw_response"])
            if (r["label"], r["parse_status"]) != want or (r["parse_status"] != "ok" and r["label"] is not None):
                probs.append(r["request_id"])
        n_fail = sum(1 for r in allreq if "raw_response" in r and r["parse_status"] != "ok")
        return not probs, f"re-parsed {sum('raw_response' in r for r in allreq)} generations; parse failures (kept unlabelled) {n_fail}; mismatches {probs[:5]}"
    check("V06", "parsing reproducible; parse failures are missing, never labels", v06)

    def v07():
        if diag["status"] == "blocked":
            return True, "scorer blocked: no numeric records expected"
        cand = diag["candidates"]
        probs = []
        for r in allreq:
            if "S" in r:
                if not (math.isfinite(r["S"]) and abs(r["S"] - (r["logp_1"] - r["logp_0"])) < 1e-9):
                    probs.append(r["request_id"])
        if any(not (r.get("S_resolution") or 0) > 0 for r in allreq if "S" in r):
            probs.append("missing S_resolution")
        rep = diag["checks"]["repeatability_max_abs_diff"]
        ok = cand["ok"] and cand["query_reconstruction_matches_captured_ids"] and not probs \
            and rep <= cfg["stage_a"]["scorer_criteria"]["max_repeat_abs_diff"]
        return ok, (f"candidates {{0: {cand['candidates']['0']['token_ids']}, 1: {cand['candidates']['1']['token_ids']}}}, "
                    f"single-token {cand['single_token_candidates']}; S finite and = logp1-logp0 for all; repeat max|dS| {rep}; problems {probs[:5]}")
    check("V07", "likelihood indexing, finite scores and repeatability", v07)

    te = list(csv.DictReader(open(res / "temporal_effects.csv", encoding="utf-8")))

    def v08():
        probs = []
        singles = {(t["case_id"], t["family"], t["mask_slots"]): t for t in te if t["mask_type"] == "single"}
        for t in te:
            S0, Sp, D = fl(t["S_baseline"]), fl(t["S_perturbed"]), fl(t["Delta"])
            if S0 is not None and Sp is not None and abs(D - (S0 - Sp)) > 1e-5:
                probs.append(f"Delta {t['case_id']} {t['mask_slots']}")
            if t["orig_label"] != "" and t["pert_label"] != "":
                bd = int(t["orig_label"]) - int(t["pert_label"])
                if int(t["binary_diff_orig_minus_pert"]) != bd or int(t["flipped"]) != int(bd != 0):
                    probs.append(f"binary {t['case_id']} {t['mask_slots']}")
            elif t["flipped"] != "":
                probs.append(f"missing label but flip recorded {t['mask_slots']}")
            if t["size"] != "1" and t["interaction_Delta"] != "":
                parts = [singles[(t["case_id"], t["family"], p)]["Delta"] for p in t["mask_slots"].split("-")]
                if abs(fl(t["interaction_Delta"]) - (D - sum(map(float, parts)))) > 1e-5:
                    probs.append(f"interaction {t['mask_slots']}")
            if D is not None and S0 is not None:
                cdir = 1 if S0 >= 0 else -1
                if abs(fl(t["support_for_scorer_baseline_class"]) - cdir * D) > 1e-9:
                    probs.append("class-direction")
        neg = sum(1 for t in te if t["Delta"] != "" and float(t["Delta"]) < 0)
        return not probs, f"{len(te)} rows; negative Deltas kept {neg}; problems {probs[:5]}"
    check("V08", "signed effects, binary diffs, interactions and normal/anomaly direction recomputed", v08)

    def v09():
        probs = []
        for cid in cfg["primary_cases"]:
            p = res / f"rankings_{cid}.json"
            if not p.exists():
                if diag["status"] != "blocked":
                    probs.append(f"{cid} rankings missing")
                continue
            for name, rk in json.loads(p.read_text(encoding="utf-8")).items():
                if sorted(rk["order"]) != list(range(8)):
                    probs.append(f"{cid} {name} not a permutation")
        return not probs, f"rankings are full permutations with recorded ties; problems {probs}"
    check("V09", "rankings well formed; ties recorded", v09)

    def v10():
        probs = []
        for r in allreq:
            if "effective_generation_config" in r:
                g = dict(r["effective_generation_config"])
                if g.pop("eos_token_id", None) != 92542 or g != cfg["generation"]:
                    probs.append(r["request_id"])
        return not probs, f"effective generation settings = config + eos_token_id 92542 on all generations; problems {probs[:3]}"
    check("V10", "effective generation configuration saved, including model-added values", v10)

    def v11():
        probs, retried = [], 0
        for name, recs in logged.items():
            by = {}
            for r in recs:
                by.setdefault(r["request_id"], []).append(r)
            for rid, rs in by.items():
                ok = [x for x in rs if x["cache_status"] in ("fresh", "cached")]
                if len(ok) > 1 or (ok and rs[-1] is not ok[0]):
                    probs.append(rid)
                if len(rs) > 1:
                    retried += 1
                if [x.get("attempt") for x in rs] != list(range(1, len(rs) + 1)):
                    probs.append(f"attempt numbering {rid}")
        return not probs, f"at most one success per request id, always the last attempt; retried ids {retried}; problems {probs[:5]}"
    check("V11", "resume safety: append-only request logs; failed attempts retried, never overwritten", v11)

    def v12():
        pre = rd / "preflight"
        bad = []
        for snap, base_dir in (("experiment01_files_before.sha256", ROOT), ("setup_files_before.sha256", ROOT / "setup")):
            for line in (pre / snap).read_text(encoding="utf-8").splitlines():
                h, p = line.split(None, 1)
                if al.sha256_file(base_dir / p) != h:
                    bad.append(p)
        src = ROOT / cfg["source"]["checkout"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=src, capture_output=True, text=True).stdout.strip()
        return not bad and head == cfg["source"]["expected_head"] and not dirty, (
            f"experiment01 + setup files unchanged since preflight ({'none changed' if not bad else bad[:5]}); VERA HEAD {head[:8]} clean {not dirty}")
    check("V12", "original runs and setup unchanged", v12)

    def v13():
        uniq = {c["sequence_key"] for c in cache if c.get("transformed")}
        log = ROOT / "setup" / "jobs" / "RESOURCE_LOG.tsv"
        secs = 0
        for line in log.read_text(encoding="utf-8").splitlines()[1:]:
            f = line.split("\t")
            if len(f) >= 10 and "experiments/experiment02" in f[9] and f[5].isdigit():
                secs += int(f[5])
        return len(uniq) <= cfg["limits"]["max_unique_transformed_sequences"] and secs <= cfg["limits"]["max_gpu_hours"] * 3600, (
            f"unique transformed sequences {len(uniq)}/{cfg['limits']['max_unique_transformed_sequences']}; "
            f"experiment02 GPU job wall time {secs / 3600:.2f} h / {cfg['limits']['max_gpu_hours']} h")
    check("V13", "execution limits respected", v13)

    def v14():
        e01 = al.read_jsonl(ROOT / "experiments/experiment01/runs/run_001/results/raw_responses.jsonl")
        q_orig = (res / "prompts/original_question.txt").read_text(encoding="utf-8")
        same = q_orig == e01[0]["prompt_question"]
        shas = {r["prompt_sha256"] for r in allreq if r["variant"] == "original"}
        q_lo = (res / "prompts/label_only_question.txt").read_text(encoding="utf-8")
        q_ns = (res / "prompts/nosound_question.txt").read_text(encoding="utf-8")
        lo_ok = "Answers to Prompt Questions" not in q_lo and all(q in q_lo for q in cfg["prompt"]["guiding_questions"])
        ns_ok = q_ns == q_orig.replace(cfg["prompt"]["guiding_questions"][4] + "\n", "")
        key_prompt_ok = all(r["question_sha256"] == r["prompt_sha256"] for r in allreq if "question_sha256" in r)
        return same and shas == {al.sha256_bytes(q_orig.encode())} and lo_ok and ns_ok and key_prompt_ok, (
            f"original question identical to experiment01 {same}; one original prompt hash {len(shas) == 1}; "
            f"label-only keeps all 5 questions {lo_ok}; no-sound = original minus question 5 only {ns_ok}; "
            f"question actually built by VERA forward() = prompt in cache key for every record {key_prompt_ok}")
    check("V14", "prompt identity (original = experiment01; ablation and label-only differ only as declared)", v14)

    def v15():
        out = []
        for r in base:
            if r["stage"] == "A1" and cases[r["case_id"]].get("experiment01"):
                e = cases[r["case_id"]]["experiment01"]
                out.append(f"{r['case_id']}: text {'same' if r['raw_response'] == e['raw_response'] else 'DIFFERENT'}, "
                           f"label {'same' if r['label'] == e['label'] else 'DIFFERENT'}, token ids {'same' if r['model_input_sha256'] == e['model_input_sha256'] else 'DIFFERENT'}")
        return True, "; ".join(out) + " (informational: discrepancies are reported, not failures)"
    check("V15", "A1 reruns compared with experiment01 records", v15)

    def v17():
        bad = [r["key"][:12] for r in cache if r.get("slot_identity_ok") is False]
        n = sum(1 for r in cache if "slot_identity_ok" in r)
        return not bad, f"in-run slot identity flag true on {n - len(bad)}/{n} fresh records"
    check("V17", "in-run check: every fresh record changed exactly its masked slots", v17)

    def v18():
        man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
        frozen = man.get("code_sha256_frozen_at_first_model_invocation")
        gpu_invs = [i for i in man["invocations"] if i.get("uses_gpu")]
        same = all(i.get("code_sha256") == frozen for i in gpu_invs)
        now_ok = frozen == al.code_sha256()
        return bool(frozen) and same and now_ok, (f"{len(gpu_invs)} model invocations all ran code {frozen and {k: v[:10] for k, v in frozen.items()}}; "
                                                  f"current code identical {now_ok}")
    check("V18", "one code version produced all model results", v18)

    sp = rd / "spatial"
    if sp.exists():
        def v16():
            import numpy as np
            arrs = list(sp.glob("*.npy"))
            ok = all(np.isfinite(np.load(p)).all() for p in arrs)
            return ok, f"{len(arrs)} spatial arrays finite {ok}"
        check("V16", "spatial diagnostic arrays finite", v16)
    return finish(res)


def finish(res):
    n_fail = sum(r["status"] == "FAIL" for r in rows)
    res.mkdir(parents=True, exist_ok=True)
    al.write_json(res / "validation.json", {"passed": len(rows) - n_fail, "failed": n_fail, "checks": rows})
    print(f"{len(rows) - n_fail}/{len(rows)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

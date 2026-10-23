#!/usr/bin/env python3
"""experiment02 runner (temporal_attribution_feasibility).

Usage (on oz, from ~/vera_research; GPU stages through setup/scripts/launch.sh):
  python experiments/experiment02/code/run_attr.py --config experiments/experiment02/config.yaml \
      --run-dir experiments/experiment02/runs/run_001 --stages prep|A|B|C

prep  (CPU)  verify/extract frames into the shared cache, build cases.json, prompts and the label-only diff
A     (GPU)  A1 original-prompt baseline reruns, A2 label-only generation + numeric score checks, A3 sound ablation
B     (GPU)  22 masks + 6 nonconsecutive controls per primary case and replacement family
C     (GPU)  cross-family removal/retention evaluation of singleton, temporal-group and random rankings

Every model call is a request with a cache key (attr_lib.request_key). Fresh results are appended once to
results/cache_records.jsonl; each stage file (baseline_responses.jsonl, prompt_ablation.jsonl,
perturbation_responses.jsonl) gets one self-contained line per request with cache_status fresh|cached.
Rerunning a stage skips requests already logged, so a crash never repeats finished work.
"""
import argparse
import difflib
import fcntl
import json
import os
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import yaml  # noqa: E402

import math  # noqa: E402

import attr_lib as al  # noqa: E402

ROOT = al.ROOT
sys.path.insert(0, str(ROOT / "setup" / "scripts"))
import run_experiment as rx  # noqa: E402  (setup loaders: frame cache, environment report)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- run directory

_LOCK = None


def open_run(args):
    global _LOCK
    cfg_path = ROOT / args.config
    raw = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    rd = ROOT / args.run_dir
    (rd / "results").mkdir(parents=True, exist_ok=True)
    (rd / "logs").mkdir(exist_ok=True)
    _LOCK = open(rd / ".run.lock", "w")
    try:
        fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"{rd} is locked by another invocation")
    frozen = rd / "resolved_config.yaml"
    man_p = rd / "run_manifest.json"
    if frozen.exists() and frozen.read_text(encoding="utf-8") != raw:
        model_calls = al.read_jsonl(rd / "results" / "cache_records.jsonl")
        if not args.amend_config_before_model_calls or model_calls:
            sys.exit(f"{cfg_path} differs from {frozen}: make a new config and a new run directory "
                     "(--amend-config-before-model-calls is accepted only while no model call has been made)")
        old = frozen.read_text(encoding="utf-8")
        k = len(list(rd.glob("resolved_config.superseded_*.yaml"))) + 1
        (rd / f"resolved_config.superseded_{k}.yaml").write_text(old, encoding="utf-8")
        frozen.write_text(raw, encoding="utf-8")
        if man_p.exists():
            m = json.loads(man_p.read_text(encoding="utf-8"))
            m.setdefault("config_amendments", []).append({
                "at": now(), "old_config_sha256": al.sha256_bytes(old.encode()), "new_config_sha256": al.sha256_bytes(raw.encode()),
                "superseded_copy": f"resolved_config.superseded_{k}.yaml", "model_calls_before_amendment": 0,
                "reason": args.amend_config_before_model_calls})
            m["config_sha256"] = al.sha256_bytes(raw.encode())
            al.write_json(man_p, m)
    if not frozen.exists():
        frozen.write_text(raw, encoding="utf-8")
    src = ROOT / cfg["source"]["checkout"]
    head = rx.git("rev-parse", "HEAD", cwd=src)
    if head != cfg["source"]["expected_head"] or rx.git("status", "--porcelain", "--untracked-files=no", cwd=src):
        sys.exit(f"source checkout is not the clean expected head (HEAD {head})")
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {
        "experiment_id": cfg["experiment_id"], "purpose": cfg["purpose"], "run_id": rd.name, "created_at": now(),
        "host": socket.getfqdn(), "config_sha256": al.sha256_bytes(raw.encode()), "source_head": head,
        "model": cfg["model"], "limits": cfg["limits"], "invocations": [], "stages": {}}
    if man["config_sha256"] != al.sha256_bytes(raw.encode()):
        sys.exit("run manifest belongs to a different config")
    return cfg, rd, man


def save_manifest(rd, man):
    al.write_json(rd / "run_manifest.json", man)


# ---------------------------------------------------------------- prep (CPU)

def stage_prep(cfg, rd, man):
    src = ROOT / cfg["source"]["checkout"]
    full_manifest = json.loads((src / "Data" / "UCF_Eval.json").read_text(encoding="utf-8"))
    official = {}
    off_path = ROOT / cfg["annotations"]["official_file"]
    for line in off_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            official[parts[0]] = parts[1:]
    videos = {}
    for vid, v in cfg["videos"].items():
        vcfg = {"video": {"path": v["path"], "sha256": v["sha256"], "expected_length": v["expected_length"]},
                "source": {"checkout": cfg["source"]["checkout"]}, "frames": {"cache_root": cfg["frames"]["cache_root"]}}
        vdir = rd / "videos" / vid
        (vdir / "results").mkdir(parents=True, exist_ok=True)
        info = rx.stage_frames(vcfg, vdir, None)  # setup loader: checksum, VERA extractor, identity, count check
        ident = json.loads((vdir / "results" / "frames_identity.json").read_text(encoding="utf-8"))
        shas = {}
        for line in (ROOT / ident["per_frame_sha256_file"]).read_text(encoding="utf-8").splitlines():
            h, name = line.split()
            shas[int(name[:6])] = h
        entry = [e for e in full_manifest if e["video"] == v["manifest_entry"]]
        if len(entry) != 1:
            sys.exit(f"{vid}: expected one UCF_Eval.json entry")
        videos[vid] = {"identity": ident, "frame_sha": shas, "manifest": entry[0],
                       "official_annotation": official.get(vid + ".mp4"), "reused_cache": info["reused_cache"]}

    s = cfg["sampling"]
    cases = []
    for c in cfg["cases"]:
        v = videos[c["video_id"]]
        n = v["identity"]["decoded_frame_count"]
        fps = v["identity"]["video_metadata_cv2"]["container_fps"]
        anchors = al.vp.anchors(n, s["sampling_rate"])
        if "anchor" in c:
            anchor = c["anchor"]
            anchor_rule = "fixed in config"
        else:  # anchor in range(0, n, 16) closest to n/2, ties to the earlier
            anchor = min(anchors, key=lambda a: (abs(a - n / 2), a))
            anchor_rule = "anchor in range(0, n, 16) closest to n/2 (ties to the earlier)"
        if anchor not in anchors:
            sys.exit(f"{c['case_id']}: anchor {anchor} is not a VERA anchor")
        ws, we, idx = al.vp.window_for_anchor(anchor, n, s["fps_for_windows"], s["t_window_s"], s["snippet_len"])
        ann = v["manifest"]["temporal_label"]
        intervals = [(ann[i], min(ann[i + 1], n)) for i in range(0, len(ann), 2) if ann[i] != -1]
        overlap = sum(max(0, min(we, b) - max(ws, a)) for a, b in intervals)
        case = {
            "case_id": c["case_id"], "role": c["role"], "video_id": c["video_id"],
            "video_path": cfg["videos"][c["video_id"]]["path"], "video_sha256": v["identity"]["source_sha256"],
            "decoded_frames": n, "fps": fps, "anchor": anchor, "anchor_rule": anchor_rule,
            "anchor_time_s": anchor / fps, "window_start": ws, "window_end": we,
            "window_time_s": [ws / fps, we / fps], "frame_indices": idx,
            "frame_times_s": [i / fps for i in idx], "frame_files": [f"{i:06d}.jpg" for i in idx],
            "frame_sha256": [v["frame_sha"][i] for i in idx], "frames_dir": v["identity"]["frames_dir"],
            "frames_identity_sha256": v["identity"]["frames_digest_sha256"],
            "benchmark_annotation": {
                "vera_ucf_eval_json": {"temporal_label": ann, "length": v["manifest"]["length"],
                                       "video_level_answer": v["manifest"]["QA"][0]["a"]},
                "official_ucf_crime_txt": v["official_annotation"],
                "convention": "compute_auc.py marks frames [start, end) zero-based; the official txt does not state "
                              "whether indices are 0- or 1-based or end-inclusive (at most one frame at each end)",
                "annotated_intervals_used": intervals, "window_overlap_frames_with_annotation": overlap},
            "provenance": json.loads((ROOT / (cfg["videos"][c["video_id"]]["path"] + ".provenance.json")).read_text(encoding="utf-8")),
        }
        if c.get("experiment01_record"):
            recs = [r for r in al.read_jsonl(ROOT / c["experiment01_record"]) if r["anchor_frame"] == anchor]
            if len(recs) != 1:
                sys.exit(f"{c['case_id']}: no unique experiment01 record")
            r = recs[0]
            agree = (r["frame_indices"] == idx and r["window_start"] == ws and r["window_end"] == we
                     and r["video_sha256"] == case["video_sha256"])
            if not agree:
                sys.exit(f"{c['case_id']}: experiment01 window/frames differ from the recomputed ones")
            case["experiment01"] = {"record_file": c["experiment01_record"], "label": r["label"],
                                    "raw_response": r["raw_response"], "model_input_sha256": r["model_input_sha256"],
                                    "window_matches_recomputation": agree}
        if c.get("expect_window") and [ws, we] != c["expect_window"]:
            sys.exit(f"{c['case_id']}: window {[ws, we]} != expected {c['expect_window']}")
        cases.append(case)
    al.write_json(rd / "cases.json", {"generated_at": now(), "sampling": s, "cases": cases})
    # eval manifest for the VERA dataset object (transform identity) and the prompts
    al.write_json(rd / "results" / "eval_manifest_cases.json",
                  [{"video": cfg["videos"][vid]["manifest_entry"], "length": videos[vid]["manifest"]["length"]}
                   for vid in cfg["videos"]])
    write_prompts(cfg, rd)
    return {"cases": [c["case_id"] for c in cases], "frames_cache_reused": {k: v["reused_cache"] for k, v in videos.items()}}


def write_prompts(cfg, rd):
    src = ROOT / cfg["source"]["checkout"]
    instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
    lo = al.label_only_instruct(instruct)
    qs = "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"])
    ns = al.nosound_questions(cfg["prompt"]["guiding_questions"])
    pdir = rd / "results" / "prompts"
    pdir.mkdir(exist_ok=True)
    (pdir / "label_only_instruct.txt").write_text(lo, encoding="utf-8")
    (pdir / "label_only_instruct.diff").write_text("".join(difflib.unified_diff(
        instruct.splitlines(True), lo.splitlines(True), "VERA_learner_instruct.txt", "label_only_instruct.txt")), encoding="utf-8")
    (pdir / "nosound_questions.diff").write_text("".join(difflib.unified_diff(
        qs.splitlines(True), ns.splitlines(True), "new_qs (released UCF questions)", "nosound_questions")), encoding="utf-8")


# ---------------------------------------------------------------- request store

class Store:
    def __init__(self, cfg, rd, gpu_seconds_prior):
        self.cfg, self.rd = cfg, rd
        self.cache_path = rd / "results" / "cache_records.jsonl"
        self.cache = {r["key"]: r for r in al.read_jsonl(self.cache_path)}
        self.logged = {}
        self.cap = cfg["limits"]["max_unique_transformed_sequences"]
        self.gpu_cap_s = cfg["limits"]["max_gpu_hours"] * 3600
        self.gpu_seconds_prior = gpu_seconds_prior
        self.t_start = time.time()
        self.errors_this_invocation = 0

    def gpu_seconds_used(self):
        return self.gpu_seconds_prior + (time.time() - self.t_start)

    def transformed_sequences(self):
        return {r["sequence_key"] for r in self.cache.values() if r.get("transformed")}

    def stage_ids(self, fname):
        """Latest logged attempt per request id; error/skipped attempts stay in the file but are retried."""
        if fname not in self.logged:
            d, n = {}, {}
            for r in al.read_jsonl(self.rd / "results" / fname):
                d[r["request_id"]] = r
                n[r["request_id"]] = n.get(r["request_id"], 0) + 1
            self.logged[fname] = d
            self.attempts = getattr(self, "attempts", {})
            self.attempts[fname] = n
        return self.logged[fname]

    def done(self, fname, request_id):
        r = self.stage_ids(fname).get(request_id)
        return r if r is not None and r["cache_status"] in ("fresh", "cached") else None

    def request(self, fname, request_id, key, seq_key, transformed, meta, compute):
        prev = self.done(fname, request_id)
        if prev is not None:
            return prev
        self.stage_ids(fname)
        attempts = self.attempts[fname].get(request_id, 0) + 1
        self.attempts[fname][request_id] = attempts
        fatal = None
        if key in self.cache:
            res, status = self.cache[key], "cached"
        elif self.gpu_seconds_used() >= self.gpu_cap_s:
            res, status = {"error": f"skipped: {self.cfg['limits']['max_gpu_hours']} GPU-hour cap reached"}, "skipped_budget"
        elif transformed and seq_key not in self.transformed_sequences() and len(self.transformed_sequences()) >= self.cap:
            res, status = {"error": f"skipped: {self.cap}-sequence cap reached"}, "skipped_budget"
        else:
            try:
                res = compute()
            except Exception as e:  # recorded as an error state, never as a label
                import traceback
                res = {"error": repr(e), "traceback": traceback.format_exc()[-2000:]}
                if "CUDA" in repr(e) or "cuda" in type(e).__name__.lower():
                    fatal = e  # sticky device errors would poison every later request: stop the stage
            res = {"key": key, "sequence_key": seq_key, "transformed": transformed, "computed_at": now(), **res}
            if "error" not in res:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(al.jsonl_line(res))
                self.cache[key] = res
            status = "fresh" if "error" not in res else "error"
        if status in ("error", "skipped_budget"):
            self.errors_this_invocation += 1
        rec = {"request_id": request_id, **meta, "cache_status": status, "result_key": key, "attempt": attempts,
               **{k: v for k, v in res.items() if k not in ("key",)}}
        with open(self.rd / "results" / fname, "a", encoding="utf-8") as f:
            f.write(al.jsonl_line(rec))
        self.stage_ids(fname)[request_id] = rec
        if fatal is not None:
            raise RuntimeError(f"stopping stage after device error: {fatal!r}")
        return rec


def load_cases(rd):
    return {c["case_id"]: c for c in json.loads((rd / "cases.json").read_text(encoding="utf-8"))["cases"]}


def make_engine(cfg, rd):
    import engine
    return engine.Engine(cfg, rd / "results" / "eval_manifest_cases.json")


def run_request(store, eng, fname, stage, target, case, mask, family, extra_meta=None, repeat=0):
    cfg = store.cfg
    spec = al.family_spec(cfg, family if mask else "none")
    variant = {"original_generation": "original", "nosound_generation": "nosound",
               "label_only_generation": "label_only", "label_only_score": "label_only"}[target]
    prompt_sha = al.sha256_bytes(eng.effective_question(variant).encode()) if eng else PROMPT_SHA[variant]
    key = al.request_key(cfg, target, prompt_sha, case, mask, spec)
    if repeat:
        key = al.sha256_bytes(f"{key}:repeat{repeat}".encode())
    seq_key = al.sequence_key(case, mask, spec)
    rid = f"{stage}|{target}|{case['case_id']}|{family if mask else 'none'}|{','.join(map(str, sorted(mask)))}|r{repeat}"
    if extra_meta and extra_meta.get("request_suffix"):
        rid += "|" + extra_meta["request_suffix"]
    meta = {"stage": stage, "target": target, "variant": variant, "case_id": case["case_id"],
            "video_id": case["video_id"], "anchor": case["anchor"], "frame_indices": case["frame_indices"],
            "mask": sorted(mask), "mask_size": len(mask), "family": family if mask else "none",
            "family_spec": spec, "prompt_sha256": prompt_sha, "repeat": repeat, **(extra_meta or {})}
    if target == "label_only_score":
        compute = lambda: eng.score(case, mask, spec)  # noqa: E731
    else:
        compute = lambda: eng.generate(variant, case, mask, spec)  # noqa: E731
    return store.request(fname, rid, key, seq_key, bool(mask), meta, compute)


PROMPT_SHA = {}


# ---------------------------------------------------------------- stage A

def scorer_class(S):
    """Single tie rule everywhere (config): S >= 0 -> anomaly (1), S < 0 -> normal (0); None stays None."""
    return None if S is None else int(S >= 0)


def finite(x):
    return x is not None and math.isfinite(x)


def stage_a(cfg, rd, man, eng, store):
    cases = load_cases(rd)
    out = {"A1": [], "A2": {}, "A3": []}
    for cid in cfg["stage_a"]["cases"]:
        c = cases[cid]
        r = run_request(store, eng, "baseline_responses.jsonl", "A1", "original_generation", c, (), "none")
        cmp = None
        if c.get("experiment01"):
            e = c["experiment01"]
            cmp = {"raw_text_identical": r.get("raw_response") == e["raw_response"],
                   "label_identical": r.get("label") == e["label"],
                   "token_ids_identical": r.get("model_input_sha256") == e["model_input_sha256"],
                   "pixel_identity": "experiment01 stored no pixel hashes; this pixel path is bit-identical to VERA's "
                                     "main loop by test_attr_lib.py, and the frame JPEGs match cases.json hashes"}
        out["A1"].append({"case_id": cid, "label": r.get("label"), "parse_status": r.get("parse_status"),
                          "cache_status": r["cache_status"], "vs_experiment01": cmp})
    # A2: label-only generation, numeric score, repeatability, alignment, strong alterations
    lo_gen, scores, reps, align = {}, {}, {}, []
    for cid in cfg["stage_a"]["cases"]:
        c = cases[cid]
        g = run_request(store, eng, "baseline_responses.jsonl", "A2", "label_only_generation", c, (), "none")
        lo_gen[cid] = {"label": g.get("label"), "parse_status": g.get("parse_status"), "raw_response": g.get("raw_response")}
        sc = [run_request(store, eng, "baseline_responses.jsonl", "A2", "label_only_score", c, (), "none", repeat=k)
              for k in range(cfg["stage_a"]["score_repeats"])]
        scores[cid] = sc[0]
        vals = [x.get("S") for x in sc]
        reps[cid] = {"S_repeats": vals,
                     "max_abs_diff": (max(vals) - min(vals)) if all(finite(v) for v in vals) else None}
    for cid in cfg["stage_a"]["cases"]:
        try:
            align.append(eng.alignment_check(cases[cid]))
        except Exception as e:  # noqa: BLE001
            align.append({"case_id": cid, "error": repr(e)})
    strong = {}
    for cid in cfg["primary_cases"]:
        for fam in cfg["stage_b"]["families"]:
            r = run_request(store, eng, "perturbation_responses.jsonl", "B", "label_only_score", cases[cid],
                            tuple(range(al.N_SLOTS)), fam, {"mask_type": "all_eight"})
            S0, Sp = scores[cid].get("S"), r.get("S")
            strong[f"{cid}|{fam}"] = {"S": Sp, "S_baseline": S0,
                                      "abs_change": abs(Sp - S0) if finite(Sp) and finite(S0) else None,
                                      "S_resolution": r.get("S_resolution"), "S_hi": r.get("S_hi"),
                                      "S_hi_baseline": scores[cid].get("S_hi"),
                                      "S_hi_ok": r.get("S_hi_rounds_to_model_logits", True)}
    # A3 sound-question ablation (a parse failure is missing, never a change)
    for cid in cfg["stage_a"]["cases"]:
        c = cases[cid]
        r = run_request(store, eng, "prompt_ablation.jsonl", "A3", "nosound_generation", c, (), "none")
        base = store.done("baseline_responses.jsonl", f"A1|original_generation|{cid}|none||r0") or {}
        lo, ln = base.get("label"), r.get("label")
        out["A3"].append({"case_id": cid, "label_original": lo, "label_nosound": ln,
                          "label_changed": None if lo is None or ln is None else lo != ln,
                          "parse_missing": [n for n, v in (("original", lo), ("nosound", ln)) if v is None],
                          "sound_sentences_original": base.get("sound_sentences"),
                          "sound_sentences_nosound": r.get("sound_sentences")})
    out["A2"] = scorer_decision(cfg, eng, cases, lo_gen, scores, reps, align, strong, store)
    al.write_json(rd / "results" / "scorer_diagnostics.json", out["A2"])
    al.write_json(rd / "results" / "stage_a_summary.json", out)
    return {"scorer_status": out["A2"]["status"], "A1": out["A1"], "A3": out["A3"]}


def scorer_decision(cfg, eng, cases, lo_gen, scores, reps, align, strong, store):
    """Pre-declared gate (config stage_a.scorer_criteria). Anything missing or non-finite counts as a failure."""
    crit = cfg["stage_a"]["scorer_criteria"]
    undecided_ties = cfg["stage_a"].get("exact_tie_handling") == "undecided"

    def agree(S, lab):
        if S is None or lab is None:
            return None
        if undecided_ties and S == 0:
            return "tie"
        return scorer_class(S) == lab
    rows = []
    for cid in cfg["stage_a"]["cases"]:
        S = scores[cid].get("S")
        orig = (store.done("baseline_responses.jsonl", f"A1|original_generation|{cid}|none||r0") or {}).get("label")
        lg = lo_gen[cid]["label"]
        rows.append({"case_id": cid, "original_prompt_label": orig, "label_only_generation_label": lg,
                     "label_only_generation_text": lo_gen[cid]["raw_response"], "S": S, "scorer_class": scorer_class(S),
                     "S_resolution": scores[cid].get("S_resolution"), "logit_0": scores[cid].get("logit_0"),
                     "logit_1": scores[cid].get("logit_1"), "candidate_mass": scores[cid].get("candidate_mass"),
                     "top5_next_tokens": scores[cid].get("top5_next_tokens"),
                     "S_hi": scores[cid].get("S_hi"),
                     "S_agrees_label_only_generation": agree(S, lg),
                     "S_agrees_original_prompt_label": agree(S, orig),
                     "repeatability": reps[cid]})
    n = len(rows)
    agree_lo = [r["S_agrees_label_only_generation"] for r in rows]
    agree_or = [r["S_agrees_original_prompt_label"] for r in rows]
    strong_vals = [v["abs_change"] for v in strong.values()]
    checks = {
        "candidate_tokenization_ok": bool(eng.candidates and eng.candidates["ok"]),
        "single_token_candidates": bool(eng.candidates and eng.candidates["single_token_candidates"]),
        "all_scores_finite": all(finite(v) for r in rows for v in r["repeatability"]["S_repeats"]),
        "repeatability_max_abs_diff": max((float("inf") if r["repeatability"]["max_abs_diff"] is None
                                           else r["repeatability"]["max_abs_diff"]) for r in rows),
        "alignment_cases": len(align),
        "alignment_max_abs_S_diff": max((a.get("abs_S_diff", float("inf")) for a in align), default=float("inf")),
        "alignment_argmax_agree": len(align) == n and all(
            a.get("argmax_scorer") is not None and a.get("argmax_scorer") == a.get("argmax_generate_step0") for a in align),
        "min_candidate_mass": min(((r["candidate_mass"] if finite(r["candidate_mass"]) else 0.0) for r in rows), default=0.0),
        "exact_ties_S_zero": sum(1 for x in agree_lo if x == "tie"),
        "S_hi_rounds_to_model_logits_all": all(x.get("S_hi_rounds_to_model_logits") is True for r in rows
                                               for x in [scores[r["case_id"]]]) and all(
            v.get("S_hi_ok", True) for v in strong.values()),
        "sign_agrees_label_only_generation": sum(1 for x in agree_lo if x is True),
        "sign_disagrees_label_only_generation": sum(1 for x in agree_lo if x is False),
        "label_only_generation_missing": sum(1 for x in agree_lo if x is None),
        "sign_agrees_original_prompt": sum(1 for x in agree_or if x is True),
        "sign_disagrees_original_prompt": sum(1 for x in agree_or if x is False),
        "original_prompt_label_missing": sum(1 for x in agree_or if x is None),
        "strong_alterations_present": len(strong_vals) == len(cfg["primary_cases"]) * len(cfg["stage_b"]["families"])
        and all(finite(v) for v in strong_vals),
        "strong_alteration_min_abs_change": min(strong_vals) if strong_vals and all(finite(v) for v in strong_vals) else None,
    }
    technical = {
        "candidates_ok_and_single_token": checks["candidate_tokenization_ok"] and checks["single_token_candidates"],
        "all_scores_finite": checks["all_scores_finite"],
        "repeatability": checks["repeatability_max_abs_diff"] <= crit["max_repeat_abs_diff"],
        "alignment_S": checks["alignment_max_abs_S_diff"] <= crit["max_alignment_abs_S_diff"],
        "alignment_argmax": checks["alignment_argmax_agree"],
        "candidate_mass": checks["min_candidate_mass"] >= crit["min_candidate_mass"],
        "sign_matches_label_only_generation": (checks["sign_disagrees_label_only_generation"] == 0
                                               and checks["label_only_generation_missing"] == 0
                                               and (checks["sign_agrees_label_only_generation"] == n or undecided_ties)),
        **({"S_hi_rounds_to_model_logits": checks["S_hi_rounds_to_model_logits_all"]} if "S_hi" in str(cfg["stage_c"].get("effect_scale")) else {}),
        "strong_alteration_moves_S": checks["strong_alterations_present"]
        and checks["strong_alteration_min_abs_change"] > crit["min_strong_alteration_abs_change"],
    }
    if not all(technical.values()):
        status = "blocked"
    elif checks["sign_agrees_original_prompt"] >= crit["min_original_agreement"]:
        status = "usable_auxiliary"
    else:
        status = "provisional_diverges_from_original"
    if status == "usable_auxiliary" and cfg["stage_a"].get("post_hoc_amendment"):
        status = "provisional_post_hoc_gate"  # rules amended after seeing run_001 Stage A: never fully usable here
    return {"status": status, "criteria": crit, "technical_gate": technical, "checks": checks, "cases": rows,
            "alignment": align, "strong_alterations": strong, "candidates": eng.candidates,
            "tie_rule": "S >= 0 -> class 1 (anomaly)",
            "note": "A 5-case agreement table is a feasibility diagnostic, not a validation of the scorer on UCF-Crime."}


# ---------------------------------------------------------------- stage B

def stage_b(cfg, rd, man, eng, store):
    cases = load_cases(rd)
    diag = json.loads((rd / "results" / "scorer_diagnostics.json").read_text(encoding="utf-8"))
    use_score = diag["status"] != "blocked"
    masks = al.primary_masks() + al.control_masks(cfg)
    n = 0
    for cid in cfg["primary_cases"]:
        c = cases[cid]
        for fam in cfg["stage_b"]["families"]:
            for mtype, m in masks:
                run_request(store, eng, "perturbation_responses.jsonl", "B", "original_generation", c, m, fam, {"mask_type": mtype})
                if use_score:
                    run_request(store, eng, "perturbation_responses.jsonl", "B", "label_only_score", c, m, fam, {"mask_type": mtype})
                n += 1
        save_example_inputs(cfg, rd, eng, c)
    return {"masks_per_case_family": len(masks), "case_family_masks": n, "numeric_scores": use_score}


def save_example_inputs(cfg, rd, eng, case):
    """Representative transformed inputs: slot 3 (and the original) of each case, per family, as 448x448 PNG."""
    from PIL import Image
    d = rd / "results" / "example_inputs"
    d.mkdir(exist_ok=True)
    fi = case["frame_indices"][3]
    frame = Image.open(ROOT / case["frames_dir"] / f"{fi:06d}.jpg").convert("RGB")
    st = eng.gis.dynamic_preprocess([frame], grid_size=1)
    st.save(d / f"{case['case_id']}_slot3_original.png")
    for fam in cfg["stage_b"]["families"]:
        al.apply_family(st, al.family_spec(cfg, fam)).save(d / f"{case['case_id']}_slot3_{fam}.png")


# ---------------------------------------------------------------- stage C

def effects_from_records(store, case_id, family):
    """Delta(G) = S0 - S(G) for the case's single, adjacent-pair and adjacent-triple masks (successful records only)."""
    recs = [r for r in store.stage_ids("perturbation_responses.jsonl").values()
            if r["case_id"] == case_id and r["target"] == "label_only_score" and r["stage"] == "B"
            and r["family"] == family and r["cache_status"] in ("fresh", "cached")]
    scale = store.cfg["stage_c"].get("effect_scale", "S")
    S0 = (store.done("baseline_responses.jsonl", f"A2|label_only_score|{case_id}|none||r0") or {}).get(scale)
    eff = {}
    for r in recs:
        if finite(r.get(scale)) and finite(S0) and r["mask_type"] in ("single", "adjacent_pair", "adjacent_triple"):
            eff[tuple(r["mask"])] = S0 - r[scale]
    return S0, eff


def stage_c(cfg, rd, man, eng, store):
    cases = load_cases(rd)
    diag = json.loads((rd / "results" / "scorer_diagnostics.json").read_text(encoding="utf-8"))
    if diag["status"] == "blocked":
        return {"skipped": "numeric scorer blocked; binary-only ranking analysis is done by analyze_attr.py"}
    fams = cfg["stage_b"]["families"]
    other = {fams[0]: fams[1], fams[1]: fams[0]}
    sizes = tuple(cfg["stage_c"]["group_sizes"])
    budgets = cfg["stage_c"]["budgets"]
    need = {m for _, m in al.primary_masks() if len(m) in (1,) + sizes}
    n_req, not_rankable = 0, []
    for cid in cfg["primary_cases"]:
        c = cases[cid]
        rankings = {}
        for src in fams:
            S0, eff = effects_from_records(store, cid, src)
            direction = 1 if scorer_class(S0) in (1, None) else -1
            missing = sorted(need - set(eff))
            for name, vals in (("singleton", al.singleton_scores(eff)),
                               ("group_size_balanced", al.group_scores_size_balanced(eff, sizes))):
                order, ties = al.rank_slots(vals, direction)
                st = al.ranking_status(vals, budgets, direction)
                if missing:
                    st = {"status": "incomplete", "missing_masks": [list(m) for m in missing]}
                rankings[f"{name}|{src}"] = {"order": order, "ties": ties, "direction": direction, "source_family": src,
                                             "eval_families": [other[src]], "values": vals, **st}
        for seed in cfg["stage_c"]["random_seeds"]:
            rankings[f"random_seed{seed}"] = {"order": al.random_ranking(seed), "ties": [], "direction": None,
                                              "source_family": None, "eval_families": fams, "status": "random_control"}
        for name, rk in rankings.items():
            if rk["status"] not in ("complete", "random_control"):
                not_rankable.append(f"{cid}:{name}:{rk['status']}")
                continue
            for fam in rk["eval_families"]:
                for b in budgets:
                    for mode, m in al.evaluation_masks(rk["order"], b).items():
                        meta = {"mask_type": f"eval_{mode}", "ranking": name, "budget": b, "mode": mode,
                                "ranking_order": rk["order"], "ranking_ties": rk["ties"],
                                "ranking_tie_spans_cut": rk.get(f"tie_spans_cut_at_{b}"),
                                "ranking_source_family": rk["source_family"], "ranking_direction": rk["direction"],
                                "request_suffix": f"{name}|b{b}|{mode}"}
                        run_request(store, eng, "perturbation_responses.jsonl", "C", "original_generation", c, m, fam, meta)
                        run_request(store, eng, "perturbation_responses.jsonl", "C", "label_only_score", c, m, fam, meta)
                        n_req += 2
        al.write_json(rd / "results" / f"rankings_{cid}.json", rankings)
    return {"evaluation_requests": n_req, "not_rankable": not_rankable}


# ---------------------------------------------------------------- main

class Terminated(Exception):
    pass


def _on_sigterm(signum, frame):  # launch.sh --time-limit sends TERM: record the stage as incomplete
    raise Terminated(f"received signal {signum}")


def gpu_seconds_prior(man):
    """Wall time of earlier GPU invocations of this run (A/B/C/D), from their recorded start/finish."""
    tot = 0.0
    for inv in man["invocations"]:
        if inv.get("uses_gpu") and inv.get("started_at") and inv.get("finished_at"):
            tot += (datetime.fromisoformat(inv["finished_at"]) - datetime.fromisoformat(inv["started_at"])).total_seconds()
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--amend-config-before-model-calls", metavar="REASON",
                    help="replace the frozen config (archived) only while no model call exists; REASON is recorded")
    args = ap.parse_args()
    cfg, rd, man = open_run(args)
    stages = args.stages.split(",")
    uses_gpu = any(s in ("A", "B", "C") for s in stages)
    code = al.code_sha256()
    if uses_gpu:
        frozen = man.setdefault("code_sha256_frozen_at_first_model_invocation", code)
        if frozen != code:
            sys.exit(f"experiment code changed since the first model invocation of this run ({frozen} -> {code}); "
                     "results would mix code versions: use a new run directory")
    inv = {"started_at": now(), "argv": sys.argv, "stages": stages, "uses_gpu": uses_gpu, "code_sha256": code,
           "job_dir": os.environ.get("JOB_DIR"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    prior = gpu_seconds_prior(man)
    man["invocations"].append(inv)
    save_manifest(rd, man)
    signal.signal(signal.SIGTERM, _on_sigterm)
    rx.environment_report(rd)
    eng = store = None
    if uses_gpu:
        if not (rd / "cases.json").exists():
            sys.exit("run the prep stage first")
        store = Store(cfg, rd, prior)
        inv["gpu_seconds_prior_invocations"] = round(prior, 1)
        eng = make_engine(cfg, rd)
        inv["engine"] = {"model_load_s": round(eng.load_s, 1), "attention": eng.attention}
        prompts = {v: eng.effective_question(v) for v in ("original", "nosound", "label_only")}
        pdir = rd / "results" / "prompts"
        pdir.mkdir(exist_ok=True)
        for v, q in prompts.items():
            (pdir / f"{v}_question.txt").write_text(q, encoding="utf-8")
    for st in stages:
        t0 = time.time()
        err0 = store.errors_this_invocation if store else 0
        print(f"== stage {st} {now()}", flush=True)
        fn = {"prep": stage_prep, "A": stage_a, "B": stage_b, "C": stage_c}[st]
        try:
            res = fn(cfg, rd, man) if st == "prep" else fn(cfg, rd, man, eng, store)
            n_err = (store.errors_this_invocation - err0) if store else 0
            status = "ok" if n_err == 0 else "ok_with_errors"
            if n_err:
                res["errors_or_skips_this_stage"] = n_err
        except SystemExit as e:
            res, status = {"error": str(e)}, "failed"
        except Terminated as e:
            res, status = {"error": str(e)}, "incomplete_terminated"
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res, status = {"error": repr(e)}, "failed"
        rec = {"status": status, "finished_at": now(), "wall_s": round(time.time() - t0, 1), **res}
        if eng:
            rec["engine_counts_cumulative_this_invocation"] = dict(eng.counts)
            rec["unique_transformed_sequences_total"] = len(store.transformed_sequences())
            rec["gpu_seconds_used_total"] = round(store.gpu_seconds_used(), 1)
        man["stages"].setdefault(st, []).append(rec)
        save_manifest(rd, man)
        print(f"== stage {st}: {status} {json.dumps(res, ensure_ascii=True)[:600]}", flush=True)
        if status not in ("ok", "ok_with_errors"):
            inv["finished_at"] = now()
            save_manifest(rd, man)
            sys.exit(1)
    inv["finished_at"] = now()  # peak GPU memory: per-call peaks in the records, device peak in the launcher samples
    save_manifest(rd, man)


if __name__ == "__main__":
    main()

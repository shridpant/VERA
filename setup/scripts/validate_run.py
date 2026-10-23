#!/usr/bin/env python3
"""Validate one VERA run directory. Exits 1 if any check fails, 0 otherwise.

Usage (on oz, with the project environment's python):
  python setup/scripts/validate_run.py --run-dir experiments/experiment01/runs/run_001 [--require-report]

Writes <run-dir>/results/validation.json. Check C13 re-runs the UNMODIFIED compute_auc.py (taken from the base
commit with `git show`) in a temporary sandbox holding only this video, and requires identical frame scores.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import yaml  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import vera_pipeline as vp  # noqa: E402

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", SCRIPTS.parent.parent)).resolve()

REQUIRED = ["run_manifest.json", "environment.txt", "resolved_config.yaml", "results/frames_identity.json",
            "results/eval_manifest_one_video.json", "results/benchmark_annotation.json",
            "results/raw_responses.jsonl", "results/segment_scores.json", "results/segment_scores_vera_format.json",
            "results/model_input_rendered.txt", "results/scene_refined_scores.json", "results/scene_refined_scores.npy",
            "results/temporal_refined_scores.json", "results/temporal_refined_scores.npy", "results/frame_scores.npy",
            "results/frame_scores.csv", "results/diagnostics.json"]

PARSER_CASES = [  # (text, expected label or None)
    ("Answers to Prompt Questions: 1. No. 2. No.\nOutput:\n0", 0),
    ("Answers to Prompt Questions: 1. Yes, a car crashes.\nOutput:\n1", 1),
    ("Answers to Prompt Questions: ...\nOutput: 1", 1),
    ("Answers to Prompt Questions: ...\n**Output:**\n[1]", 1),
    ("```\nAnswers to Prompt Questions: ...\nOutput:\n0\n```", 0),
    ("Answers to Prompt Questions: ...\nOutput:\n1.", 1),
    ("Answers to Prompt Questions: 1. No.", None),                       # no output field
    ("Answers ...\nOutput:\n", None),                                     # empty field
    ("Answers ...\nOutput:\n0 or 1", None),                               # not a single label
    ("Answers ...\nOutput:\nYes, there is an anomaly", None),             # words instead of the integer
    ("Answers ...\nOutput:\n10", None),                                   # not 0/1
    ("Output: 0\nAnswers ...\nOutput: 1", None),                          # two output fields
    ("", None),
]


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, cid, desc, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # a crashing check is a failed check
            ok, detail = False, f"exception: {e!r}"
        self.rows.append({"id": cid, "check": desc, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {cid} {desc}: {detail}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--require-report", action="store_true")
    a = ap.parse_args()
    rd = Path(a.run_dir) if os.path.isabs(a.run_dir) else (ROOT / a.run_dir)
    res = rd / "results"
    C = Checks()

    missing = [f for f in REQUIRED if not (rd / f).exists()]
    C.check("C01", "required artifacts exist", lambda: (not missing, f"missing {missing}" if missing else f"{len(REQUIRED)} files"))
    if missing:
        return finish(C, res)

    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text(encoding="utf-8"))
    man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
    ident = json.loads((res / "frames_identity.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (res / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    seg = json.loads((res / "segment_scores.json").read_text(encoding="utf-8"))
    segv = json.loads((res / "segment_scores_vera_format.json").read_text(encoding="utf-8"))
    scene = json.loads((res / "scene_refined_scores.json").read_text(encoding="utf-8"))
    temporal = json.loads((res / "temporal_refined_scores.json").read_text(encoding="utf-8"))
    diag = json.loads((res / "diagnostics.json").read_text(encoding="utf-8"))
    n = ident["decoded_frame_count"]
    s = cfg["sampling"]

    def c02():
        bad = []
        for p in sorted(res.glob("*.json")) + [rd / "run_manifest.json"]:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if json.loads(json.dumps(obj)) != obj:
                bad.append(p.name)
        for r in recs:
            if json.loads(json.dumps(r)) != r:
                bad.append("raw_responses.jsonl")
                break
        return not bad, f"round-trip failures {bad}" if bad else "all JSON round-trips"
    C.check("C02", "JSON artifacts parse and round-trip", c02)

    def c03():
        last = {st: man["stages"].get(st, [{}])[-1].get("status") for st in ("frames", "vlm", "refine")}
        one = man["video"]["id"] == cfg["video"]["id"] and len({r["video_id"] for r in recs}) == 1
        return all(v == "ok" for v in last.values()) and one, f"latest stage status {last}; single video {one}"
    C.check("C03", "manifest: all three stages ok, exactly one video", c03)

    def c04():
        em = json.loads((res / "eval_manifest_one_video.json").read_text(encoding="utf-8"))
        ok = len(em) == 1 and em[0]["video"] == cfg["video"]["manifest_entry"] and set(em[0]) == {"video", "length"}
        return ok, f"{len(em)} entry; keys {sorted(em[0])}; video {em[0]['video']}"
    C.check("C04", "Step 1 manifest is the one requested video, without benchmark labels", c04)

    def c05():
        video = ROOT / cfg["video"]["path"]
        vsha = sha256_file(video)
        fdir = ROOT / ident["frames_dir"]
        names = sorted(os.listdir(fdir))
        lines = "".join(f"{sha256_file(fdir / x)}  {x}\n" for x in names)
        digest_ok = hashlib.sha256(lines.encode()).hexdigest() == ident["frames_digest_sha256"]
        ok = (vsha == cfg["video"]["sha256"] == ident["source_sha256"] and n == cfg["video"]["expected_length"]
              and names == [f"{i:06d}.jpg" for i in range(n)] and digest_ok)
        return ok, (f"video sha {vsha[:12]}; decoded {n} vs manifest {cfg['video']['expected_length']}; "
                    f"{len(names)} frame files; digest match {digest_ok}")
    C.check("C05", "source video checksum, decoding and frame cache identity", c05)

    expected_anchors = vp.anchors(n, s["sampling_rate"])

    def c06():
        got = [r["anchor_frame"] for r in recs]
        ok = got == expected_anchors and len(got) == cfg["video"]["expected_anchor_count"] and \
            [r["window_index"] for r in recs] == list(range(len(recs)))
        return ok, f"{len(got)} records; expected {len(expected_anchors)} (config {cfg['video']['expected_anchor_count']}); missing {sorted(set(expected_anchors) - set(got))[:5]}"
    C.check("C06", "one record per anchor, range(0, frames, 16), none missing or duplicated", c06)

    src = ROOT / cfg["source"]["checkout"]
    instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
    qs = "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"])
    flag = "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?'"
    q = instruct.replace("$Data", "".join(f"Frame{i + 1}: <image>\n" for i in range(s["snippet_len"])))
    expected_q = q[:q.find(flag)] + qs + q[q.find(flag):]

    def c07():
        problems = []
        for r in recs:
            st, en, idx = vp.window_for_anchor(r["anchor_frame"], n, s["fps_for_windows"], s["t_window_s"], s["snippet_len"])
            fi = r["frame_indices"]
            if fi != idx or len(fi) != 8 or any(not isinstance(i, int) or i < 0 or i >= n for i in fi) \
                    or fi != sorted(fi) or (r["window_start"], r["window_end"]) != (st, en) or not (st <= fi[0] and fi[-1] < en):
                problems.append(f"{r['anchor_frame']}: window/frames")
            if r["prompt_question"] != expected_q:
                problems.append(f"{r['anchor_frame']}: prompt")
            if not r["raw_response"].strip() or r["parse_status"] != "ok" or r["label"] not in (0, 1) or \
                    vp.strict_parse_label(r["raw_response"]) != (r["label"], "ok"):
                problems.append(f"{r['anchor_frame']}: label {r['label']} {r['parse_status']}")
            # InternVL's batch_chat adds eos_token_id (<|im_end|> = 92542) to the dict in place, as upstream does.
            g = dict(r["generation_config"])
            if g.pop("eos_token_id", None) != 92542 or g != cfg["generation"] or r["audio_supplied"] is not False:
                problems.append(f"{r['anchor_frame']}: generation/audio")
        n_inputs = len({r["model_input_sha256"] for r in recs})
        if n_inputs != 1 or not all(r["model_input_rendered_matches_file"] for r in recs):
            problems.append(f"{n_inputs} distinct model token inputs")
        return not problems, f"{len(problems)} problems {problems[:6]}"
    C.check("C07", "every window: 8 in-range frames from the upstream formula, exact prompt, raw text, strict label", c07)

    def c08():
        ann = json.loads((res / "benchmark_annotation.json").read_text(encoding="utf-8"))
        rendered = (res / "model_input_rendered.txt").read_text(encoding="utf-8")
        cat = re.sub(r"\d+.*", "", cfg["video"]["id"])
        bad_tokens = [cfg["video"]["id"], cat] + [str(x) for x in ann["temporal_label"] if x != -1]
        hits = [t for t in bad_tokens if re.search(rf"(?<![0-9A-Za-z]){re.escape(t)}(?![0-9])", expected_q + rendered)]
        return not hits, f"searched prompt and rendered model input for {bad_tokens}: hits {hits}"
    C.check("C08", "no video name, category or annotation frame in the prompt", c08)

    def c09():
        bad = [(t, e, vp.strict_parse_label(t)) for t, e in PARSER_CASES if vp.strict_parse_label(t)[0] != e]
        return not bad, f"{len(PARSER_CASES) - len(bad)}/{len(PARSER_CASES)} parser cases as expected; failures {bad}"
    C.check("C09", "strict parser on representative valid and malformed outputs", c09)

    def c10():
        lab = {str(r["anchor_frame"]): r["label"] for r in recs}
        ok = seg["n_unparsed"] == 0 and list(segv) == [str(x) for x in expected_anchors] and \
            all(segv[k]["score"] == lab[k] == seg["anchors"][k]["score"] for k in segv)
        return ok, f"{sum(lab.values())} of {len(lab)} anchors labelled 1; unparsed {seg['n_unparsed']}"
    C.check("C10", "segment score files agree with the raw records", c10)

    feats = np.load(ROOT / cfg["features"]["path"])

    def c11():
        fsha = sha256_file(ROOT / cfg["features"]["path"])
        ok = fsha == cfg["features"]["sha256"] == scene["features"]["sha256"] and feats.shape[0] == len(expected_anchors) \
            and np.isfinite(feats).all() and scene["anchors"] == expected_anchors
        return ok, f"shape {feats.shape} {feats.dtype}; rows vs anchors {feats.shape[0]}/{len(expected_anchors)}; sha {fsha[:12]}"
    C.check("C11", "feature rows align one-to-one with scored anchors", c11)

    fs = np.load(res / "frame_scores.npy")

    def c12():
        sc, tm = np.load(res / "scene_refined_scores.npy"), np.load(res / "temporal_refined_scores.npy")
        raw = np.array(scene["raw_labels"])
        csv = np.loadtxt(res / "frame_scores.csv", delimiter=",", skiprows=1)
        ok = (np.isfinite(sc).all() and np.isfinite(tm).all() and np.isfinite(fs).all()
              and len(sc) == len(tm) == len(expected_anchors) and len(fs) == n and set(np.unique(raw)) <= {0.0, 1.0}
              and sc.min() >= 0 and sc.max() <= 1 + 1e-12 and tm.min() >= 0 and tm.max() <= 1 + 1e-12
              and fs.min() >= 0 and fs.max() <= 1 + 1e-12 and csv.shape[0] == n
              and np.allclose(csv[:, 2], fs, atol=1e-7) and np.array_equal(csv[:, 0], np.arange(n))
              and np.allclose(np.array(temporal["temporal_smoothed"]), tm))
        return ok, (f"scene [{sc.min():.3f},{sc.max():.3f}] temporal [{tm.min():.3f},{tm.max():.3f}] "
                    f"frames {len(fs)} [{fs.min():.4f},{fs.max():.4f}]")
    C.check("C12", "refined arrays numeric, finite, in range, frame scores match decoded frame count", c12)

    def c13():
        base = cfg["source"]["base_commit"]
        upstream_src = subprocess.run(["git", "show", f"{base}:compute_auc.py"], cwd=src, check=True,
                                      capture_output=True, text=True).stdout
        ann = json.loads((res / "benchmark_annotation.json").read_text(encoding="utf-8"))
        key = cfg["video"]["id"]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            td = Path(td)
            (td / "Data" / "segment_level_score").mkdir(parents=True)
            (td / "Data" / "vision_features").mkdir()
            (td / "compute_auc_upstream.py").write_text(upstream_src, encoding="utf-8")
            (td / "Data" / "UCF_Eval.json").write_text(json.dumps([{
                "video": ann["video"], "temporal_label": ann["temporal_label"], "length": ann["length"],
                "QA": [{"q": "Is there any anomaly in the video?", "a": ann["video_level_answer"]}]}]), encoding="utf-8")
            (td / "Data" / "segment_level_score" / f"{key}.json").write_text((res / "segment_scores_vera_format.json").read_text(encoding="utf-8"))
            np.save(td / "Data" / "vision_features" / f"{key}.npy", feats)
            code = ("import runpy, numpy as np; g = runpy.run_path('compute_auc_upstream.py'); "
                    "np.save('upstream_all_predict_score.npy', np.array(g['all_predict_score'])); "
                    "open('upstream_auc.txt', 'w').write(repr(g['roc_auc']))")
            p = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=td, capture_output=True, text=True)
            if p.returncode != 0:
                return False, f"upstream script failed: {p.stderr[-400:]}"
            up = np.load(td / "upstream_all_predict_score.npy")
            up_auc = float((td / "upstream_auc.txt").read_text(encoding="utf-8"))
        ours_auc = diag.get("frame_level_roc_auc_this_video")
        auc_ok = (ours_auc is None and math.isnan(up_auc)) or (ours_auc is not None and abs(up_auc - ours_auc) < 1e-12)
        return np.array_equal(up, fs) and auc_ok, (f"unmodified compute_auc.py @ {base[:8]}: identical frame scores "
                                                   f"{np.array_equal(up, fs)} ({len(up)} values); its printed AUC {up_auc:.6f} vs ours {ours_auc}")
    C.check("C13", "frame scores identical to the unmodified upstream compute_auc.py", c13)

    def c14():
        v = {ident["source_sha256"], seg["video_sha256"], scene["video_sha256"], temporal["video_sha256"],
             *{r["video_sha256"] for r in recs}, man["video"]["sha256"]}
        return len(v) == 1, f"{len(v)} distinct video checksums across artifacts"
    C.check("C14", "all artifacts reference the same source video checksum", c14)

    def c15():
        ref = diag.get("reference_official_initial_scores", {})
        ok = "single video only" in diag["scope"] and (not ref or ref["label"].startswith("REFERENCE ONLY"))
        return ok, f"scope: {diag['scope'][:60]}...; reference block labelled {bool(ref)}"
    C.check("C15", "diagnostics scoped to this video; reference outputs labelled", c15)

    if a.require_report:
        def c16():
            rep = (rd / "REPORT.md").read_text(encoding="utf-8")
            quoted = any(len(r["raw_response"]) >= 60 and r["raw_response"][:60] in rep for r in recs)
            ok = "run_experiment.py" in rep and quoted and "validate_run.py" in rep
            return ok, f"reproduction command present {'run_experiment.py' in rep}; verbatim raw excerpt present {quoted}"
        C.check("C16", "REPORT.md has the rerun command and verbatim model text", c16)
    return finish(C, res)


def finish(C, res):
    n_fail = sum(r["status"] == "FAIL" for r in C.rows)
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "validation.json", "w", encoding="utf-8") as f:
        json.dump({"passed": len(C.rows) - n_fail, "failed": n_fail, "checks": C.rows}, f, indent=2)
    print(f"{len(C.rows) - n_fail}/{len(C.rows)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GPU-free acceptance tests for the reusable VERA pieces. Exit code 1 on any failure.

Usage (on oz): setup/environment/env/bin/python setup/scripts/test_vera_pipeline.py
Needs the pinned VERA checkout (for the unmodified compute_auc.py) and numpy/scikit-learn/scipy.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import vera_pipeline as vp  # noqa: E402
from validate_run import PARSER_CASES  # noqa: E402

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", SCRIPTS.parent.parent)).resolve()
SRC = ROOT / "setup" / "source" / "VERA"
BASE = "15b8bcb8574a977c229c577f50bfe6f06d07106e"
failures = []


def test(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


# parser
for text, want in PARSER_CASES:
    got = vp.strict_parse_label(text)
    test(f"parser {text[-30:]!r}", got[0] == want and ((want is None) == (got[1] != "ok")), str(got))
test("upstream root parser maps text without an output field to 1 (why it is not used for labels)",
     vp.upstream_root_parse("I cannot tell.") == 1)
test("upstream Inference-variant parser maps a trailing period to 0", vp.upstream_inference_variant_parse("Output:\n1.") == 0)

# anchors and windows
test("primary 2580 frames -> 162 anchors", len(vp.anchors(2580)) == 162)
test("fallback 1181 frames -> 74 anchors", len(vp.anchors(1181)) == 74)
for n in (2580, 1181, 300, 17):
    bad = []
    for a in vp.anchors(n):
        st, en, idx = vp.window_for_anchor(a, n)
        if not (len(idx) == 8 and idx == sorted(idx) and st <= idx[0] and idx[-1] < en <= n and st >= 0
                and st == max(a - 150, 0) and en == min(a + 150, n)):
            bad.append(a)
    test(f"windows for {n} frames: 8 sorted in-range frames within [anchor-150, anchor+150)", not bad, f"bad {bad[:5]}")


def run_upstream(scores, feats, gt_label, length, key="SyntheticVideo001_x264"):
    upstream = subprocess.run(["git", "show", f"{BASE}:compute_auc.py"], cwd=SRC, check=True,
                              capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "Data" / "segment_level_score").mkdir(parents=True)
        (td / "Data" / "vision_features").mkdir()
        (td / "u.py").write_text(upstream, encoding="utf-8")
        (td / "Data" / "UCF_Eval.json").write_text(json.dumps([{"video": f"frames/X/{key}.mp4", "temporal_label": gt_label,
                                                                 "length": length, "QA": []}]), encoding="utf-8")
        (td / "Data" / "segment_level_score" / f"{key}.json").write_text(json.dumps(scores), encoding="utf-8")
        np.save(td / "Data" / "vision_features" / f"{key}.npy", feats)
        code = ("import runpy, numpy as np; g = runpy.run_path('u.py'); "
                "np.save('o.npy', np.array(g['all_predict_score']))")
        p = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=td, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(p.stderr[-500:])
        return np.load(td / "o.npy")


rng = np.random.default_rng(0)
for n in (2580, 1181, 700):
    anc = vp.anchors(n)
    scores = {}
    for a in anc:
        st, en, _ = vp.window_for_anchor(a, n)
        scores[str(a)] = {"start": st, "end": en, "score": int(rng.random() < 0.3)}
    feats = rng.normal(size=(len(anc), 1024))
    ours = vp.vera_refine(scores, feats)
    up = run_upstream(scores, feats, [n // 3, n // 2, -1, -1], n)
    test(f"vera_refine == unmodified compute_auc.py on synthetic {n}-frame video", np.array_equal(ours["frame_final"], up),
         f"{len(up)} frame scores")
    test(f"frame scores length equals frame count ({n})", len(ours["frame_final"]) == n)

# edge cases inspected before execution (documented upstream behaviour, preserved, not fixed)
anc = vp.anchors(150)  # 10 segments: top_n = int(1.5) = 1
sc = {str(a): {"start": vp.window_for_anchor(a, 150)[0], "end": vp.window_for_anchor(a, 150)[1], "score": 1} for a in anc}
o = vp.vera_refine(sc, rng.normal(size=(len(anc), 8)))
test("short video (10 segments): temporal smoothing output is 15 long (np.convolve 'same' keeps the longer length), "
     "so per-segment smoothed values are misaligned upstream", len(o["temporal_smoothed"]) == 15, f"top_n={o['top_n']}")
anc = vp.anchors(96)  # 6 segments: top_n = 0 -> argsort[:, -0:] keeps ALL segments as neighbours
sc = {str(a): {"start": vp.window_for_anchor(a, 96)[0], "end": vp.window_for_anchor(a, 96)[1], "score": 0} for a in anc}
o = vp.vera_refine(sc, rng.normal(size=(len(anc), 8)))
test("very short video (6 segments): top_n == 0 makes every segment a neighbour upstream",
     o["top_n"] == 0 and o["neighbor_indices"].shape == (6, 6))
try:
    vp.vera_refine({"0": {"start": 0, "end": 150, "score": 1}, "16": {"start": 0, "end": 166, "score": 0}},
                   rng.normal(size=(3, 8)))
    test("feature/anchor row mismatch raises", False)
except ValueError:
    test("feature/anchor row mismatch raises", True)

print(f"{'ALL PASS' if not failures else f'{len(failures)} FAILED: {failures}'}")
sys.exit(1 if failures else 0)

#!/usr/bin/env python3
"""Synthetic tests for code/validate3b.py (standard library only).

    python3 -I -B experiments/experiment03B/code/tests/test_validate3b.py

Every fixture builds a synthetic run folder that follows RECORDS.md (inputs.json, frozen/config.json,
frozen/manifest.json, requests/*.jsonl) in a temporary directory, then runs the validator's functions
on it. The builder below computes its own request schedule (fixed order, adaptive order) and does not
call the validator. Passing these tests establishes implementation correctness, not attribution truth.
"""

import csv
import hashlib
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import unittest
from fractions import Fraction
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import validate3b as v  # noqa: E402

SLOTS = tuple(range(8))
PAIRS = tuple(combinations(SLOTS, 2))
SCORE, ORIG, NOSOUND = "label_only_score", "original_generation", "nosound_generation"
EXP02_RUN = os.path.normpath(os.path.join(HERE, "..", "..", "..", "experiment02", "runs", "run_002"))
VIDEO_LENGTH = 100000


# ----------------------------------------------------------------------------------------- builder

def _h(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bf16(x):
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


def frames_for(anchor):
    """Full-length VERA window: start a-150, step 37.5 (exact in binary floating point)."""
    return [anchor - 150 + int(37.5 * i) for i in range(8)]


class Spec:
    """One synthetic input.

    score(family, mask) -> S_hi;  gen(target, family, mask) -> list of labels per repeat
    (None = unparsed); the first parsed entry ends the repeats.
    """

    def __init__(self, video, anchor, score, gen, category="RoadAccidents", strata=("event",),
                 metadata_flags=(), status="ok", fail=()):
        self.video, self.anchor, self.score, self.gen = video, anchor, score, gen
        self.category, self.strata = category, list(strata)
        self.metadata_flags, self.status = list(metadata_flags), status
        self.fail = set(fail)  # (target, family, mask) requests that error on all three attempts

    @property
    def uid(self):
        return "%s@%d" % (self.video, self.anchor)


def slot_hashes(uid, family, mask):
    return [_h(uid, s, family if s in mask else "orig") for s in SLOTS]


def score_line(uid, phase, purpose, family, mask, value, status="fresh"):
    shi = _f32(value)
    l1 = _bf16(shi)
    hs = slot_hashes(uid, family, mask)
    seq = _h("seq", *hs)
    return {"request_id": "%s|%s|%s|%s|r0" % (uid, SCORE, family, ",".join(map(str, mask))),
            "window_uid": uid, "phase": phase, "purpose": purpose, "target": SCORE, "family": family,
            "mask": list(mask), "repeat": 0, "cache_status": status, "result_key": _h("rk", SCORE, seq, 0),
            "sequence_key": seq, "attempt": 1, "slot_tensor_sha256": hs, "slot_identity_ok": True,
            "S": l1 - 0.0, "S_hi": shi, "logit_0": 0.0, "logit_1": l1, "hi_logit_0": 0.0, "hi_logit_1": shi,
            "S_resolution": 0.25, "S_hi_rounds_to_model_logits": True, "candidate_mass": 0.9, "finite": True}


def gen_line(uid, phase, purpose, target, family, mask, repeat, label):
    hs = slot_hashes(uid, family, mask)
    seq = _h("seq", *hs)
    raw = ("The frames show ... Output: %d" % label) if label is not None else "The frames show ... (no label)"
    return {"request_id": "%s|%s|%s|%s|r%d" % (uid, target, family, ",".join(map(str, mask)), repeat),
            "window_uid": uid, "phase": phase, "purpose": purpose, "target": target, "family": family,
            "mask": list(mask), "repeat": repeat, "cache_status": "fresh",
            "result_key": _h("rk", target, seq, repeat), "sequence_key": seq, "attempt": 1,
            "slot_tensor_sha256": hs, "slot_identity_ok": True, "raw_response": raw, "label": label,
            "parse_status": "ok" if label is not None else "no_output_line"}


def error_lines(uid, phase, purpose, target, family, mask):
    return [{"request_id": "%s|%s|%s|%s|r0" % (uid, target, family, ",".join(map(str, mask))),
             "window_uid": uid, "phase": phase, "purpose": purpose, "target": target, "family": family,
             "mask": list(mask), "repeat": 0, "cache_status": "error", "attempt": a,
             "error": "RuntimeError: CUDA error (synthetic)"} for a in (1, 2, 3)]


def build_run(root, specs, seed=0, B=200, name="run"):
    """Write a RECORDS-conformant run folder for the specs, following the PROTOCOL 6 schedule."""
    run = os.path.join(root, name)
    os.makedirs(os.path.join(run, "frozen"))
    os.makedirs(os.path.join(run, "requests"))
    lines = {"baseline": [], "fixed": [], "adaptive": [], "evaluation": []}
    inputs, videos = [], {}
    for sp in specs:
        uid = sp.uid
        fr = frames_for(sp.anchor)
        videos[sp.video] = {"video_id": sp.video, "category": sp.category, "length": VIDEO_LENGTH}
        inputs.append({"window_uid": uid, "video_id": sp.video, "category": sp.category, "anchor": sp.anchor,
                       "frame_indices": fr, "frame_sha256": [_h(sp.video, f) for f in fr],
                       "video_sha256": _h(sp.video), "strata": sp.strata, "window_record_ids": [uid + "#w"],
                       "development": sp.video in ("RoadAccidents127_x264", "Normal_Videos_018_x264"),
                       "metadata_flags": sp.metadata_flags, "annotated_slots": [], "clipped": False,
                       "status": sp.status})
        logged = set()

        def score_req(phase, purpose, family, mask):
            if (SCORE, family, mask) in logged:
                return
            logged.add((SCORE, family, mask))
            if (SCORE, family, mask) in sp.fail:
                lines[phase].extend(error_lines(uid, phase, purpose, SCORE, family, mask))
                return
            lines[phase].append(score_line(uid, phase, purpose, family, mask, sp.score(family, mask)))

        def gen_req(phase, purpose, target, family, mask):
            if (target, family, mask) in logged:
                return None
            logged.add((target, family, mask))
            if (target, family, mask) in sp.fail:
                lines[phase].extend(error_lines(uid, phase, purpose, target, family, mask))
                return None
            first = None
            for rep, lab in enumerate(sp.gen(target, family, mask)[:3]):
                lines[phase].append(gen_line(uid, phase, purpose, target, family, mask, rep, lab))
                if lab is not None:
                    first = lab
                    break
            return first

        score_req("baseline", "baseline", "none", ())
        L0 = gen_req("baseline", "baseline", ORIG, "none", ())
        gen_req("baseline", "baseline", NOSOUND, "none", ())
        if L0 is None:
            continue
        q = 1 if L0 == 1 else -1
        for fam in ("blur", "grey"):
            for i in SLOTS:
                score_req("fixed", "single", fam, (i,))
            for p in PAIRS:
                score_req("fixed", "pair", fam, p)
        for p in PAIRS:
            gen_req("fixed", "pair", ORIG, "grey", p)
        base = _f32(sp.score("none", ()))
        eff = [q * (base - _f32(sp.score("blur", (i,)))) for i in SLOTS]
        order = sorted(SLOTS, key=lambda i: (-eff[i], i))
        for k in (1, 3, 4):
            m = tuple(sorted(order[:k]))
            score_req("evaluation", "fixed_prefix", "grey", m)
            gen_req("evaluation", "fixed_prefix", ORIG, "grey", m)
        aorder = []
        for step in range(1, 5):
            st = base if step == 1 else _f32(sp.score("blur", tuple(sorted(aorder))))
            best = None
            for j in SLOTS:
                if j in aorder:
                    continue
                m = tuple(sorted(aorder + [j]))
                s = _f32(sp.score("blur", m))
                if len(m) >= 3:
                    score_req("adaptive", "adaptive_candidate", "blur", m)
                e = q * (st - s)
                if best is None or e > best[0]:
                    best = (e, j)
            aorder.append(best[1])
        for k in (3, 4):
            m = tuple(sorted(aorder[:k]))
            score_req("evaluation", "adaptive_prefix", "grey", m)
            gen_req("evaluation", "adaptive_prefix", ORIG, "grey", m)
    with open(os.path.join(run, "inputs.json"), "w") as fh:
        json.dump({"inputs": inputs}, fh)
    with open(os.path.join(run, "frozen", "config.json"), "w") as fh:
        json.dump({"bootstrap": {"seed": seed, "B": B}, "protocol_version": "1.0"}, fh)
    with open(os.path.join(run, "frozen", "manifest.json"), "w") as fh:
        json.dump({"version": "synthetic", "videos": list(videos.values()), "windows": []}, fh)
    for phase, ls in lines.items():
        with open(os.path.join(run, "requests", phase + ".jsonl"), "w") as fh:
            for ln in ls:
                fh.write(json.dumps(ln) + "\n")
    return run


def additive(base, weights):
    """S_hi(R) = base - sum_{i in R} w_i, same on every family."""
    return lambda fam, mask: base - sum(weights[i] for i in mask)


def gen_const(L0, grey_label=1, overrides=None):
    """Baseline label L0 (and nosound), grey generations grey_label unless overridden per mask."""
    overrides = overrides or {}

    def g(target, family, mask):
        if family == "none":
            return [L0] if not isinstance(L0, list) else L0
        if mask in overrides:
            o = overrides[mask]
            return o if isinstance(o, list) else [o]
        return [grey_label]
    return g


def flip_spec(video, anchor, flips=(), missing=(), repeats=None, L0=1, category="RoadAccidents",
              strata=("event",), metadata_flags=()):
    """T_w = [0, 1] from blur weights 4, 3, 1, ...; grey pair labels 0 (flip) for `flips`, unparsed in all
    three repeats for `missing`, explicit per-repeat label lists in `repeats`, label 1 otherwise."""
    w = [4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ov = {p: 0 for p in flips}
    for p in missing:
        ov[p] = [None, None, None]
    for p, labs in (repeats or {}).items():
        ov[p] = labs
    return Spec(video, anchor, additive(10.0, w), gen_const(L0, 1, ov), category=category, strata=strata,
                metadata_flags=metadata_flags)


def recompute(run_dir):
    run = v.load_run(run_dir)
    auth = v.build_authoritative(run.lines)
    store = v.Store(auth)
    calcs = [v.compute_input(inp, store) for inp in run.inputs]
    return run, auth, store, {c["window_uid"]: c for c in calcs}, calcs


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "-".join(str(i) for i in x)
    if isinstance(x, float):
        return repr(x)
    return str(x)


def write_mirror_outputs(run_dir):
    """Analyzer-format outputs written from the validator's own recomputation (a perfect analyzer)."""
    run, auth, store, by_uid, calcs = recompute(run_dir)
    seed, B, _ = v.bootstrap_params(run.config)
    prim = v.compute_primary(calcs, seed, B)
    os.makedirs(os.path.join(run_dir, "derived"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)
    with open(os.path.join(run_dir, "derived", "rankings.jsonl"), "w") as fh:
        for c in calcs:
            r = v.rankings_record(c)
            if r is not None:
                fh.write(json.dumps(r) + "\n")
    with open(os.path.join(run_dir, "results", "primary.json"), "w") as fh:
        json.dump(prim, fh)
    with open(os.path.join(run_dir, "results", "inputs_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_uid", "video_id", "L0", "q", "T_w", "d_w", "F_top2", "F_random_mean",
                    "n_pair_labels", "e_w", "tau_b_blur_grey", "tau_b_native_vs_S_hi",
                    "overlap_top2_blur_grey", "overlap_top2_native"])
        for c in calcs:
            w.writerow([_fmt(x) for x in (c["window_uid"], c["video_id"], c["L0"], c["q"], c["T_w"], c["d_w"],
                                          c["F_top2"], c["F_random_mean"], c["n_pair_labels"], c["e_w"],
                                          c["tau_blur_grey"], c["tau_native_hi"], c["overlap_blur_grey"],
                                          c["overlap_native_hi"])])
    with open(os.path.join(run_dir, "results", "pairs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_uid", "family", "pair", "S_hi", "S", "D", "J", "label", "flip"])
        for c in calcs:
            for r in c["pairs"]:
                w.writerow([_fmt(x) for x in (c["window_uid"], r["family"], r["pair"], r["S_hi"], r["S"], r["D"],
                                              r["J"], r["label"], r["flip"])])
    with open(os.path.join(run_dir, "results", "trajectories.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_uid", "mode", "k", "mask", "label", "flip", "q_dS_hi", "q_dS", "new_model_calls"])
        for c in calcs:
            for r in c["trajectories"]:
                w.writerow([_fmt(x) for x in (c["window_uid"], r["mode"], r["k"], r["mask"], r["label"], r["flip"],
                                              r["q_dS_hi"], r["q_dS"], "")])
    return prim


def check_by_id(report, cid):
    for c in report["checks"]:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def rewrite_jsonl(path, fn):
    with open(path) as fh:
        recs = [json.loads(x) for x in fh.read().split("\n") if x.strip()]
    recs = fn(recs)
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------------------- fixtures

def interaction_score(fam, mask):
    a = 0 if 0 in mask else 1
    b = 0 if 1 in mask else 1
    c = 0 if 2 in mask else 1
    return float(10 * a + 5 * b + c + 5 * a * c)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="exp03b_validate_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAnalyticInteraction(Base):
    """F = 10a + 5b + c + 5ac on slots 0, 1, 2 (anomaly target, L0 = 1)."""

    def test_fixed_and_adaptive_orders(self):
        sp = Spec("Arson011_x264", 1600, interaction_score, gen_const(1, 1), category="Arson")
        run_dir = build_run(self.tmp, [sp])
        _, _, _, by_uid, _ = recompute(run_dir)
        c = by_uid[sp.uid]
        self.assertEqual(c["L0"], 1)
        self.assertEqual(c["q"], 1)
        self.assertEqual(c["baseline_S_hi"], 21.0)
        blur = c["rankings"]["blur"]
        self.assertEqual({k: blur["effects"][k] for k in ("0", "1", "2")}, {"0": 15.0, "1": 5.0, "2": 6.0})
        self.assertTrue(all(blur["effects"][str(i)] == 0.0 for i in range(3, 8)))
        self.assertEqual(blur["order"][:3], [0, 2, 1])
        self.assertEqual(blur["order"], [0, 2, 1, 3, 4, 5, 6, 7])
        self.assertEqual(blur["ties"], [[3, 4, 5, 6, 7]])
        self.assertEqual(blur["tie_spans_cut"], {"1": False, "2": False, "3": False, "4": True})
        self.assertEqual(blur["top2_margin"], 1.0)
        ad = c["adaptive"]
        self.assertEqual(ad["order"][:3], [0, 1, 2])
        self.assertEqual(ad["order"], [0, 1, 2, 3])
        s1, s2, s3, s4 = ad["states"]
        self.assertEqual((s1["masked"], s1["state_score"], s1["selected"]), ([], 21.0, 0))
        self.assertEqual((s2["masked"], s2["state_score"], s2["selected"]), ([0], 6.0, 1))
        eff2 = {cd["slot"]: cd["effect"] for cd in s2["candidates"]}
        self.assertEqual(sorted(eff2), [1, 2, 3, 4, 5, 6, 7])  # every remaining slot is a candidate
        self.assertEqual((eff2[1], eff2[2]), (5.0, 1.0))
        self.assertTrue(all(eff2[j] == 0.0 for j in range(3, 8)))
        self.assertEqual({cd["slot"]: cd["mask"] for cd in s2["candidates"]}[2], [0, 2])
        self.assertEqual((s3["masked"], s3["state_score"], s3["selected"]), ([0, 1], 1.0, 2))
        self.assertEqual({cd["slot"]: cd["effect"] for cd in s3["candidates"]}[2], 1.0)
        self.assertEqual((s4["masked"], s4["state_score"], s4["selected"], s4["tied_max"]),
                         ([0, 1, 2], 0.0, 3, [4, 5, 6, 7]))
        self.assertEqual(c["prefixes"]["fixed"], {"1": [0], "2": [0, 2], "3": [0, 1, 2], "4": [0, 1, 2, 3]})
        self.assertEqual(c["prefixes"]["adaptive"], {"1": [0], "2": [0, 1], "3": [0, 1, 2], "4": [0, 1, 2, 3]})
        self.assertEqual(c["T_w"], [0, 2])
        # the record-level checks pass on this run
        rep = v.validate(run_dir)
        for cid in ("run_files", "records_schema", "result_key_consistency", "repeat_rule", "retry_limit",
                    "score_rounding", "inputs_identity", "window_arithmetic", "slot_hashes_present",
                    "slot_identity", "mask_hash_consistency", "planned_requests", "cross_family_isolation",
                    "adaptive_invariants"):
            self.assertEqual(check_by_id(rep, cid)["status"], "passed", cid)


class TestAdditive(Base):
    def test_additive_recovers_weights_and_signs(self):
        w = [3.0, -1.0, 0.5, 2.0, 0.0, 1.5, -2.0, 4.0]
        u = [0.25, 1.0, -0.5, 0.0, 2.0, -1.0, 0.75, 0.5]
        anomaly = Spec("Burglary005_x264", 800, additive(2.5, w), gen_const(1), category="Burglary")
        # normal target: masking slot i raises the anomaly score by u_i (S = -1.5 + sum u_i)
        normal = Spec("Normal_Videos_100_x264", 640, lambda f, m: -1.5 + sum(u[i] for i in m), gen_const(0),
                      category="Normal", strata=("normal",))
        run_dir = build_run(self.tmp, [anomaly, normal])
        _, _, _, by_uid, _ = recompute(run_dir)
        ca, cn = by_uid[anomaly.uid], by_uid[normal.uid]
        self.assertEqual((ca["L0"], ca["q"]), (1, 1))
        self.assertEqual((cn["L0"], cn["q"]), (0, -1))
        for c, wt in ((ca, w), (cn, u)):
            for fam in ("blur", "grey"):
                eff = c["rankings"][fam]["effects"]
                self.assertEqual([eff[str(i)] for i in SLOTS], wt, fam)
            for row in c["pairs"]:
                i, j = row["pair"]
                self.assertEqual(row["D"], wt[i] + wt[j])
                self.assertEqual(row["J"], 0.0)
        self.assertEqual(ca["rankings"]["blur"]["order"], [7, 0, 3, 5, 2, 4, 1, 6])
        # normal target, q = -1: positive effect = masking reduced support for class 0 (raised S)
        self.assertEqual(cn["rankings"]["blur"]["order"], [4, 1, 6, 7, 0, 3, 2, 5])
        self.assertGreater(cn["rankings"]["blur"]["effects"]["4"], 0)
        self.assertLess(cn["rankings"]["blur"]["effects"]["5"], 0)
        self.assertEqual(cn["T_w"], [1, 4])
        # additive => adaptive equals fixed
        self.assertEqual(ca["adaptive"]["order"], [7, 0, 3, 5])
        self.assertEqual(cn["adaptive"]["order"], [4, 1, 6, 7])
        # e_w for an additive grey family: (w_T) - mean over pairs of (w_i + w_j) = w_T - 7*sum(w)/28
        self.assertAlmostEqual(ca["e_w"], (4.0 + 3.0) - sum(w) / 4.0, places=12)
        self.assertAlmostEqual(cn["e_w"], (2.0 + 1.0) - sum(u) / 4.0, places=12)
        self.assertEqual(ca["tau_blur_grey"], 1.0)
        # S equals S_hi on this grid, so the native ranking agrees exactly
        self.assertEqual(ca["tau_native_hi"], 1.0)
        self.assertEqual(ca["overlap_blur_grey"], 2)
        # the normal input is not in the primary population (L0 = 0)
        self.assertEqual([c["window_uid"] for c in v.primary_population([ca, cn])], [anomaly.uid])


class TestMissingLabels(Base):
    def test_missing_pair_label_and_bounds(self):
        A = flip_spec("Assault006_x264", 480, flips=[(0, 1), (2, 5)], missing=[(3, 4)], category="Assault")
        Bs = flip_spec("Explosion002_x264", 960, flips=[(2, 3), (4, 5), (6, 7)], missing=[(0, 1)],
                       category="Explosion")
        C = flip_spec("Fighting003_x264", 320, repeats={(0, 1): [None, 0]}, category="Fighting")
        run_dir = build_run(self.tmp, [A, Bs, C], B=300)
        run, auth, _, by_uid, calcs = recompute(run_dir)
        a, b, c = by_uid[A.uid], by_uid[Bs.uid], by_uid[C.uid]
        for x in (a, b, c):
            self.assertEqual(x["T_w"], [0, 1])
        # A: one comparator label missing after three repeats -> d_w missing
        self.assertIsNone(a["pair_labels"][(3, 4)])
        self.assertIsNone(a["d_w"])
        self.assertEqual(a["F_top2"], 1)
        self.assertEqual(a["n_pair_labels"], 27)
        # B: T_w's own label missing
        self.assertIsNone(b["F_top2"])
        self.assertIsNone(b["d_w"])
        # C: repeat 0 unparsed, repeat 1 parsed 0 -> label 0 is used
        self.assertEqual(c["pair_labels"][(0, 1)], 0)
        self.assertEqual(c["n_pair_labels"], 28)
        self.assertAlmostEqual(c["d_w"], 27 / 28, places=15)
        # hand-worked bounds (selected / comparator)
        self.assertAlmostEqual(a["d_lower"], 1 - 3 / 28, places=15)    # sel 1; comp 2 flips + missing->1
        self.assertAlmostEqual(a["d_upper"], 1 - 2 / 28, places=15)    # sel 1; comp 2 flips + missing->0
        self.assertAlmostEqual(b["d_lower"], 0 - 4 / 28, places=15)    # T missing: sel 0, comp 3 + 1
        self.assertAlmostEqual(b["d_upper"], 1 - 3 / 28, places=15)    # T missing: sel 1, comp 3 + 0
        prim = v.compute_primary(calcs, 0, 300)
        self.assertEqual((prim["n_inputs"], prim["n_inputs_missing_dw"], prim["n_videos"]), (3, 2, 1))
        self.assertAlmostEqual(prim["estimate_pp"], float(100 * Fraction(27, 28)), places=10)
        lo = Fraction(1, 3) * (Fraction(25, 28) + Fraction(-4, 28) + Fraction(27, 28))
        hi = Fraction(1, 3) * (Fraction(26, 28) + Fraction(25, 28) + Fraction(27, 28))
        self.assertAlmostEqual(prim["bounds_pp"][0], float(100 * lo), places=10)
        self.assertAlmostEqual(prim["bounds_pp"][1], float(100 * hi), places=10)
        self.assertEqual(prim["d_w"], {A.uid: None, Bs.uid: None, C.uid: c["d_w"]})
        # one video: every bootstrap statistic equals the estimate
        self.assertEqual(prim["ci95_pp"], [prim["estimate_pp"], prim["estimate_pp"]])
        # repeat rule passes; the regenerated input is flagged non-deterministic (texts differ)
        ck = v.check_repeat_rule(run, auth)
        self.assertEqual(ck.status(), "passed")
        self.assertGreaterEqual(ck.notes["non_deterministic_generation_inputs"], 1)


class TestCrossFamilyIsolation(Base):
    def test_grey_records_cannot_change_blur_choices(self):
        rnd = random.Random(7)
        blur_w = [rnd.choice([0.25, 0.5, 1.0, 2.0, -0.5]) for _ in SLOTS]
        grey_a = {}
        grey_b = {}

        def score_factory(grey_table):
            def s(fam, mask):
                if fam == "grey":
                    # grey values never coincide with a blur value (blur stays within [-1, 13])
                    return grey_table.setdefault(mask, rnd.choice([-30.0, -27.5, -25.0, -22.5, -20.0]))
                if fam == "blur":
                    # a non-additive blur readout with an interaction on slots 2 and 5
                    return 9.0 - sum(blur_w[i] for i in mask) - (1.5 if (2 in mask and 5 in mask) else 0.0)
                return 9.0
            return s
        g1 = gen_const(1, 1, {p: 0 for p in PAIRS[::3]})
        g2 = gen_const(1, 0, {p: 1 for p in PAIRS[::2]})
        r1 = build_run(self.tmp, [Spec("Robbery010_x264", 2400, score_factory(grey_a), g1, category="Robbery")],
                       name="run1")
        r2 = build_run(self.tmp, [Spec("Robbery010_x264", 2400, score_factory(grey_b), g2, category="Robbery")],
                       name="run2")
        uid = "Robbery010_x264@2400"
        _, auth1, _, c1, calcs1 = recompute(r1)
        _, _, _, c2, _ = recompute(r2)
        x1, x2 = c1[uid], c2[uid]
        self.assertNotEqual(x1["rankings"]["grey"], x2["rankings"]["grey"])  # grey really differs
        self.assertNotEqual(x1["pair_labels"], x2["pair_labels"])
        for f in ("adaptive", "prefixes", "T_w"):
            self.assertEqual(x1[f], x2[f], f)
        for f in ("blur", "blur_native"):
            self.assertEqual(x1["rankings"][f], x2["rankings"][f], f)
        run1 = v.load_run(r1)
        ck = v.check_cross_family_isolation(calcs1, auth1, run1.inputs_by_uid)
        self.assertEqual(ck.status(), "passed")
        self.assertEqual(ck.notes["inputs_whose_grey_ranking_changed_under_perturbation"], 1)
        # an analyzer whose adaptive step used a grey score is caught
        write_mirror_outputs(r1)

        def use_grey(recs):
            for r in recs:
                cand = r["adaptive"]["states"][1]["candidates"][0]
                cand["score"] = auth1[(uid, SCORE, "grey", tuple(cand["mask"]), 0)]["S_hi"]
            return recs
        rewrite_jsonl(os.path.join(r1, "derived", "rankings.jsonl"), use_grey)
        rep = v.validate(r1)
        self.assertEqual(check_by_id(rep, "rankings_compare")["status"], "failed")


class TestVideoClusterAggregation(Base):
    def _specs(self):
        return [
            flip_spec("VidA_x264", 160, flips=[(0, 1), (2, 3), (4, 5), (6, 7)]),   # d = 24/28
            flip_spec("VidA_x264", 480, flips=[(2, 3), (4, 5)]),                    # d = -2/28
            flip_spec("VidB_x264", 320, flips=[(0, 1)]),                            # d = 27/28
            flip_spec("VidC_x264", 160, missing=[(5, 6)]),                          # d missing
            flip_spec("VidC_x264", 640, flips=[(1, 2), (3, 4), (5, 6)]),            # d = -3/28
            flip_spec("RoadAccidents127_x264", 2240, flips=[(0, 1)]),               # development, d = 27/28
            flip_spec("VidD_x264", 160, flips=[(0, 1)], L0=0),                      # L0 = 0: not primary
        ]

    def test_hand_worked_estimate_and_bootstrap(self):
        run_dir = build_run(self.tmp, self._specs(), B=500)
        _, _, _, by_uid, calcs = recompute(run_dir)
        d = {u: by_uid[u]["d_w"] for u in by_uid}
        self.assertAlmostEqual(d["VidA_x264@160"], 24 / 28, places=15)
        self.assertAlmostEqual(d["VidA_x264@480"], -2 / 28, places=15)
        self.assertAlmostEqual(d["VidB_x264@320"], 27 / 28, places=15)
        self.assertIsNone(d["VidC_x264@160"])
        self.assertAlmostEqual(d["VidC_x264@640"], -3 / 28, places=15)
        prim = v.compute_primary(calcs, 0, 500)
        exact = Fraction(1, 3) * (Fraction(11, 28) + Fraction(27, 28) + Fraction(-3, 28))
        self.assertEqual(exact, Fraction(5, 12))
        self.assertAlmostEqual(prim["estimate_pp"], float(100 * exact), places=12)
        self.assertEqual(sorted(prim["video_means_pp"]), ["VidA_x264", "VidB_x264", "VidC_x264"])
        self.assertAlmostEqual(prim["video_means_pp"]["VidA_x264"], 100 * 11 / 28, places=12)
        self.assertEqual((prim["n_videos"], prim["n_inputs"], prim["n_inputs_missing_dw"]), (3, 5, 1))
        # with development videos: (11 + 27 - 3 + 27) / (28 * 4)
        wd = prim["sensitivity"]["with_development"]
        self.assertAlmostEqual(wd["estimate_pp"], float(100 * Fraction(62, 112)), places=12)
        self.assertEqual(wd["n_inputs"], 6)
        # bootstrap: reproducible for a fixed seed, and equal to the protocol's recipe written out here
        means = [11 / 28, 27 / 28, -3 / 28]
        _, vmeans, _ = v.aggregate([(by_uid[u]["video_id"], by_uid[u]["d_w"]) for u in sorted(by_uid)
                                    if by_uid[u]["L0"] == 1 and not by_uid[u]["development"]])
        for a, b in zip(vmeans, means):
            self.assertAlmostEqual(a, b, places=15)
        rng = random.Random(0)
        stats = sorted(sum(vmeans[rng.randrange(3)] for _ in range(3)) / 3 for _ in range(500))
        pos_lo, pos_hi = 499 * 0.025, 499 * 0.975
        expect = [stats[int(pos_lo)] + (stats[int(pos_lo) + 1] - stats[int(pos_lo)]) * (pos_lo - int(pos_lo)),
                  stats[int(pos_hi)] + (stats[int(pos_hi) + 1] - stats[int(pos_hi)]) * (pos_hi - int(pos_hi))]
        self.assertEqual(v.bootstrap_ci(vmeans, 0, 500), expect)
        self.assertEqual(prim["ci95_pp"], [100.0 * expect[0], 100.0 * expect[1]])
        again = v.compute_primary(calcs, 0, 500)
        self.assertEqual(again["ci95_pp"], prim["ci95_pp"])
        self.assertEqual([s["seed"] for s in prim["bootstrap"]["stability"]], [1, 2, 3, 4])
        self.assertEqual(prim["bootstrap"]["stability"][0]["ci95_pp"], pp_ci(vmeans, 1, 500))
        # a different seed draws a different resample sequence (the 3-video percentiles may still coincide)
        r0, r1 = random.Random(0), random.Random(1)
        self.assertNotEqual([r0.randrange(3) for _ in range(30)], [r1.randrange(3) for _ in range(30)])

    def test_percentile_interpolation(self):
        vals = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        # position (B-1) q = 6 * 0.025 = 0.15 and 6 * 0.975 = 5.85
        self.assertAlmostEqual(v.percentile_sorted(vals, 0.025), 0.15, places=15)
        self.assertAlmostEqual(v.percentile_sorted(vals, 0.975), 16.0 + 0.85 * 16.0, places=12)


def pp_ci(means, seed, B):
    rng = random.Random(seed)
    stats = sorted(sum(means[rng.randrange(len(means))] for _ in range(len(means))) / len(means)
                   for _ in range(B))
    out = []
    for qq in (0.025, 0.975):
        pos = (B - 1) * qq
        lo = math.floor(pos)
        out.append(100.0 * (stats[lo] + (stats[min(lo + 1, B - 1)] - stats[lo]) * (pos - lo)))
    return out


class TestTies(Base):
    def test_ties_cuts_and_flags(self):
        X = Spec("Shooting002_x264", 1600, additive(5.0, [2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), gen_const(1),
                 category="Shooting")
        Y = Spec("Shooting004_x264", 1600, additive(5.0, [0.0] * 8), gen_const(1), category="Shooting")
        Z = Spec("Stealing003_x264", 1600, additive(5.0, [3.0, 2.0, 2.0, 0.0, -1.0, -1.0, -1.0, -1.0]),
                 gen_const(1), category="Stealing")
        W = Spec("Vandalism007_x264", 1600, additive(5.0, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -0.5]),
                 gen_const(1), category="Vandalism")
        run_dir = build_run(self.tmp, [X, Y, Z, W])
        _, _, _, by_uid, calcs = recompute(run_dir)
        x = by_uid[X.uid]["rankings"]["blur"]
        self.assertEqual(x["order"], list(range(8)))
        self.assertEqual(x["ties"], [[0, 1], [2, 3, 4], [5, 6, 7]])
        self.assertEqual(x["tie_spans_cut"], {"1": True, "2": False, "3": True, "4": True})
        self.assertEqual(x["top2_margin"], 1.0)
        self.assertFalse(x["all_tied"])
        self.assertFalse(x["all_nonpositive"])
        y = by_uid[Y.uid]["rankings"]["blur"]
        self.assertTrue(y["all_tied"])
        self.assertTrue(y["all_nonpositive"])
        self.assertEqual(y["ties"], [list(range(8))])
        self.assertEqual(y["tie_spans_cut"], {"1": True, "2": True, "3": True, "4": True})
        self.assertEqual(y["top2_margin"], 0.0)
        st1 = by_uid[Y.uid]["adaptive"]["states"][0]
        self.assertEqual((st1["selected"], st1["tied_max"]), (0, [1, 2, 3, 4, 5, 6, 7]))
        z = by_uid[Z.uid]["rankings"]["blur"]
        self.assertEqual(z["order"], [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(z["tie_spans_cut"]["2"])
        self.assertEqual(z["top2_margin"], 0.0)
        w = by_uid[W.uid]["rankings"]["blur"]
        self.assertTrue(w["all_nonpositive"])
        self.assertFalse(w["all_tied"])
        self.assertEqual(w["order"], [7, 0, 1, 2, 3, 4, 5, 6])
        # tie-free sensitivity drops inputs whose tie straddles the cut after position 2 (Y and Z)
        tf = [c["window_uid"] for c in v.sensitivity_population("tie_free", calcs)]
        self.assertEqual(sorted(tf), sorted([X.uid, W.uid]))
        # flagged inputs stay in the primary population
        self.assertEqual(len(v.primary_population(calcs)), 4)


class TestCorruptedRecords(Base):
    def _run(self):
        sp = Spec("Abuse028_x264", 400, interaction_score, gen_const(1), category="Abuse")
        return sp, build_run(self.tmp, [sp])

    def test_result_key_conflict_and_slot_hash_mismatch_flagged(self):
        sp, run_dir = self._run()
        self.assertTrue(v.validate(run_dir)["summary"]["failed"] == 0)
        path = os.path.join(run_dir, "requests", "fixed.jsonl")

        def corrupt(recs):
            dup = None
            for r in recs:
                if r["target"] == SCORE and r["family"] == "blur" and r["mask"] == [3]:
                    dup = dict(r)
                if r["target"] == SCORE and r["family"] == "blur" and r["mask"] == [1, 2]:
                    r["slot_tensor_sha256"] = list(r["slot_tensor_sha256"])
                    r["slot_tensor_sha256"][6] = "0" * 64           # an unmasked slot differs
            dup["S_hi"] = dup["S_hi"] + 0.5                          # same result_key, different S_hi
            dup["hi_logit_1"] = dup["S_hi"]
            dup["cache_status"] = "cached"
            dup["attempt"] = 2
            recs.append(dup)
            return recs
        rewrite_jsonl(path, corrupt)
        rep = v.validate(run_dir)
        self.assertFalse(rep["passed"])
        rk = check_by_id(rep, "result_key_consistency")
        self.assertEqual(rk["status"], "failed")
        self.assertTrue(any(d.get("field") == "S_hi" for d in rk["discrepancies"]))
        si = check_by_id(rep, "slot_identity")
        self.assertEqual(si["status"], "failed")
        self.assertEqual(si["discrepancies"][0]["unmasked_slots_changed"], [6])
        self.assertEqual(si["discrepancies"][0]["mask"], [1, 2])
        mh = check_by_id(rep, "mask_hash_consistency")
        self.assertEqual(mh["status"], "passed")  # blur pairs carry a single (score) line
        self.assertEqual(v.exit_code(rep), 1)

    def test_masked_slot_unchanged_and_identity_flag_false_flagged(self):
        sp, run_dir = self._run()
        path = os.path.join(run_dir, "requests", "fixed.jsonl")
        ref = slot_hashes(sp.uid, "none", ())

        def corrupt(recs):
            for r in recs:
                if r["target"] == SCORE and r["family"] == "blur" and r["mask"] == [0, 5]:
                    r["slot_tensor_sha256"][5] = ref[5]               # a masked slot equals the original
                if r["target"] == ORIG and r["mask"] == [2, 3]:
                    r["slot_identity_ok"] = False
            return recs
        rewrite_jsonl(path, corrupt)
        rep = v.validate(run_dir)
        si = check_by_id(rep, "slot_identity")
        self.assertEqual(si["status"], "failed")
        self.assertEqual(si["discrepancies"][0]["masked_slots_unchanged"], [5])
        self.assertEqual(check_by_id(rep, "slot_hashes_present")["status"], "failed")
        # blur [0, 5] has a single line, so no cross-target disagreement exists to report
        self.assertEqual(check_by_id(rep, "mask_hash_consistency")["status"], "passed")

    def test_cross_target_hash_disagreement_flagged(self):
        sp, run_dir = self._run()
        path = os.path.join(run_dir, "requests", "fixed.jsonl")

        def corrupt(recs):
            for r in recs:
                if r["target"] == ORIG and r["family"] == "grey" and r["mask"] == [4, 7]:
                    r["slot_tensor_sha256"] = list(r["slot_tensor_sha256"])
                    r["slot_tensor_sha256"][4] = _h("other grey")
            return recs
        rewrite_jsonl(path, corrupt)
        rep = v.validate(run_dir)
        self.assertEqual(check_by_id(rep, "mask_hash_consistency")["status"], "failed")

    def test_bad_repeat_and_rounding_flagged(self):
        sp, run_dir = self._run()
        path = os.path.join(run_dir, "requests", "fixed.jsonl")

        def corrupt(recs):
            extra = None
            for r in recs:
                if r["target"] == ORIG and r["mask"] == [0, 1]:
                    extra = dict(r, repeat=1, request_id=r["request_id"][:-2] + "r1")  # repeat after a parse
                if r["target"] == SCORE and r["family"] == "grey" and r["mask"] == [6, 7]:
                    r["logit_1"] = r["logit_1"] + 0.25                                 # BF16 logit mismatch
            recs.append(extra)
            return recs
        rewrite_jsonl(path, corrupt)
        rep = v.validate(run_dir)
        self.assertEqual(check_by_id(rep, "repeat_rule")["status"], "failed")
        self.assertEqual(check_by_id(rep, "score_rounding")["status"], "failed")

    def test_planned_request_missing_or_extra_flagged(self):
        sp, run_dir = self._run()

        def drop(recs):
            return [r for r in recs if r["mask"] != [0, 1, 3]]   # an adaptive state-3 candidate
        rewrite_jsonl(os.path.join(run_dir, "requests", "adaptive.jsonl"), drop)
        rep = v.validate(run_dir)
        pr = check_by_id(rep, "planned_requests")
        self.assertEqual(pr["status"], "failed")
        self.assertEqual(pr["discrepancies"][0]["mask"], [0, 1, 3])

    def test_unplanned_blur_generation_flagged(self):
        sp, run_dir = self._run()

        def add(recs):
            recs.append(gen_line(sp.uid, "fixed", "pair", ORIG, "blur", (0, 1), 0, 1))
            return recs
        rewrite_jsonl(os.path.join(run_dir, "requests", "fixed.jsonl"), add)
        rep = v.validate(run_dir)
        pr = check_by_id(rep, "planned_requests")
        self.assertEqual(pr["status"], "failed")
        self.assertEqual((pr["discrepancies"][0]["family"], pr["discrepancies"][0]["target"]), ("blur", ORIG))


class TestKendall(unittest.TestCase):
    def test_tau_b_values(self):
        self.assertEqual(v.kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(v.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        # ties: nc = 4, nd = 0, n0 = 6, one tied pair in x, one in y -> 4 / sqrt(5 * 5)
        self.assertAlmostEqual(v.kendall_tau_b([1, 1, 2, 3], [1, 2, 2, 3]), 0.8, places=15)
        self.assertIsNone(v.kendall_tau_b([1, 1, 1], [1, 2, 3]))
        # a pair tied in both vectors counts in both tie totals: nc = 5, nd = 0, tx = ty = 1
        self.assertAlmostEqual(v.kendall_tau_b([1, 1, 2, 3], [5, 5, 6, 7]), 5 / 5, places=15)
        # 8-slot example with mixed signs and ties
        x = [3.0, 1.0, 2.0, 2.0, 0.0, -1.0, 0.0, 5.0]
        y = [1.0, 0.0, 2.0, 1.0, 0.0, -2.0, 1.0, 4.0]
        nc = nd = tx = ty = 0
        for i in range(8):
            for j in range(i + 1, 8):
                a = (x[i] > x[j]) - (x[i] < x[j])
                b = (y[i] > y[j]) - (y[i] < y[j])
                tx += a == 0
                ty += b == 0
                if a and b:
                    nc += a == b
                    nd += a != b
        self.assertAlmostEqual(v.kendall_tau_b(x, y), (nc - nd) / math.sqrt((28 - tx) * (28 - ty)), places=15)


class TestWindowArithmetic(unittest.TestCase):
    def test_vera_frames_match_experiment02_cases(self):
        # frame lists recorded in experiment02 cases.json for RoadAccidents127_x264 (2580 frames)
        self.assertEqual(v.vera_frames(0, 2580)[0], [0, 18, 37, 56, 75, 93, 112, 131])
        self.assertEqual(v.vera_frames(496, 2580)[0], [346, 383, 421, 458, 496, 533, 571, 608])
        self.assertEqual(v.vera_frames(2240, 2580)[0], [2090, 2127, 2165, 2202, 2240, 2277, 2315, 2352])
        self.assertEqual(v.vera_frames(2576, 2580)[0], [2426, 2445, 2464, 2483, 2503, 2522, 2541, 2560])
        self.assertEqual(v.vera_frames(592, 1181)[0], [442, 479, 517, 554, 592, 629, 667, 704])

    def test_mismatching_frames_flagged(self):
        tmp = tempfile.mkdtemp(prefix="exp03b_validate_test_")
        try:
            run_dir = build_run(tmp, [flip_spec("Arrest001_x264", 1600)])
            p = os.path.join(run_dir, "inputs.json")
            with open(p) as fh:
                obj = json.load(fh)
            obj["inputs"][0]["frame_indices"][3] += 1
            with open(p, "w") as fh:
                json.dump(obj, fh)
            run = v.load_run(run_dir)
            self.assertEqual(v.check_window_arithmetic(run).status(), "failed")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestErratumE1(Base):
    def _score(self, cid, family, mask, hi1, hi0=30.0, repeat=0, stage="B"):
        l1, l0 = _bf16(_f32(hi1)), _bf16(_f32(hi0))
        return {"request_id": "%s|%s|%s|%s" % (stage, cid, family, mask), "stage": stage, "target": SCORE,
                "case_id": cid, "family": family, "mask": mask, "repeat": repeat, "cache_status": "fresh",
                "frame_indices": [1, 2, 3, 4, 5, 6, 7, 8], "S_hi": _f32(_f32(hi1) - _f32(hi0)),
                "hi_logit_1": _f32(hi1), "hi_logit_0": _f32(hi0), "logit_1": l1, "logit_0": l0,
                "S": (l1 - l0) + 1e-8, "S_resolution": 0.25}

    def test_synthetic_e1(self):
        root = os.path.join(self.tmp, "exp02")
        os.makedirs(os.path.join(root, "results"))
        base = [self._score("C1", "none", [], 32.1, repeat=r, stage="A2") for r in (0, 1, 2)]
        pert = [self._score("C1", "blur", [0], 32.2),       # dS_hi = -0.1 (sub-step), BF16 32.0 -> 32.25: changed
                self._score("C1", "blur", [1], 32.05),      # sub-step, BF16 unchanged
                self._score("C1", "constant", [0, 1], 31.0)]  # dS_hi = 1.1: not sub-step
        with open(os.path.join(root, "results", "baseline_responses.jsonl"), "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in base) + "\n")
        with open(os.path.join(root, "results", "perturbation_responses.jsonl"), "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in pert) + "\n")
        ck, summ = v.check_e1(root, expected={"n_rows": 3, "n_substep": 2, "n_substep_bf16_changed": 1})
        self.assertEqual(ck.status(), "passed", ck.to_json())
        self.assertEqual((summ["n_rows"], summ["n_substep"], summ["n_substep_bf16_changed"]), (3, 2, 1))
        # the literal stored-S difference (with float noise) is reported separately
        self.assertEqual(summ["n_substep_raw_S_field_nonzero"], 1)
        ck2, _ = v.check_e1(root)  # against the documented experiment02 counts: must fail here
        self.assertEqual(ck2.status(), "failed")

    @unittest.skipUnless(os.path.isdir(EXP02_RUN), "experiment02 run_002 not present")
    def test_experiment02_e1(self):
        ck, summ = v.check_e1(EXP02_RUN)
        self.assertEqual(ck.status(), "passed", json.dumps(ck.to_json())[:2000])
        self.assertEqual((summ["n_rows"], summ["n_substep"], summ["n_substep_bf16_changed"]), (168, 109, 36))


class TestEndToEnd(Base):
    def _specs(self):
        return [
            Spec("Arson011_x264", 1600, interaction_score, gen_const(1, 1, {(0, 2): 0, (0, 1): 0}), category="Arson"),
            flip_spec("Burglary005_x264", 800, flips=[(0, 1), (3, 6)], category="Burglary"),
            flip_spec("Burglary005_x264", 1120, flips=[(2, 7)], missing=[(5, 6)], category="Burglary",
                      strata=("event", "negative")),
            flip_spec("Fighting003_x264", 320, flips=[(0, 1)], metadata_flags=("fps_not_30",), category="Fighting"),
            flip_spec("RoadAccidents127_x264", 2240, flips=[(0, 1)]),
            Spec("Normal_Videos_100_x264", 640, additive(-1.25, [0.25, 1.0, -0.5, 0.0, 2.0, -1.0, 0.75, 0.5]),
                 gen_const(0, 0, {(1, 4): 1}), category="Normal", strata=("normal",)),
            Spec("Normal_Videos_200_x264", 960, additive(-2.0, [0.5] * 8), gen_const([None, None, None]),
                 category="Normal", strata=("normal",)),                      # L0 missing: Phase 0 only
        ]

    def test_matching_analyzer_outputs_pass_and_corruptions_fail(self):
        run_dir = build_run(self.tmp, self._specs(), B=400)
        rep = v.validate(run_dir)
        self.assertEqual(rep["summary"]["failed"], 0, [c for c in rep["checks"] if c["status"] == "failed"])
        for cid in ("rankings_compare", "primary_compare", "inputs_metrics_compare", "pairs_compare",
                    "trajectories_compare", "erratum_e1"):
            c = check_by_id(rep, cid)
            self.assertEqual(c["status"], "skipped", cid)
            self.assertIsNone(c["passed"])               # skipped is never reported as passed
        self.assertFalse(rep["passed"])
        self.assertEqual(v.exit_code(rep), 3)
        counts = rep["recomputed"]["counts"]
        self.assertEqual((counts["n_inputs"], counts["n_L0_1"], counts["n_L0_0"], counts["n_L0_missing"]),
                         (7, 5, 1, 1))
        prim = write_mirror_outputs(run_dir)
        exp02 = EXP02_RUN if os.path.isdir(EXP02_RUN) else None
        rep = v.validate(run_dir, exp02)
        bad = [c for c in rep["checks"] if c["status"] != "passed"]
        if exp02:
            self.assertTrue(rep["passed"], bad)
            self.assertEqual(v.exit_code(rep), 0)
        else:
            self.assertEqual([c["id"] for c in bad], ["erratum_e1"])
        self.assertEqual(rep["recomputed"]["primary"]["estimate_pp"], prim["estimate_pp"])
        self.assertEqual(rep["recomputed"]["primary"]["n_inputs"], 4)   # development and L0 != 1 excluded
        sens = rep["recomputed"]["primary"]["sensitivity"]
        self.assertEqual(sens["with_development"]["n_inputs"], 5)
        self.assertEqual(sens["metadata_clean"]["n_inputs"], 3)
        # CLI end to end
        out = os.path.join(self.tmp, "validation.json")
        argv = ["--run-dir", run_dir, "--out", out] + (["--exp02-run-dir", exp02] if exp02 else [])
        devnull = open(os.devnull, "w")
        old = sys.stdout
        sys.stdout = devnull
        try:
            code = v.main(argv)
        finally:
            sys.stdout = old
            devnull.close()
        self.assertEqual(code, 0 if exp02 else 3)
        with open(out) as fh:
            self.assertEqual(json.load(fh)["passed"], bool(exp02))

        # corruptions of each analyzer output are caught
        res = os.path.join(run_dir, "results")

        def corrupt_rank(recs):
            recs[0]["fixed"]["blur"]["order"] = list(reversed(recs[0]["fixed"]["blur"]["order"]))
            return recs
        self._expect_fail(run_dir, os.path.join(run_dir, "derived", "rankings.jsonl"), corrupt_rank,
                          "rankings_compare", jsonl=True)

        def corrupt_primary(p):
            p["ci95_pp"][0] = p["ci95_pp"][0] + 1e-6       # outside the 1e-9 pp comparison tolerance
            return p
        self._expect_fail(run_dir, os.path.join(res, "primary.json"), corrupt_primary, "primary_compare")

        def corrupt_metrics(rows):
            for r in rows:
                if r["d_w"]:
                    r["d_w"] = repr(float(r["d_w"]) + 1e-6)
                    break
            return rows
        self._expect_fail(run_dir, os.path.join(res, "inputs_metrics.csv"), corrupt_metrics,
                          "inputs_metrics_compare", csvfile=True)

        def corrupt_pairs(rows):
            rows[5]["J"] = repr(float(rows[5]["J"]) + 0.01)
            return rows
        self._expect_fail(run_dir, os.path.join(res, "pairs.csv"), corrupt_pairs, "pairs_compare", csvfile=True)

        def corrupt_traj(rows):
            for r in rows:
                if r["label"] in ("0", "1"):
                    r["label"] = "1" if r["label"] == "0" else "0"
                    break
            return rows
        self._expect_fail(run_dir, os.path.join(res, "trajectories.csv"), corrupt_traj, "trajectories_compare",
                          csvfile=True)

    def _expect_fail(self, run_dir, path, fn, cid, jsonl=False, csvfile=False):
        with open(path, newline="") as fh:
            original = fh.read()
        try:
            if jsonl:
                rewrite_jsonl(path, fn)
            elif csvfile:
                with open(path, newline="") as fh:
                    rdr = csv.DictReader(fh)
                    rows, headers = list(rdr), rdr.fieldnames
                rows = fn(rows)
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=headers)
                    w.writeheader()
                    w.writerows(rows)
            else:
                with open(path) as fh:
                    obj = fn(json.load(fh))
                with open(path, "w") as fh:
                    json.dump(obj, fh)
            rep = v.validate(run_dir)
            self.assertEqual(check_by_id(rep, cid)["status"], "failed", cid)
            self.assertEqual(v.exit_code(rep), 1)
        finally:
            with open(path, "w", newline="") as fh:
                fh.write(original)


class TestIncompleteRanking(Base):
    def test_failed_single_makes_ranking_incomplete(self):
        sp = flip_spec("Shoplifting001_x264", 800, flips=[(0, 1), (2, 3)], category="Shoplifting")
        sp.fail = {(SCORE, "blur", (3,))}
        ok = flip_spec("Shoplifting005_x264", 800, flips=[(0, 1)], category="Shoplifting")
        run_dir = build_run(self.tmp, [sp, ok])
        _, _, _, by_uid, calcs = recompute(run_dir)
        c = by_uid[sp.uid]
        self.assertEqual(c["rankings"]["blur"], {"status": "incomplete", "missing": [3]})
        self.assertNotIn("status", c["rankings"]["grey"])
        self.assertEqual(c["adaptive"]["status"], "incomplete")
        self.assertIsNone(c["T_w"])
        self.assertIsNone(c["d_w"])
        self.assertIsNone(c["F_top2"])
        self.assertAlmostEqual(c["d_lower"], 0 - 2 / 28, places=15)   # T_w undefined: selected 0 / 1
        self.assertAlmostEqual(c["d_upper"], 1 - 2 / 28, places=15)
        self.assertEqual(c["trajectories"], [])
        rep = v.validate(run_dir)
        self.assertEqual(rep["summary"]["failed"], 0, [x for x in rep["checks"] if x["status"] == "failed"])
        self.assertEqual(check_by_id(rep, "retry_limit")["status"], "passed")
        self.assertEqual(check_by_id(rep, "planned_requests")["notes"]["inputs_with_undetermined_schedule"], 1)
        write_mirror_outputs(run_dir)
        rep = v.validate(run_dir)
        for cid in ("rankings_compare", "primary_compare", "inputs_metrics_compare", "pairs_compare",
                    "trajectories_compare"):
            self.assertEqual(check_by_id(rep, cid)["status"], "passed", cid)
        self.assertEqual(rep["recomputed"]["primary"]["n_inputs_missing_dw"], 1)

    def test_fourth_error_attempt_flagged(self):
        sp = flip_spec("Shoplifting001_x264", 800)
        sp.fail = {(SCORE, "grey", (2,))}
        run_dir = build_run(self.tmp, [sp])

        def add(recs):
            e = [r for r in recs if r["cache_status"] == "error"][0]
            recs.append(dict(e, attempt=4))
            return recs
        rewrite_jsonl(os.path.join(run_dir, "requests", "fixed.jsonl"), add)
        rep = v.validate(run_dir)
        self.assertEqual(check_by_id(rep, "retry_limit")["status"], "failed")


class TestConfigAndManifest(Base):
    def test_bootstrap_params(self):
        self.assertEqual(v.bootstrap_params({"bootstrap": {"seed": 5, "B": 2000}})[:2], (5, 2000))
        self.assertEqual(v.bootstrap_params({"seed": 3, "analysis": {"bootstrap_B": 999}})[:2], (3, 999))
        self.assertEqual(v.bootstrap_params({"analysis": {"bootstrap": {"seed": 7, "resamples": 50}}, "seed": 1})[:2],
                         (7, 50))
        seed, B, info = v.bootstrap_params({})
        self.assertEqual((seed, B, info["seed_source"]), (0, 10000, "protocol default"))

    def test_manifest_windows_checked(self):
        sp = flip_spec("Arrest001_x264", 1600)
        run_dir = build_run(self.tmp, [sp])
        p = os.path.join(run_dir, "frozen", "manifest.json")
        with open(p) as fh:
            man = json.load(fh)
        man["windows"] = [{"window_record_id": "w1", "video_id": "Arrest001_x264", "anchor": 1600,
                           "window": [1450.0, 1750.0], "frame_indices": frames_for(1600), "clipped": False}]
        with open(p, "w") as fh:
            json.dump(man, fh)
        self.assertEqual(v.check_window_arithmetic(v.load_run(run_dir)).status(), "passed")
        man["windows"][0]["clipped"] = True
        with open(p, "w") as fh:
            json.dump(man, fh)
        ck = v.check_window_arithmetic(v.load_run(run_dir))
        self.assertEqual(ck.status(), "failed")
        self.assertEqual(ck.discrepancies[0]["problem"], "clipped flag differs")

    def test_unavailable_window_record_not_checked(self):
        # regression: an 'unavailable' benchmark-negative record has no window (PROTOCOL 3.1)
        run_dir = build_run(self.tmp, [flip_spec("Arrest001_x264", 1600)])
        p = os.path.join(run_dir, "frozen", "manifest.json")
        with open(p) as fh:
            man = json.load(fh)
        rec = {"window_record_id": "Arrest001_x264#negative", "stratum": "negative", "video_id": "Arrest001_x264",
               "anchor": None, "window": None, "frame_indices": [], "clipped": None, "status": "unavailable"}
        man["windows"] = [rec]
        with open(p, "w") as fh:
            json.dump(man, fh)
        self.assertEqual(v.check_window_arithmetic(v.load_run(run_dir)).status(), "passed")
        man["windows"] = [dict(rec, status="selected")]   # the same record claiming to be selected is flagged
        with open(p, "w") as fh:
            json.dump(man, fh)
        self.assertEqual(v.check_window_arithmetic(v.load_run(run_dir)).status(), "failed")


class TestReviewRegressions(Base):
    """Regression tests for defects found in the adversarial review of validate3b.py."""

    def test_bootstrap_ci_float_noise_is_not_a_discrepancy(self):
        # an analyzer that follows PROTOCOL 7.1 but scales to pp before averaging differs only in the last bits
        specs = [flip_spec("VidA_x264", 160, flips=[(0, 1), (2, 3), (4, 5), (6, 7)]),
                 flip_spec("VidA_x264", 480, flips=[(2, 3), (4, 5)]),
                 flip_spec("VidA_x264", 800, flips=[(2, 3)]),
                 flip_spec("VidB_x264", 320, flips=[(0, 2), (3, 4), (2, 6)]),
                 flip_spec("VidC_x264", 640, flips=[(1, 2), (3, 4), (5, 6)]),
                 flip_spec("VidC_x264", 960, flips=[(0, 1), (3, 4), (5, 6), (2, 5), (0, 3)]),
                 flip_spec("VidE_x264", 640, flips=[(0, 1), (1, 2), (3, 4), (5, 6), (0, 7)]),
                 flip_spec("VidF_x264", 640, flips=[(2, 4), (3, 5)])]
        run_dir = build_run(self.tmp, specs, B=2000)
        write_mirror_outputs(run_dir)
        _, _, _, _, calcs = recompute(run_dir)
        by = {}
        for c in v.primary_population(calcs):
            by.setdefault(c["video_id"], []).append(100.0 * c["d_w"])
        means_pp = [sum(by[k]) / len(by[k]) for k in sorted(by)]

        def pp_first(seed):
            rng = random.Random(seed)
            V = len(means_pp)
            stats = sorted(sum(means_pp[rng.randrange(V)] for _ in range(V)) / V for _ in range(2000))
            out = []
            for qq in (0.025, 0.975):
                pos = 1999 * qq
                lo = math.floor(pos)
                out.append(stats[lo] + (stats[lo + 1] - stats[lo]) * (pos - lo))
            return out
        p = os.path.join(run_dir, "results", "primary.json")
        with open(p) as fh:
            obj = json.load(fh)
        alt = pp_first(0)
        self.assertNotEqual(alt, obj["ci95_pp"])            # the fixture really differs in the last bits
        self.assertLess(max(abs(a - b) for a, b in zip(alt, obj["ci95_pp"])), 1e-12)
        obj["ci95_pp"] = alt
        for s in obj["bootstrap"]["stability"]:
            s["ci95_pp"] = pp_first(s["seed"])
        with open(p, "w") as fh:
            json.dump(obj, fh)
        self.assertEqual(check_by_id(v.validate(run_dir), "primary_compare")["status"], "passed")
        obj["bootstrap"]["stability"][2]["ci95_pp"][1] += 1e-6  # a real difference is still caught
        with open(p, "w") as fh:
            json.dump(obj, fh)
        self.assertEqual(check_by_id(v.validate(run_dir), "primary_compare")["status"], "failed")

    def test_third_retry_after_three_errors_flagged(self):
        sp = flip_spec("Shoplifting001_x264", 800)
        run_dir = build_run(self.tmp, [sp])

        def add(n_errors):
            def fn(recs):
                out = []
                for r in recs:
                    if r["target"] == SCORE and r["family"] == "grey" and r["mask"] == [2]:
                        out.extend(e for e in error_lines(sp.uid, "fixed", "single", SCORE, "grey", (2,))[:n_errors])
                        r = dict(r, attempt=n_errors + 1)
                    out.append(r)
                return out
            return fn
        path = os.path.join(run_dir, "requests", "fixed.jsonl")
        with open(path) as fh:
            original = fh.read()
        rewrite_jsonl(path, add(2))       # error, error, success: two retries
        self.assertEqual(check_by_id(v.validate(run_dir), "retry_limit")["status"], "passed")
        with open(path, "w") as fh:
            fh.write(original)
        rewrite_jsonl(path, add(3))       # error, error, error, success: three retries
        ck = check_by_id(v.validate(run_dir), "retry_limit")
        self.assertEqual(ck["status"], "failed")
        self.assertEqual((ck["discrepancies"][0]["n_error_attempts"], ck["discrepancies"][0]["n_retries"]), (3, 3))

    def test_unparsed_generation_must_be_regenerated(self):
        for i, labs in enumerate(([None], [None, None])):
            sp = flip_spec("Assault006_x264", 480, flips=[(0, 1)], repeats={(3, 4): labs}, category="Assault")
            run_dir = build_run(self.tmp, [sp], name="run%d" % i)
            ck = check_by_id(v.validate(run_dir), "repeat_rule")
            self.assertEqual(ck["status"], "failed", labs)
            self.assertEqual((ck["discrepancies"][0]["mask"], ck["discrepancies"][0]["repeat"]), ([3, 4], len(labs) - 1))
        sp = flip_spec("Assault006_x264", 480, flips=[(0, 1)], repeats={(3, 4): [None, None, None]},
                       category="Assault")
        run_dir = build_run(self.tmp, [sp], name="run_all_three")
        self.assertEqual(check_by_id(v.validate(run_dir), "repeat_rule")["status"], "passed")

    def test_accumulated_replacement_flagged(self):
        sp = Spec("Abuse028_x264", 400, interaction_score, gen_const(1), category="Abuse")
        run_dir = build_run(self.tmp, [sp])
        self.assertEqual(check_by_id(v.validate(run_dir), "slot_identity")["status"], "passed")

        def reblur(recs):
            for r in recs:
                if r["family"] == "blur" and r["mask"] == [0, 1, 3]:
                    r["slot_tensor_sha256"][0] = _h(sp.uid, 0, "blur(blur)")   # slot 0 blurred twice
                    r["sequence_key"] = _h("seq", *r["slot_tensor_sha256"])
            return recs
        rewrite_jsonl(os.path.join(run_dir, "requests", "adaptive.jsonl"), reblur)
        ck = check_by_id(v.validate(run_dir), "slot_identity")
        self.assertEqual(ck["status"], "failed")
        self.assertEqual((ck["discrepancies"][0]["family"], ck["discrepancies"][0]["slot"]), ("blur", 0))


def store_from(uid, scores=None, gens=None):
    """A Store over hand-made authoritative records: scores {(family, mask): (S_hi, S)},
    gens {(target, family, mask, repeat): label or None}."""
    auth = {}
    for (fam, mask), (shi, s) in (scores or {}).items():
        # protocol v1.1: the native margin is logit_1 - logit_0 (the stored S keeps its own value)
        auth[(uid, SCORE, fam, tuple(mask), 0)] = {"S_hi": shi, "S": s, "logit_0": 36.0, "logit_1": 36.0 + s}
    for (target, fam, mask, rep), lab in (gens or {}).items():
        auth[(uid, target, fam, tuple(mask), rep)] = {"label": lab,
                                                      "parse_status": "ok" if lab is not None else "no_output_line"}
    return v.Store(auth)


class TestProtocolPins(Base):
    """Rules the earlier fixtures could not tell apart from plausible wrong implementations (review finding:
    each test below fails for a named mutant of validate3b.py)."""

    def test_bootstrap_videos_sorted_by_video_id(self):
        # inputs listed out of video_id order; PROTOCOL 7.1 sorts videos before drawing indices
        specs = [flip_spec("VidE_x264", 160, flips=[(0, 1), (2, 3)]),
                 flip_spec("VidC_x264", 160, flips=[(2, 3), (4, 5), (6, 7), (1, 5)]),
                 flip_spec("VidA_x264", 160, flips=[(0, 1)]),
                 flip_spec("VidD_x264", 160, flips=[(1, 4), (2, 6), (3, 7)]),
                 flip_spec("VidB_x264", 160, flips=[(0, 1), (2, 3), (4, 5), (6, 7), (0, 7), (1, 6)])]
        run_dir = build_run(self.tmp, specs, B=300)
        _, _, _, _, calcs = recompute(run_dir)
        prim = v.compute_primary(calcs, 0, 300)
        self.assertEqual(list(prim["video_means_pp"]), sorted(prim["video_means_pp"]))
        in_order = [100.0 * c["d_w"] for c in calcs]                       # one input per video
        sorted_means = [m for _, m in sorted(zip([c["video_id"] for c in calcs], in_order))]
        self.assertNotEqual(pp_ci([m / 100 for m in in_order], 0, 300), pp_ci([m / 100 for m in sorted_means], 0, 300))
        for a, b in zip(prim["ci95_pp"], pp_ci([m / 100 for m in sorted_means], 0, 300)):
            self.assertAlmostEqual(a, b, places=9)

    def test_scorer_agreement_uses_native_margin_class(self):
        X = flip_spec("VidX_x264", 160, flips=[(0, 1)])
        Y = flip_spec("VidY_x264", 160, flips=[(0, 1)])
        Z = flip_spec("VidZ_x264", 160, flips=[(0, 1)])
        run_dir = build_run(self.tmp, [X, Y, Z])
        native = {X.uid: 0.0, Y.uid: -0.25, Z.uid: 0.5}                   # S_hi stays +10 for all three

        def set_s(recs):
            for r in recs:
                if r["target"] == SCORE and r["family"] == "none":  # v1.1: native = logit_1 - logit_0
                    r["S"] = native[r["window_uid"]]
                    r["logit_0"], r["logit_1"] = 36.0, 36.0 + native[r["window_uid"]]
            return recs
        rewrite_jsonl(os.path.join(run_dir, "requests", "baseline.jsonl"), set_s)
        _, _, _, by_uid, calcs = recompute(run_dir)
        self.assertEqual(by_uid[X.uid]["baseline_native_class"], "undecided")   # S = 0 is undecided
        self.assertEqual([c["window_uid"] for c in v.sensitivity_population("scorer_agreement", calcs)], [Z.uid])

    def test_label_rule_repeat0_then_first_parsed(self):
        uid = "VidR_x264@160"
        st = store_from(uid, gens={(ORIG, "grey", (0, 1), 0): None, (ORIG, "grey", (0, 1), 1): 0,
                                   (ORIG, "grey", (0, 1), 2): 1,
                                   (ORIG, "grey", (0, 2), 0): 1, (ORIG, "grey", (0, 2), 1): 0})
        self.assertEqual(st.label(uid, ORIG, "grey", (0, 1)), 0)   # first parsed among repeats 1, 2
        self.assertEqual(st.label(uid, ORIG, "grey", (0, 2)), 1)   # repeat 0 parsed wins
        self.assertIsNone(st.label(uid, ORIG, "grey", (0, 3)))

    def test_latest_successful_line_is_authoritative(self):
        base = {"window_uid": "u@0", "target": SCORE, "family": "blur", "mask": [3], "repeat": 0}
        lines = [dict(base, cache_status="fresh", S_hi=1.0), dict(base, cache_status="cached", S_hi=2.0),
                 dict(base, cache_status="error"), dict(base, cache_status="skipped_budget")]
        auth = v.build_authoritative(lines)
        self.assertEqual(auth[("u@0", SCORE, "blur", (3,), 0)]["S_hi"], 2.0)

    def test_development_by_protocol_name(self):
        inp = {"window_uid": "RoadAccidents127_x264@2240", "video_id": "RoadAccidents127_x264",
               "development": False}
        self.assertTrue(v.compute_input(inp, store_from(inp["window_uid"]))["development"])

    def test_native_ranking_uses_logit_margin_and_q(self):
        uid = "Normal_Videos_100_x264@640"
        s_single = [-1.0, -0.75, -0.5, -1.25, -1.0, -1.0, -1.0, -1.0]
        scores = {("none", ()): (-1.0, -1.0)}
        for i in SLOTS:
            scores[("blur", (i,))] = (-1.0 + 0.125 * i, s_single[i])           # S_hi order differs from S order
        st = store_from(uid, scores, {(ORIG, "none", (), 0): 0})
        c = v.compute_input({"window_uid": uid, "video_id": "Normal_Videos_100_x264"}, st)
        nat = c["rankings"]["blur_native"]
        # q = -1: effect_i = -(S(x) - S(x_i)) = S(x_i) - S(x)
        self.assertEqual([nat["effects"][str(i)] for i in SLOTS], [s + 1.0 for s in s_single])
        self.assertEqual(nat["order"], [2, 1, 0, 4, 5, 6, 7, 3])
        self.assertEqual(c["rankings"]["blur"]["order"], [7, 6, 5, 4, 3, 2, 1, 0])     # S_hi effects 0.125 * i

    def test_clipped_boundary_is_strict(self):
        # a + 150 == length and a - 150 == 0 are full-length windows (PROTOCOL 3.1 eligibility uses <=, >=)
        run_dir = build_run(self.tmp, [flip_spec("Arrest001_x264", 1600), flip_spec("Arrest002_x264", 160)])
        p = os.path.join(run_dir, "frozen", "manifest.json")
        with open(p) as fh:
            man = json.load(fh)
        for vd in man["videos"]:
            vd["length"] = 1750 if vd["video_id"] == "Arrest001_x264" else 310
        man["windows"] = [{"window_record_id": "w1", "video_id": "Arrest001_x264", "anchor": 1600,
                           "window": [1450.0, 1750], "frame_indices": frames_for(1600), "clipped": False}]
        with open(p, "w") as fh:
            json.dump(man, fh)
        ck = v.check_window_arithmetic(v.load_run(run_dir))
        self.assertEqual(ck.status(), "passed", ck.discrepancies)

    def test_flips_are_relative_to_L0_for_normal_target(self):
        u = [-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]                      # q = -1: effects 3, 2, 1, 0, ...
        sp = Spec("Normal_Videos_100_x264", 640, additive(-1.0, u),
                  gen_const(0, 0, {(0,): 1, (0, 1): 1}), category="Normal", strata=("normal",))
        run_dir = build_run(self.tmp, [sp])
        _, _, _, by_uid, _ = recompute(run_dir)
        c = by_uid[sp.uid]
        self.assertEqual(c["rankings"]["blur"]["order"], list(SLOTS))
        traj = {(r["mode"], r["k"]): r for r in c["trajectories"]}
        self.assertEqual((traj[("fixed", 1)]["label"], traj[("fixed", 1)]["flip"]), (1, 1))
        self.assertEqual((traj[("fixed", 3)]["label"], traj[("fixed", 3)]["flip"]), (0, 0))
        pairs = {(r["family"], r["pair"]): r for r in c["pairs"]}
        self.assertEqual(pairs[("grey", (0, 1))]["flip"], 1)
        self.assertEqual(pairs[("grey", (2, 3))]["flip"], 0)

    def test_pair_nonadditivity_uses_same_family_singles(self):
        wb = [3.0, -1.0, 0.5, 2.0, 0.0, 1.5, -2.0, 4.0]
        wg = [0.25, 1.0, -0.5, 0.0, 2.0, -1.0, 0.75, 0.5]
        sp = Spec("Burglary005_x264", 800,
                  lambda fam, mask: 2.5 - sum((wg if fam == "grey" else wb)[i] for i in mask), gen_const(1),
                  category="Burglary")
        run_dir = build_run(self.tmp, [sp])
        _, _, _, by_uid, _ = recompute(run_dir)
        for row in by_uid[sp.uid]["pairs"]:
            i, j = row["pair"]
            w = wg if row["family"] == "grey" else wb
            self.assertEqual(row["D"], w[i] + w[j])
            self.assertEqual(row["J"], 0.0, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)

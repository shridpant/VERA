#!/usr/bin/env python3
"""validate3b.py - independent validator for experiment03B.

Written from experiments/experiment03B/PROTOCOL.md (v1.0) and RECORDS.md (v1.0) only. It does not
import or read the runner or the analyzer. Standard library only.

    python3 -I -B validate3b.py --run-dir <runs/run_id> [--out <file>]
                                [--exp02-run-dir <experiments/experiment02/runs/run_002>]

From the raw records (inputs.json, requests/*.jsonl, frozen/config.json, and frozen/manifest.json when
present) it rebuilds the authoritative values, the input-identity checks, the fixed and adaptive
rankings, every prefix, d_w, e_w, the primary estimate with its video-cluster bootstrap, the worst-case
bounds and the declared sensitivity estimates. It compares them with derived/rankings.jsonl and the
analyzer outputs in results/ when those files exist, and reproduces experiment02's erratum E1 when
--exp02-run-dir is given.

Output: validation.json (default <run-dir>/validation/validation.json).
Exit status: 0 every check passed; 1 at least one check failed; 3 no check failed but at least one
was skipped (a skipped check is never reported as passed); 2 usage error.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
from itertools import combinations

VALIDATOR_VERSION = "validate3b 1.0"
PROTOCOL_VERSION = "1.0"

SLOTS = tuple(range(8))
PAIRS = tuple(combinations(SLOTS, 2))  # the 28 pairs, lexicographic
N_PAIRS = len(PAIRS)

SCORE = "label_only_score"
ORIG = "original_generation"
NOSOUND = "nosound_generation"
TARGETS = (ORIG, NOSOUND, SCORE)
GEN_TARGETS = (ORIG, NOSOUND)
FAMILIES = ("none", "blur", "grey")
PHASES = ("baseline", "fixed", "adaptive", "evaluation")
SUCCESS = ("fresh", "cached", "prior_cache")
CACHE_STATUSES = SUCCESS + ("error", "skipped_budget")
SCORE_FIELDS = ("S", "S_hi", "logit_0", "logit_1", "hi_logit_0", "hi_logit_1", "S_resolution",
                "S_hi_rounds_to_model_logits", "candidate_mass", "finite")
GEN_FIELDS = ("raw_response", "label", "parse_status")
RESULT_FIELDS = {SCORE: SCORE_FIELDS, ORIG: GEN_FIELDS, NOSOUND: GEN_FIELDS}
ALL_RESULT_FIELDS = SCORE_FIELDS + GEN_FIELDS
LINE_FIELDS = ("request_id", "window_uid", "phase", "purpose", "target", "family", "mask", "repeat",
               "cache_status", "attempt")
SUCCESS_LINE_FIELDS = ("result_key", "sequence_key", "slot_tensor_sha256", "slot_identity_ok")
INPUT_FIELDS = ("window_uid", "video_id", "category", "anchor", "frame_indices", "frame_sha256",
                "video_sha256", "strata", "window_record_ids", "development", "metadata_flags",
                "annotated_slots", "clipped", "status")
STRATA = ("event", "negative", "normal")
DEVELOPMENT_VIDEOS = ("RoadAccidents127_x264", "Normal_Videos_018_x264")
MAX_ERROR_ATTEMPTS = 3  # one attempt plus at most two retries per request (PROTOCOL 6)
MAX_BUDGET = 4

FLOAT_TOL = 1e-9
DEFAULT_SEED = 0
DEFAULT_B = 10000
CI_QUANTILES = (0.025, 0.975)
N_STABILITY = 4
MAX_EXAMPLES = 20

# VERA arithmetic (setup/scripts/vera_pipeline.py: anchors, window_for_anchor)
VERA_SAMPLING_RATE = 16
VERA_FPS = 30
VERA_T_WINDOW = 10
VERA_SNIPPET_LEN = 8

# experiment02 erratum E1 (168 effect rows; 109 have |Delta| < S_resolution; 36 of those change native S)
E1_EXPECTED = {"n_rows": 168, "n_substep": 109, "n_substep_bf16_changed": 36}
E2_RESULT_TOL = 1e-6  # temporal_effects.csv is printed with 6 decimals
E1_S_NOISE_TOL = 1e-6  # stored S = log p('1') - log p('0') differs from logit_1 - logit_0 by float noise only

SENSITIVITIES = ("tie_free", "scorer_agreement", "with_development", "metadata_clean")


# ----------------------------------------------------------------------------------------- utilities

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v):
    """A finite number, else None (non-finite readouts are treated as missing)."""
    if _is_num(v) and math.isfinite(v):
        return v
    return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def f32(x):
    """Round a Python float to IEEE float32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    """Round a float32 value to bfloat16 (round to nearest, ties to even)."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_mask(m):
    """RECORDS mask: a list of distinct slots sorted ascending. Returns a tuple or None if invalid."""
    if not isinstance(m, list):
        return None
    if not all(_is_int(s) and 0 <= s <= 7 for s in m):
        return None
    if len(set(m)) != len(m) or list(m) != sorted(m):
        return None
    return tuple(m)


def float_eq(a, b, tol=FLOAT_TOL):
    if a is None and b is None:
        return True
    if not (_is_num(a) and _is_num(b)):
        return False
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def exact_eq(a, b):
    """Exact equality; a bool never equals a number."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if _is_num(a) and _is_num(b):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(exact_eq(x, y) for x, y in zip(a, b))
    return a == b


def _abs_diff(a, b):
    if _is_num(a) and _is_num(b):
        try:
            return abs(a - b)
        except (TypeError, OverflowError):
            return None
    return None


def _jsonable(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return obj


def pp(x):
    return None if x is None else 100.0 * x


def pp_list(xs):
    return None if xs is None else [100.0 * v for v in xs]


# ----------------------------------------------------------------------------------------- checks

class Check:
    """One validation check. A check with no examined item is reported as skipped, never passed."""

    def __init__(self, cid, description):
        self.id = cid
        self.description = description
        self.n_checked = 0
        self.discrepancies = []
        self.n_discrepancies = 0
        self.skip_reason = None
        self.notes = {}

    def tick(self, n=1):
        self.n_checked += n

    def fail(self, **example):
        self.n_discrepancies += 1
        if len(self.discrepancies) < MAX_EXAMPLES:
            self.discrepancies.append(_jsonable(example))

    def skip(self, reason):
        self.skip_reason = reason

    def status(self):
        if self.n_discrepancies:
            return "failed"
        if self.skip_reason is not None:
            return "skipped"
        if self.n_checked == 0:
            return "skipped"
        return "passed"

    def to_json(self):
        st = self.status()
        out = {"id": self.id, "description": self.description,
               "passed": None if st == "skipped" else (st == "passed"),
               "status": st, "n_checked": self.n_checked,
               "discrepancies": self.discrepancies, "n_discrepancies": self.n_discrepancies}
        if st == "skipped":
            out["skip_reason"] = self.skip_reason or "nothing to check"
        if self.notes:
            out["notes"] = _jsonable(self.notes)
        return out


def cmp_exact(check, ctx, field, mine, theirs):
    check.tick()
    if not exact_eq(mine, theirs):
        ex = dict(ctx)
        ex.update(field=field, expected=mine, found=theirs)
        d = _abs_diff(mine, theirs)
        if d is not None:
            ex["abs_diff"] = d
        check.fail(**ex)
        return False
    return True


def cmp_float(check, ctx, field, mine, theirs, tol=FLOAT_TOL):
    check.tick()
    if not float_eq(mine, theirs, tol):
        ex = dict(ctx)
        ex.update(field=field, expected=mine, found=theirs)
        d = _abs_diff(mine, theirs)
        if d is not None:
            ex["abs_diff"] = d
        check.fail(**ex)
        return False
    return True


# ----------------------------------------------------------------------------------------- IO

def read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path):
    """Returns [(line_number, obj_or_None, error_or_None)]. Lines are split on '\\n' only."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    out = []
    for i, line in enumerate(text.split("\n"), start=1):
        if line.strip() == "":
            continue
        try:
            obj = json.loads(line)
        except ValueError as exc:
            out.append((i, None, "json: %s" % exc))
            continue
        if not isinstance(obj, dict):
            out.append((i, None, "line is not a JSON object"))
            continue
        out.append((i, obj, None))
    return out


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
        headers = list(rdr.fieldnames or [])
    return headers, rows


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_col(headers, names=(), predicate=None):
    normed = [(h, _norm(h)) for h in headers]
    for name in names:
        n = _norm(name)
        for h, hn in normed:
            if hn == n:
                return h
    if predicate is not None:
        for h, hn in normed:
            if predicate(hn):
                return h
    return None


_BLANK = ("", "nan", "none", "null", "na", "n/a")


def csv_num(s):
    if s is None or s.strip().lower() in _BLANK:
        return None
    try:
        return float(s)
    except ValueError:
        return s  # keep the raw text so the comparison reports it


def csv_int(s):
    if s is None or s.strip().lower() in _BLANK:
        return None
    t = s.strip()
    if t.lower() in ("true", "false"):
        return 1 if t.lower() == "true" else 0
    try:
        return int(t)
    except ValueError:
        try:
            f = float(t)
        except ValueError:
            return t
        return int(f) if f.is_integer() else f


def csv_bool(s):
    if s is None or s.strip().lower() in _BLANK:
        return None
    t = s.strip().lower()
    if t in ("true", "1", "yes"):
        return True
    if t in ("false", "0", "no"):
        return False
    return s


def csv_ints(s):
    if s is None:
        return None
    t = s.strip()
    if t.lower() in _BLANK[1:]:
        return None
    return [int(x) for x in re.findall(r"\d+", t)]


# ----------------------------------------------------------------------------------------- run data

class RunData:
    def __init__(self, run_dir):
        self.run_dir = os.path.abspath(run_dir)
        self.load_problems = []   # (where, message)
        self.provenance = {}
        self.inputs = []
        self.inputs_by_uid = {}
        self.config = None
        self.manifest = None
        self.lines = []           # every parsed request line, in global order
        self.line_parse_errors = []
        self.request_files = []

    def path(self, *parts):
        return os.path.join(self.run_dir, *parts)

    def note_file(self, rel):
        p = self.path(rel)
        if os.path.isfile(p):
            self.provenance[rel] = sha256_file(p)


def load_run(run_dir):
    run = RunData(run_dir)
    # inputs.json
    p = run.path("inputs.json")
    if not os.path.isfile(p):
        run.load_problems.append(("inputs.json", "missing"))
    else:
        run.note_file("inputs.json")
        try:
            obj = read_json(p)
            inputs = obj.get("inputs") if isinstance(obj, dict) else None
            if not isinstance(inputs, list):
                run.load_problems.append(("inputs.json", "no 'inputs' list"))
            else:
                run.inputs = [x for x in inputs if isinstance(x, dict)]
                if len(run.inputs) != len(inputs):
                    run.load_problems.append(("inputs.json", "non-object entries in 'inputs'"))
        except ValueError as exc:
            run.load_problems.append(("inputs.json", "json: %s" % exc))
    for inp in run.inputs:
        uid = inp.get("window_uid")
        if isinstance(uid, str) and uid not in run.inputs_by_uid:
            run.inputs_by_uid[uid] = inp
    # frozen/config.json
    p = run.path("frozen", "config.json")
    if not os.path.isfile(p):
        run.load_problems.append(("frozen/config.json", "missing"))
    else:
        run.note_file("frozen/config.json")
        try:
            run.config = read_json(p)
        except ValueError as exc:
            run.load_problems.append(("frozen/config.json", "json: %s" % exc))
    # frozen/manifest.json (optional for this validator; used for the VERA window arithmetic check)
    p = run.path("frozen", "manifest.json")
    if os.path.isfile(p):
        run.note_file("frozen/manifest.json")
        try:
            run.manifest = read_json(p)
        except ValueError as exc:
            run.load_problems.append(("frozen/manifest.json", "json: %s" % exc))
    # requests/*.jsonl (global order: file name, then line number)
    rdir = run.path("requests")
    if not os.path.isdir(rdir):
        run.load_problems.append(("requests/", "missing"))
    else:
        names = sorted(n for n in os.listdir(rdir) if n.endswith(".jsonl"))
        if not names:
            run.load_problems.append(("requests/", "no *.jsonl files"))
        for name in names:
            rel = "requests/" + name
            run.request_files.append(rel)
            run.note_file(rel)
            for lineno, obj, err in read_jsonl(os.path.join(rdir, name)):
                src = "%s:%d" % (rel, lineno)
                if err:
                    run.line_parse_errors.append((src, err))
                    continue
                obj["_src"] = src
                obj["_seq"] = len(run.lines)
                run.lines.append(obj)
    return run


def line_key(rec):
    """(window_uid, target, family, mask, repeat) or None if any part is invalid."""
    uid = rec.get("window_uid")
    target = rec.get("target")
    family = rec.get("family")
    mask = canon_mask(rec.get("mask"))
    rep = rec.get("repeat")
    if not isinstance(uid, str) or target not in TARGETS or family not in FAMILIES or mask is None:
        return None
    if not _is_int(rep) or rep not in (0, 1, 2):
        return None
    if (family == "none") != (mask == ()):
        return None
    return (uid, target, family, mask, rep)


def is_success(rec):
    return rec.get("cache_status") in SUCCESS


def build_authoritative(lines):
    """RECORDS: the value of a tuple is the latest successful line (global order)."""
    auth = {}
    for rec in lines:
        if not is_success(rec):
            continue
        k = line_key(rec)
        if k is None:
            continue
        auth[k] = rec
    return auth


def is_parsed(rec):
    lab = rec.get("label")
    return rec.get("parse_status") == "ok" and _is_int(lab) and lab in (0, 1)


class Store:
    """Read access to authoritative values."""

    def __init__(self, auth):
        self.auth = auth

    def rec(self, uid, target, family, mask, repeat=0):
        return self.auth.get((uid, target, family, tuple(mask), repeat))

    def score(self, uid, family, mask, field="S_hi"):
        r = self.rec(uid, SCORE, family, mask, 0)
        if r is None:
            return None
        if field == "S_native":  # protocol v1.1: native margin = exact BF16 logit difference (logit_1 - logit_0)
            l0, l1 = _num(r.get("logit_0")), _num(r.get("logit_1"))
            return None if l0 is None or l1 is None else l1 - l0
        return _num(r.get(field))

    def label(self, uid, target, family, mask):
        """repeat 0 if parsed; otherwise the first parsed label among repeats 1 and 2; else missing."""
        for rep in (0, 1, 2):
            r = self.rec(uid, target, family, mask, rep)
            if r is not None and is_parsed(r):
                return r["label"]
        return None


# ----------------------------------------------------------------------------------------- core math

def vera_frames(anchor, length):
    """np.arange(s, e, (e - s)/8).astype(int) emulated with the standard library (numpy fill rule)."""
    half = 1 / 2 * VERA_FPS * VERA_T_WINDOW
    s = max(anchor - half, 0)
    e = min(anchor + half, length)
    step = (e - s) / VERA_SNIPPET_LEN
    if step <= 0:
        return [], s, e
    n = math.ceil((e - s) / step)
    v0 = s + 0.0
    v1 = s + step
    delta = v1 - v0
    vals = [v0, v1][:n] + [v0 + i * delta for i in range(2, n)]
    return [int(v) for v in vals], s, e


def fixed_ranking(q, base, singles):
    """RECORDS RANKING from a baseline value and the eight single-slot values on one readout/family."""
    missing = []
    if base is None:
        missing.append("baseline")
    missing.extend(i for i in SLOTS if singles.get(i) is None)
    if missing:
        return {"status": "incomplete", "missing": missing}
    eff = [q * (base - singles[i]) for i in SLOTS]
    order = sorted(SLOTS, key=lambda i: (-eff[i], i))
    groups = []
    for i in order:
        if groups and eff[groups[-1][0]] == eff[i]:
            groups[-1].append(i)
        else:
            groups.append([i])
    return {"effects": {str(i): eff[i] for i in SLOTS},
            "order": list(order),
            "ties": [sorted(g) for g in groups if len(g) > 1],
            "tie_spans_cut": {str(k): eff[order[k - 1]] == eff[order[k]] for k in (1, 2, 3, 4)},
            "top2_margin": eff[order[1]] - eff[order[2]],
            "all_tied": all(v == eff[0] for v in eff),
            "all_nonpositive": all(v <= 0 for v in eff)}


def ranking_complete(r):
    return isinstance(r, dict) and r.get("status") != "incomplete" and "order" in r


def adaptive_trajectory(q, base, blur_score, max_budget=MAX_BUDGET):
    """PROTOCOL 5.2 from blur S_hi only. blur_score(mask_tuple) -> S_hi or None."""
    states, order = [], []
    for step in range(1, max_budget + 1):
        masked = sorted(order)
        state_score = base if step == 1 else blur_score(tuple(masked))
        if state_score is None:
            return {"status": "incomplete", "failed_step": step,
                    "missing": [masked] if step > 1 else [[]], "states": states, "order": order}
        cands, missing = [], []
        for j in SLOTS:
            if j in order:
                continue
            m = tuple(sorted(masked + [j]))
            s = blur_score(m)
            if s is None:
                missing.append(list(m))
                continue
            cands.append({"slot": j, "mask": list(m), "score": s, "effect": q * (state_score - s)})
        if missing:
            return {"status": "incomplete", "failed_step": step, "missing": missing,
                    "states": states, "order": order}
        best = max(c["effect"] for c in cands)
        tied = sorted(c["slot"] for c in cands if c["effect"] == best)
        sel = tied[0]
        states.append({"step": step, "masked": masked, "state_score": state_score,
                       "candidates": cands, "selected": sel, "tied_max": tied[1:]})
        order.append(sel)
    return {"family": "blur", "readout": "S_hi", "max_budget": max_budget, "states": states,
            "order": order}


def kendall_tau_b(x, y):
    """Kendall tau_b with the standard tie correction; None when undefined (a constant vector)."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    nc = nd = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0:
                tx += 1
            if dy == 0:
                ty += 1
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                nc += 1
            else:
                nd += 1
    n0 = n * (n - 1) // 2
    denom = math.sqrt((n0 - tx) * (n0 - ty))
    if denom == 0:
        return None
    return (nc - nd) / denom


def aggregate(entries):
    """Video-cluster estimator. entries: [(video_id, value_or_None)] in input order.
    Returns (videos sorted by video_id, video means, mean of video means or None)."""
    by_video = {}
    for vid, val in entries:
        if val is None:
            continue
        by_video.setdefault(vid, []).append(val)
    videos = sorted(by_video)
    means = [sum(by_video[v]) / len(by_video[v]) for v in videos]
    est = sum(means) / len(means) if means else None
    return videos, means, est


def percentile_sorted(vals, qtl):
    pos = (len(vals) - 1) * qtl
    lo = math.floor(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def bootstrap_ci(means, seed, B):
    """PROTOCOL 7.1: rng = random.Random(seed); B resamples of V indices with rng.randrange(V)."""
    V = len(means)
    if V == 0 or B < 1:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(B):
        idx = [rng.randrange(V) for _ in range(V)]
        stats.append(sum(means[i] for i in idx) / V)
    stats.sort()
    return [percentile_sorted(stats, CI_QUANTILES[0]), percentile_sorted(stats, CI_QUANTILES[1])]


def bootstrap_params(config):
    """Seed and B from frozen/config.json (bootstrap-related keys preferred), else protocol defaults."""
    seeds, bs = [], []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                p = path + [str(k)]
                kl = str(k).lower()
                parents = [x.lower() for x in path]
                in_boot = any("bootstrap" in x for x in p[:-1]) or "bootstrap" in kl
                in_run = len(path) == 0 or any(x in ("analysis", "execution", "run") for x in parents)
                if _is_int(v):
                    if kl == "bootstrap_seed" or (kl == "seed" and in_boot):
                        seeds.append((0, ".".join(p), v))
                    elif kl in ("execution_seed", "seed", "analysis_seed") and in_run:
                        seeds.append((1, ".".join(p), v))
                    if kl in ("bootstrap_b", "bootstrap_resamples", "n_bootstrap", "bootstrap_n",
                              "n_bootstrap_resamples") or \
                            (kl in ("b", "n_resamples", "resamples", "n", "n_boot") and in_boot):
                        if v > 0:
                            bs.append((0, ".".join(p), v))
                walk(v, p)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])

    if isinstance(config, dict):
        walk(config, [])
    info = {}

    def pick(found, default, name):
        if not found:
            info[name + "_source"] = "protocol default"
            return default
        best = min(f[0] for f in found)
        cands = [f for f in found if f[0] == best]
        vals = sorted(set(f[2] for f in cands))
        info[name + "_source"] = "frozen/config.json:" + cands[0][1]
        if len(vals) > 1:
            info[name + "_ambiguous"] = [[f[1], f[2]] for f in cands]
        return cands[0][2]

    seed = pick(seeds, DEFAULT_SEED, "seed")
    B = pick(bs, DEFAULT_B, "B")
    return seed, B, info


# ----------------------------------------------------------------------------------------- per input

def compute_input(inp, store):
    """Every recomputed per-input quantity, from authoritative records only."""
    uid = inp.get("window_uid")
    c = {"window_uid": uid, "video_id": inp.get("video_id"), "category": inp.get("category"),
         # PROTOCOL 3: the two named videos are development; an inputs.json tag is honoured as well
         "development": inp.get("video_id") in DEVELOPMENT_VIDEOS or inp.get("development") is True,
         "metadata_flags": list(inp.get("metadata_flags") or []),
         "strata": list(inp.get("strata") or []), "status": inp.get("status")}
    L0 = store.label(uid, ORIG, "none", ())
    c["L0"] = L0
    c["nosound_label"] = store.label(uid, NOSOUND, "none", ())
    b_hi = store.score(uid, "none", (), "S_hi")
    b_s = store.score(uid, "none", (), "S_native")
    c["baseline_S_hi"], c["baseline_S"] = b_hi, b_s
    c["baseline_native_class"] = None if b_s is None else (1 if b_s > 0 else (0 if b_s < 0 else "undecided"))
    c["label_available"] = L0 is not None
    for k in ("q", "rankings", "adaptive", "prefixes", "T_w", "pair_labels", "F", "n_pair_labels",
              "F_top2", "F_random_mean", "d_w", "d_lower", "d_upper", "e_w", "tau_blur_grey",
              "tau_native_hi", "overlap_blur_grey", "overlap_native_hi"):
        c[k] = None
    c["pairs"], c["trajectories"] = [], []
    if L0 is None:
        return c
    q = 1 if L0 == 1 else -1
    c["q"] = q
    blur_hi = {i: store.score(uid, "blur", (i,), "S_hi") for i in SLOTS}
    grey_hi = {i: store.score(uid, "grey", (i,), "S_hi") for i in SLOTS}
    blur_s = {i: store.score(uid, "blur", (i,), "S_native") for i in SLOTS}
    rk = {"blur": fixed_ranking(q, b_hi, blur_hi),
          "grey": fixed_ranking(q, b_hi, grey_hi),
          "blur_native": fixed_ranking(q, b_s, blur_s)}
    c["rankings"] = rk
    ad = adaptive_trajectory(q, b_hi, lambda m: store.score(uid, "blur", m, "S_hi"))
    c["adaptive"] = ad
    pref = {"fixed": None, "adaptive": None}
    if ranking_complete(rk["blur"]):
        o = rk["blur"]["order"]
        pref["fixed"] = {str(k): sorted(o[:k]) for k in range(1, MAX_BUDGET + 1)}
    if ad.get("status") != "incomplete":
        o = ad["order"]
        pref["adaptive"] = {str(k): sorted(o[:k]) for k in range(1, MAX_BUDGET + 1)}
    c["prefixes"] = pref
    T = tuple(pref["fixed"]["2"]) if pref["fixed"] else None
    c["T_w"] = list(T) if T else None

    # grey-pair original labels and flips F (PROTOCOL 7.1: F = 1 if label 0, 0 if label 1)
    labels = {P: store.label(uid, ORIG, "grey", P) for P in PAIRS}
    F = {P: (None if labels[P] is None else (1 if labels[P] == 0 else 0)) for P in PAIRS}
    c["pair_labels"], c["F"] = labels, F
    avail = [F[P] for P in PAIRS if F[P] is not None]
    c["n_pair_labels"] = len(avail)
    if L0 == 1:
        c["F_top2"] = F[T] if T else None
        c["F_random_mean"] = (sum(avail) / len(avail)) if avail else None
        if T is not None and len(avail) == N_PAIRS and F[T] is not None:
            c["d_w"] = F[T] - sum(F[P] for P in PAIRS) / N_PAIRS
        # worst-case bounds: lower = missing selected 0, missing comparator 1; upper = the reverse
        sel_lo = F[T] if (T and F[T] is not None) else 0
        sel_hi = F[T] if (T and F[T] is not None) else 1
        c["d_lower"] = sel_lo - sum((F[P] if F[P] is not None else 1) for P in PAIRS) / N_PAIRS
        c["d_upper"] = sel_hi - sum((F[P] if F[P] is not None else 0) for P in PAIRS) / N_PAIRS

    # e_w (secondary 1)
    grey_pair_hi = {P: store.score(uid, "grey", P, "S_hi") for P in PAIRS}
    if b_hi is not None and T is not None and all(grey_pair_hi[P] is not None for P in PAIRS):
        sel = q * (b_hi - grey_pair_hi[T])
        comp = sum(q * (b_hi - grey_pair_hi[P]) for P in PAIRS) / N_PAIRS
        c["e_w"] = sel - comp

    # agreement measures
    if ranking_complete(rk["blur"]) and ranking_complete(rk["grey"]):
        xb = [rk["blur"]["effects"][str(i)] for i in SLOTS]
        xg = [rk["grey"]["effects"][str(i)] for i in SLOTS]
        c["tau_blur_grey"] = kendall_tau_b(xb, xg)
        c["overlap_blur_grey"] = len(set(rk["blur"]["order"][:2]) & set(rk["grey"]["order"][:2]))
    if ranking_complete(rk["blur"]) and ranking_complete(rk["blur_native"]):
        xb = [rk["blur"]["effects"][str(i)] for i in SLOTS]
        xn = [rk["blur_native"]["effects"][str(i)] for i in SLOTS]
        c["tau_native_hi"] = kendall_tau_b(xn, xb)
        c["overlap_native_hi"] = len(set(rk["blur"]["order"][:2]) & set(rk["blur_native"]["order"][:2]))

    # pairs.csv rows
    for fam in ("blur", "grey"):
        single_D = {}
        for i in SLOTS:
            s = store.score(uid, fam, (i,), "S_hi")
            single_D[i] = None if (s is None or b_hi is None) else q * (b_hi - s)
        for P in PAIRS:
            s_hi = store.score(uid, fam, P, "S_hi")
            s_n = store.score(uid, fam, P, "S_native")
            D = None if (s_hi is None or b_hi is None) else q * (b_hi - s_hi)
            J = None
            if D is not None and single_D[P[0]] is not None and single_D[P[1]] is not None:
                J = D - single_D[P[0]] - single_D[P[1]]
            row = {"family": fam, "pair": P, "S_hi": s_hi, "S": s_n, "D": D, "J": J,
                   "label": None, "flip": None}
            if fam == "grey":
                lab = labels[P]
                row["label"] = lab
                row["flip"] = None if lab is None else int(lab != L0)
            c["pairs"].append(row)

    # trajectories.csv rows (prefixes are evaluated under grey)
    for mode in ("fixed", "adaptive"):
        if not pref[mode]:
            continue
        for k in range(1, MAX_BUDGET + 1):
            m = tuple(pref[mode][str(k)])
            lab = store.label(uid, ORIG, "grey", m)
            s_hi = store.score(uid, "grey", m, "S_hi")
            s_n = store.score(uid, "grey", m, "S_native")
            c["trajectories"].append({
                "mode": mode, "k": k, "mask": list(m), "label": lab,
                "flip": None if lab is None else int(lab != L0),
                "q_dS_hi": None if (s_hi is None or b_hi is None) else q * (b_hi - s_hi),
                "q_dS": None if (s_n is None or b_s is None) else q * (b_s - s_n)})
    return c


def rankings_record(c):
    """The derived/rankings.jsonl record this validator expects for a label-available input."""
    if not c.get("label_available"):
        return None
    return {"window_uid": c["window_uid"], "L0": c["L0"], "q": c["q"],
            "baseline": {"S_hi": c["baseline_S_hi"], "S": c["baseline_S"]},
            "fixed": dict(c["rankings"]), "adaptive": c["adaptive"],
            "prefixes": c["prefixes"], "T_w": c["T_w"]}


# ----------------------------------------------------------------------------------------- populations

def primary_population(calcs, include_development=False):
    """PROTOCOL 3.3: unique inputs, not development, parsed L0 == 1, any stratum."""
    return [c for c in calcs if c["L0"] == 1 and (include_development or not c["development"])]


def sensitivity_population(name, calcs):
    prim = primary_population(calcs)
    if name == "tie_free":
        return [c for c in prim
                if not (ranking_complete(c["rankings"]["blur"]) and c["rankings"]["blur"]["tie_spans_cut"]["2"])]
    if name == "scorer_agreement":
        return [c for c in prim if c["baseline_native_class"] == c["L0"]]
    if name == "with_development":
        return primary_population(calcs, include_development=True)
    if name == "metadata_clean":
        return [c for c in prim if not c["metadata_flags"]]
    raise KeyError(name)


def estimate_block(pop, seed, B, stability=True):
    videos, means, est = aggregate([(c["video_id"], c["d_w"]) for c in pop])
    ci = bootstrap_ci(means, seed, B)
    out = {"estimate_pp": pp(est), "ci95_pp": pp_list(ci),
           "bootstrap": {"seed": seed, "B": B},
           "n_videos": len(videos),
           "n_videos_population": len(set(c["video_id"] for c in pop)),
           "n_inputs": len(pop),
           "n_inputs_missing_dw": sum(1 for c in pop if c["d_w"] is None),
           "video_means_pp": {v: 100.0 * m for v, m in zip(videos, means)},
           "d_w": {c["window_uid"]: c["d_w"] for c in pop}}
    if stability:
        stab = [{"seed": seed + i, "ci95_pp": pp_list(bootstrap_ci(means, seed + i, B))}
                for i in range(1, N_STABILITY + 1)]
        out["bootstrap"]["stability"] = stab
        allci = [out["ci95_pp"]] + [s["ci95_pp"] for s in stab]
        if all(x is not None for x in allci):
            out["bootstrap"]["stability_range_pp"] = {
                "lower": [min(x[0] for x in allci), max(x[0] for x in allci)],
                "upper": [min(x[1] for x in allci), max(x[1] for x in allci)]}
    _, _, lo = aggregate([(c["video_id"], c["d_lower"]) for c in pop])
    _, _, hi = aggregate([(c["video_id"], c["d_upper"]) for c in pop])
    out["bounds_pp"] = [pp(lo), pp(hi)]
    return out


def compute_primary(calcs, seed, B):
    prim = estimate_block(primary_population(calcs), seed, B, stability=True)
    prim["sensitivity"] = {name: estimate_block(sensitivity_population(name, calcs), seed, B, stability=False)
                           for name in SENSITIVITIES}
    return prim


# ----------------------------------------------------------------------------------------- record checks

def check_run_files(run):
    ck = Check("run_files", "required run files exist and parse (inputs.json, frozen/config.json, requests/*.jsonl)")
    ck.tick(3)
    for where, msg in run.load_problems:
        ck.fail(file=where, problem=msg)
    for src, err in run.line_parse_errors:
        ck.fail(file=src, problem=err)
    ck.notes["request_files"] = run.request_files
    return ck


def check_records_schema(run):
    ck = Check("records_schema", "every request line follows RECORDS (fields, enums, canonical masks, family/mask, "
                                 "known window_uid, result fields on successes, error lines carry 'error' only)")
    for rec in run.lines:
        ck.tick()
        src = rec["_src"]
        probs = [f for f in LINE_FIELDS if f not in rec]
        if probs:
            ck.fail(src=src, problem="missing fields", fields=probs)
            continue
        if rec["target"] not in TARGETS:
            ck.fail(src=src, problem="unknown target", value=rec["target"])
        if rec["family"] not in FAMILIES:
            ck.fail(src=src, problem="unknown family", value=rec["family"])
        if rec["phase"] not in PHASES:
            ck.fail(src=src, problem="unknown phase", value=rec["phase"])
        if rec["cache_status"] not in CACHE_STATUSES:
            ck.fail(src=src, problem="unknown cache_status", value=rec["cache_status"])
        m = canon_mask(rec["mask"])
        if m is None:
            ck.fail(src=src, problem="mask is not a sorted list of distinct slots 0..7", value=rec["mask"])
        elif (rec["family"] == "none") != (m == ()):
            ck.fail(src=src, problem="family 'none' must go with mask [] and only with it",
                    family=rec["family"], mask=rec["mask"])
        if not (_is_int(rec["repeat"]) and rec["repeat"] in (0, 1, 2)):
            ck.fail(src=src, problem="repeat not in {0,1,2}", value=rec["repeat"])
        if not (_is_int(rec["attempt"]) and rec["attempt"] >= 1):
            ck.fail(src=src, problem="attempt is not a positive integer", value=rec["attempt"])
        if rec["window_uid"] not in run.inputs_by_uid:
            ck.fail(src=src, problem="window_uid not in inputs.json", value=rec["window_uid"])
        fields = RESULT_FIELDS.get(rec["target"], ())
        if is_success(rec):
            miss = [f for f in fields if f not in rec] + [f for f in SUCCESS_LINE_FIELDS if f not in rec]
            if miss:
                ck.fail(src=src, problem="successful line lacks fields", fields=miss)
            if rec["target"] == SCORE:
                bad = [f for f in ("S", "S_hi", "logit_0", "logit_1", "hi_logit_0", "hi_logit_1", "S_resolution")
                       if f in rec and not _is_num(rec[f])]
                if bad:
                    ck.fail(src=src, problem="non-numeric score fields", fields=bad)
            else:
                lab = rec.get("label")
                if not (lab is None or (_is_int(lab) and lab in (0, 1))):
                    ck.fail(src=src, problem="label not in {0, 1, null}", value=lab)
                if (rec.get("parse_status") == "ok") != (_is_int(lab) and lab in (0, 1)):
                    ck.fail(src=src, problem="parse_status 'ok' must go with a 0/1 label and only with it",
                            parse_status=rec.get("parse_status"), label=lab)
        else:
            present = [f for f in ALL_RESULT_FIELDS if f in rec]
            if present:
                ck.fail(src=src, problem="unsuccessful line carries result fields", fields=present)
            if rec["cache_status"] == "error" and not isinstance(rec.get("error"), str):
                ck.fail(src=src, problem="error line without an 'error' string")
    return ck


def check_result_key_consistency(run):
    ck = Check("result_key_consistency", "all successful lines sharing a result_key carry identical result fields "
                                         "(and the same target)")
    groups = {}
    for rec in run.lines:
        if is_success(rec) and isinstance(rec.get("result_key"), str):
            groups.setdefault(rec["result_key"], []).append(rec)
    for rk, recs in groups.items():
        ck.tick()
        first = recs[0]
        for other in recs[1:]:
            if other.get("target") != first.get("target"):
                ck.fail(result_key=rk, problem="different targets share a result_key",
                        lines=[first["_src"], other["_src"]])
                continue
            for f in RESULT_FIELDS.get(first.get("target"), ALL_RESULT_FIELDS):
                a, b = first.get(f, "<absent>"), other.get(f, "<absent>")
                if not exact_eq(a, b):
                    ck.fail(result_key=rk, field=f, lines=[first["_src"], other["_src"]], values=[a, b])
    ck.notes["n_result_keys"] = len(groups)
    # informational: tuples whose successful lines disagree (the latest line is authoritative)
    by_tuple = {}
    for rec in run.lines:
        k = line_key(rec)
        if k is not None and is_success(rec):
            by_tuple.setdefault(k, []).append(rec)
    divergent = []
    for k, recs in by_tuple.items():
        fields = RESULT_FIELDS[k[1]]
        vals = set(json.dumps([r.get(f) for f in fields], sort_keys=True) for r in recs)
        if len(vals) > 1:
            divergent.append({"tuple": [k[0], k[1], k[2], list(k[3]), k[4]],
                              "lines": [r["_src"] for r in recs], "used": recs[-1]["_src"]})
    ck.notes["tuples_with_divergent_successful_lines"] = len(divergent)
    if divergent:
        ck.notes["divergent_examples"] = divergent[:MAX_EXAMPLES]
    return ck


def check_repeat_rule(run, auth):
    ck = Check("repeat_rule", "repeats 1/2 exist only for generations, only after an unparsed repeat r-1 of the "
                              "identical input, and an unparsed repeat 0/1 is always regenerated (repeat r+1 logged); "
                              "label = repeat 0 if parsed, else first parsed of repeats 1, 2")
    nondet = []
    keys = set(k for k in auth)
    logged = set(k for k in (line_key(r) for r in run.lines) if k is not None)
    for (uid, target, family, mask, rep) in sorted(keys, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        ck.tick()
        # PROTOCOL 6: a parse failure is regenerated up to two more times (repeat 1, 2)
        if target in GEN_TARGETS and rep < 2 and not is_parsed(auth[(uid, target, family, mask, rep)]) \
                and (uid, target, family, mask, rep + 1) not in logged:
            ck.fail(window_uid=uid, target=target, family=family, mask=list(mask), repeat=rep,
                    problem="unparsed generation not regenerated (no repeat %d line)" % (rep + 1))
        if rep == 0:
            continue
        if target not in GEN_TARGETS:
            ck.fail(window_uid=uid, target=target, family=family, mask=list(mask), repeat=rep,
                    problem="repeat > 0 on a score request")
            continue
        prev = auth.get((uid, target, family, mask, rep - 1))
        if prev is None or is_parsed(prev):
            ck.fail(window_uid=uid, target=target, family=family, mask=list(mask), repeat=rep,
                    problem="repeat without an unparsed predecessor",
                    predecessor="missing" if prev is None else "parsed")
    # non-determinism flag: repeats of one generation input that disagree
    gen = {}
    for (uid, target, family, mask, rep), r in auth.items():
        if target in GEN_TARGETS:
            gen.setdefault((uid, target, family, mask), []).append(r)
    for k, recs in gen.items():
        if len(recs) > 1:
            vals = set(json.dumps([r.get("raw_response"), r.get("label")]) for r in recs)
            if len(vals) > 1:
                nondet.append([k[0], k[1], k[2], list(k[3])])
    ck.notes["non_deterministic_generation_inputs"] = len(nondet)
    if nondet:
        ck.notes["non_deterministic_examples"] = nondet[:MAX_EXAMPLES]
    return ck


def check_retry_limit(run):
    ck = Check("retry_limit", "errors are retried at most twice per request (at most %d error attempts per request_id, "
                              "and at most %d attempts that follow an error attempt)"
               % (MAX_ERROR_ATTEMPTS, MAX_ERROR_ATTEMPTS - 1))
    errs, retries, after_error = {}, {}, {}
    seen = set()
    for rec in run.lines:
        rid = rec.get("request_id")
        if not isinstance(rid, str):
            continue
        seen.add(rid)
        if after_error.get(rid):
            retries.setdefault(rid, []).append(rec["_src"])  # an attempt made after an error is a retry
        after_error[rid] = rec.get("cache_status") == "error"
        if after_error[rid]:
            errs.setdefault(rid, []).append(rec["_src"])
    for rid in sorted(seen):
        ck.tick()
        n = len(errs.get(rid, []))
        n_retry = len(retries.get(rid, []))
        if n > MAX_ERROR_ATTEMPTS or n_retry > MAX_ERROR_ATTEMPTS - 1:
            ck.fail(request_id=rid, n_error_attempts=n, n_retries=n_retry,
                    lines=(errs.get(rid, []) + retries.get(rid, []))[:6])
    return ck


def check_score_rounding(run):
    ck = Check("score_rounding", "every successful score line: finite readouts, S_hi_rounds_to_model_logits is true "
                                 "and equals bf16(hi_logit_k) == logit_k, S_hi is the FP32 difference of the head logits")
    for rec in run.lines:
        if rec.get("target") != SCORE or not is_success(rec):
            continue
        ck.tick()
        src = rec["_src"]
        vals = {f: rec.get(f) for f in ("S", "S_hi", "logit_0", "logit_1", "hi_logit_0", "hi_logit_1")}
        if not all(_num(v) is not None for v in vals.values()):
            ck.fail(src=src, problem="non-finite or missing score readout", values=vals)
            continue
        if rec.get("finite") is not True:
            ck.fail(src=src, problem="finite is not true", value=rec.get("finite"))
        recomputed = bf16(vals["hi_logit_0"]) == vals["logit_0"] and bf16(vals["hi_logit_1"]) == vals["logit_1"]
        flag = rec.get("S_hi_rounds_to_model_logits")
        if flag is not True or recomputed is not True:
            ck.fail(src=src, problem="FP32 head logits do not round to the model's BF16 logits",
                    flag=flag, recomputed=recomputed,
                    hi_logits=[vals["hi_logit_0"], vals["hi_logit_1"]], logits=[vals["logit_0"], vals["logit_1"]])
        diff = vals["hi_logit_1"] - vals["hi_logit_0"]
        if not (vals["S_hi"] == f32(diff) or vals["S_hi"] == diff):
            ck.fail(src=src, problem="S_hi != hi_logit_1 - hi_logit_0", S_hi=vals["S_hi"], difference=diff)
    return ck


def check_inputs(run):
    ck = Check("inputs_identity", "inputs.json: unique window_uid == video_id@anchor, unique (video_id, frame_indices), "
                                  "8 frame indices, 8 frame hashes, strata, development tags, slots in 0..7")
    seen_uid, seen_frames = set(), {}
    for idx, inp in enumerate(run.inputs):
        ck.tick()
        uid = inp.get("window_uid")
        ctx = {"input_index": idx, "window_uid": uid}
        miss = [f for f in INPUT_FIELDS if f not in inp]
        if miss:
            ck.fail(problem="missing input fields", fields=miss, **ctx)
        if not isinstance(uid, str):
            ck.fail(problem="window_uid is not a string", **ctx)
            continue
        if uid in seen_uid:
            ck.fail(problem="duplicate window_uid", **ctx)
        seen_uid.add(uid)
        vid, anchor = inp.get("video_id"), inp.get("anchor")
        if not _is_int(anchor) or anchor < 0 or anchor % VERA_SAMPLING_RATE != 0:
            ck.fail(problem="anchor is not a VERA anchor (non-negative multiple of 16)", anchor=anchor, **ctx)
        if uid != "%s@%s" % (vid, anchor):
            ck.fail(problem="window_uid != '<video_id>@<anchor>'", video_id=vid, anchor=anchor, **ctx)
        fi = inp.get("frame_indices")
        if not (isinstance(fi, list) and len(fi) == 8 and all(_is_int(x) and x >= 0 for x in fi)):
            ck.fail(problem="frame_indices is not 8 non-negative integers", value=fi, **ctx)
        else:
            key = (vid, tuple(fi))
            if key in seen_frames:
                ck.fail(problem="two inputs share (video_id, frame_indices); inputs must be deduplicated",
                        other=seen_frames[key], **ctx)
            seen_frames[key] = uid
            if any(fi[i] > fi[i + 1] for i in range(7)):
                ck.fail(problem="frame_indices not in non-decreasing slot order", value=fi, **ctx)
        fs = inp.get("frame_sha256")
        if not (isinstance(fs, list) and len(fs) == 8 and all(isinstance(x, str) and x for x in fs)):
            ck.fail(problem="frame_sha256 is not 8 strings", **ctx)
        st = inp.get("strata")
        if not (isinstance(st, list) and st and len(set(st)) == len(st) and all(s in STRATA for s in st)):
            ck.fail(problem="strata is not a non-empty deduplicated subset of event/negative/normal", value=st, **ctx)
        dev = inp.get("development")
        if dev is not (vid in DEVELOPMENT_VIDEOS):
            ck.fail(problem="development tag does not match the protocol's development videos",
                    development=dev, video_id=vid, **ctx)
        mf = inp.get("metadata_flags")
        if not (isinstance(mf, list) and all(isinstance(x, str) for x in mf)):
            ck.fail(problem="metadata_flags is not a list of strings", value=mf, **ctx)
        an = inp.get("annotated_slots")
        if not (isinstance(an, list) and all(_is_int(x) and 0 <= x <= 7 for x in an)):
            ck.fail(problem="annotated_slots not a list of slots 0..7", value=an, **ctx)
        if not isinstance(inp.get("clipped"), bool):
            ck.fail(problem="clipped is not a bool", value=inp.get("clipped"), **ctx)
    return ck


def check_window_arithmetic(run):
    ck = Check("window_arithmetic", "frame_indices, window and clipped flag equal VERA's arithmetic for the anchor "
                                    "and the manifest length (inputs.json and frozen/manifest.json windows)")
    man = run.manifest
    if not isinstance(man, dict) or not isinstance(man.get("videos"), list):
        ck.skip("frozen/manifest.json with a 'videos' list is absent")
        return ck
    lengths = {}
    for v in man["videos"]:
        if isinstance(v, dict) and isinstance(v.get("video_id"), str) and _is_int(v.get("length")):
            lengths[v["video_id"]] = v["length"]
    if not lengths:
        ck.skip("frozen/manifest.json lists no video lengths")
        return ck

    def one(where, vid, anchor, frames, window=None, clipped=None):
        ck.tick()
        L = lengths.get(vid)
        if L is None:
            ck.fail(where=where, problem="video not in manifest", video_id=vid)
            return
        if not _is_int(anchor) or not (0 <= anchor < L) or anchor % VERA_SAMPLING_RATE:
            ck.fail(where=where, problem="anchor not in range(0, length, 16)", anchor=anchor, length=L)
            return
        exp, s, e = vera_frames(anchor, L)
        if frames != exp:
            ck.fail(where=where, problem="frame_indices differ from VERA arithmetic", expected=exp, found=frames,
                    anchor=anchor, length=L)
        if window is not None:
            if not (isinstance(window, list) and len(window) == 2 and float_eq(window[0], s) and float_eq(window[1], e)):
                ck.fail(where=where, problem="window differs from VERA arithmetic", expected=[s, e], found=window)
        if clipped is not None:
            exp_clip = (anchor - VERA_FPS * VERA_T_WINDOW / 2 < 0) or (anchor + VERA_FPS * VERA_T_WINDOW / 2 > L)
            if clipped is not exp_clip:
                ck.fail(where=where, problem="clipped flag differs", expected=exp_clip, found=clipped)

    for inp in run.inputs:
        one("inputs.json:" + str(inp.get("window_uid")), inp.get("video_id"), inp.get("anchor"),
            inp.get("frame_indices"), clipped=inp.get("clipped"))
    for w in man.get("windows") or []:
        # PROTOCOL 3.1: an 'unavailable' record has no eligible window, so there is no arithmetic to check
        if not isinstance(w, dict) or w.get("frame_indices") is None or w.get("status") == "unavailable":
            continue
        one("manifest:" + str(w.get("window_record_id")), w.get("video_id"), w.get("anchor"),
            w.get("frame_indices"), window=w.get("window"), clipped=w.get("clipped"))
    return ck


def _unmasked_reference(run):
    """window_uid -> (reference hashes, problems) from successful unmasked lines (family none, mask [])."""
    refs, conflicts = {}, []
    for rec in run.lines:
        if not is_success(rec) or rec.get("family") != "none" or rec.get("mask") != []:
            continue
        h = rec.get("slot_tensor_sha256")
        if not (isinstance(h, list) and len(h) == 8):
            continue
        uid = rec.get("window_uid")
        if uid in refs:
            if refs[uid][0] != h:
                conflicts.append({"window_uid": uid, "lines": [refs[uid][1], rec["_src"]]})
        else:
            refs[uid] = (h, rec["_src"])
    return refs, conflicts


def check_slot_hashes_present(run):
    ck = Check("slot_hashes_present", "every successful line has 8 slot tensor hashes and slot_identity_ok is true")
    for rec in run.lines:
        if not is_success(rec):
            continue
        ck.tick()
        h = rec.get("slot_tensor_sha256")
        if not (isinstance(h, list) and len(h) == 8 and all(isinstance(x, str) and x for x in h)):
            ck.fail(src=rec["_src"], problem="slot_tensor_sha256 is not 8 hashes",
                    n=len(h) if isinstance(h, list) else None)
        if rec.get("slot_identity_ok") is not True:
            ck.fail(src=rec["_src"], problem="slot_identity_ok is not true", value=rec.get("slot_identity_ok"))
    return ck


def check_slot_identity(run):
    ck = Check("slot_identity", "per input: unmasked slots of every masked line equal the unmasked input's slot hashes "
                                "and every masked slot differs; unmasked lines agree across targets; a replaced slot "
                                "has one hash per (input, family, slot) across all masks (no accumulated replacement)")
    refs, conflicts = _unmasked_reference(run)
    for cf in conflicts:
        ck.fail(problem="unmasked lines of one input carry different slot hashes", **cf)
    replaced = {}  # (window_uid, family, slot) -> {hash: first line}
    for rec in run.lines:
        if not is_success(rec) or rec.get("family") not in ("blur", "grey"):
            continue
        m = canon_mask(rec.get("mask"))
        h = rec.get("slot_tensor_sha256")
        if m is None or not (isinstance(h, list) and len(h) == 8):
            continue
        ck.tick()
        uid = rec.get("window_uid")
        for s in m:
            replaced.setdefault((uid, rec.get("family"), s), {}).setdefault(h[s], rec["_src"])
        if uid not in refs:
            ck.fail(src=rec["_src"], window_uid=uid, problem="no successful unmasked line to compare against")
            continue
        ref = refs[uid][0]
        same_masked = [s for s in m if h[s] == ref[s]]
        diff_unmasked = [s for s in SLOTS if s not in m and h[s] != ref[s]]
        if same_masked or diff_unmasked:
            ck.fail(src=rec["_src"], window_uid=uid, family=rec.get("family"), mask=list(m),
                    masked_slots_unchanged=same_masked, unmasked_slots_changed=diff_unmasked,
                    reference=refs[uid][1])
    # PROTOCOL 4/5.1: each mask is rebuilt from the ORIGINAL frames, so a replaced slot's tensor depends only on
    # (input, family, slot); a second hash means a repeated/accumulated replacement
    for (uid, fam, s), hs in sorted(replaced.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2])):
        if len(hs) > 1:
            ck.fail(window_uid=uid, family=fam, slot=s, problem="a replaced slot has different tensor hashes across "
                                                              "masks", lines=sorted(hs.values())[:6])
    return ck


def check_mask_hash_consistency(run):
    ck = Check("mask_hash_consistency", "the same (input, family, mask) always yields identical slot hashes and "
                                        "sequence_key across targets and repeats; distinct pixel inputs never share a "
                                        "sequence_key")
    groups = {}
    for rec in run.lines:
        if not is_success(rec):
            continue
        m = canon_mask(rec.get("mask"))
        h = rec.get("slot_tensor_sha256")
        if m is None or not isinstance(h, list):
            continue
        k = (rec.get("window_uid"), rec.get("family"), m)
        groups.setdefault(k, []).append(rec)
    seq_owner = {}
    for k, recs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2])):
        ck.tick()
        hs = set(json.dumps(r.get("slot_tensor_sha256")) for r in recs)
        if len(hs) > 1:
            ck.fail(window_uid=k[0], family=k[1], mask=list(k[2]), problem="different slot hashes for one input",
                    lines=[r["_src"] for r in recs][:6])
        sk = set(r.get("sequence_key") for r in recs)
        if len(sk) > 1:
            ck.fail(window_uid=k[0], family=k[1], mask=list(k[2]), problem="different sequence_keys for one input",
                    sequence_keys=sorted(str(x) for x in sk))
        for s in sk:
            if s is None:
                continue
            if s in seq_owner and seq_owner[s] != k:
                o = seq_owner[s]
                ck.fail(problem="one sequence_key used for two different pixel inputs", sequence_key=s,
                        first=[o[0], o[1], list(o[2])], second=[k[0], k[1], list(k[2])])
            seq_owner.setdefault(s, k)
    return ck


def planned_tuples(c):
    """PROTOCOL 6 schedule for one input, as a set of (target, family, mask) plus undetermined parts."""
    P = {(SCORE, "none", ()), (ORIG, "none", ()), (NOSOUND, "none", ())}
    undetermined = []
    if c["L0"] is None:
        return P, undetermined
    for i in SLOTS:
        P.add((SCORE, "blur", (i,)))
        P.add((SCORE, "grey", (i,)))
    for pr in PAIRS:
        P.add((SCORE, "blur", pr))
        P.add((SCORE, "grey", pr))
        P.add((ORIG, "grey", pr))
    pref = c["prefixes"]
    if pref["fixed"]:
        for k in (1, 3, 4):
            m = tuple(pref["fixed"][str(k)])
            P.add((SCORE, "grey", m))
            P.add((ORIG, "grey", m))
    else:
        undetermined.append("fixed_prefixes")
    ad = c["adaptive"]
    for st in ad.get("states", []):
        if st["step"] >= 3:
            for cand in st["candidates"]:
                P.add((SCORE, "blur", tuple(cand["mask"])))
    if ad.get("status") == "incomplete":
        if ad.get("failed_step", 0) >= 3:
            for m in ad.get("missing", []):
                if len(m) >= 3:
                    P.add((SCORE, "blur", tuple(m)))
        undetermined.append("adaptive_after_step_%s" % ad.get("failed_step"))
    if pref["adaptive"]:
        for k in (3, 4):
            m = tuple(pref["adaptive"][str(k)])
            P.add((SCORE, "grey", m))
            P.add((ORIG, "grey", m))
    return P, undetermined


def check_planned_requests(run, calcs):
    ck = Check("planned_requests", "PROTOCOL 6 schedule: every planned (target, family, mask) has at least one logged "
                                   "line; no unplanned requests; only Phase 0 when L0 is missing")
    logged = {}
    for rec in run.lines:
        k = line_key(rec)
        if k is not None:
            logged.setdefault(k[0], set()).add((k[1], k[2], k[3]))
    skipped_status = []
    n_undetermined = 0
    for c in calcs:
        uid = c["window_uid"]
        if c["status"] != "ok":
            skipped_status.append([uid, c["status"]])
            continue
        ck.tick()
        plan, undetermined = planned_tuples(c)
        have = logged.get(uid, set())
        missing = sorted(plan - have, key=lambda t: (t[0], t[1], len(t[2]), t[2]))
        for t in missing[:3]:
            ck.fail(window_uid=uid, problem="planned request never logged", target=t[0], family=t[1], mask=list(t[2]),
                    n_missing_for_input=len(missing))
        if undetermined:
            n_undetermined += 1
            continue
        extra = sorted(have - plan, key=lambda t: (t[0], t[1], len(t[2]), t[2]))
        for t in extra[:3]:
            ck.fail(window_uid=uid, problem="unplanned request logged" + (" (L0 missing: Phase 0 only)"
                                                                          if c["L0"] is None else ""),
                    target=t[0], family=t[1], mask=list(t[2]), n_extra_for_input=len(extra))
    ck.notes["inputs_not_ok_not_checked"] = skipped_status[:MAX_EXAMPLES]
    ck.notes["n_inputs_not_ok"] = len(skipped_status)
    ck.notes["inputs_with_undetermined_schedule"] = n_undetermined
    return ck


def check_cross_family_isolation(calcs, auth, inputs_by_uid):
    ck = Check("cross_family_isolation", "replacing every grey record (scores and labels) leaves every blur ranking, "
                                         "native blur ranking, adaptive trajectory, prefix and T_w unchanged")
    perturbed = {}
    for k, rec in auth.items():
        if k[2] == "grey":
            r = dict(rec)
            for f in ("S_hi", "S"):
                if _num(r.get(f)) is not None:
                    r[f] = -3.0 * r[f] + 11.25
            if _is_int(r.get("label")) and r.get("label") in (0, 1):
                r["label"] = 1 - r["label"]
            perturbed[k] = r
        else:
            perturbed[k] = rec
    pstore = Store(perturbed)
    changed_grey = 0
    for c in calcs:
        if not c["label_available"]:
            continue
        ck.tick()
        c2 = compute_input(inputs_by_uid[c["window_uid"]], pstore)
        for field in ("blur", "blur_native"):
            if json.dumps(_jsonable(c["rankings"][field]), sort_keys=True) != \
                    json.dumps(_jsonable(c2["rankings"][field]), sort_keys=True):
                ck.fail(window_uid=c["window_uid"], problem="grey records changed the %s ranking" % field)
        for field in ("adaptive", "prefixes", "T_w"):
            if json.dumps(_jsonable(c[field]), sort_keys=True) != json.dumps(_jsonable(c2[field]), sort_keys=True):
                ck.fail(window_uid=c["window_uid"], problem="grey records changed %s" % field)
        if json.dumps(_jsonable(c["rankings"]["grey"]), sort_keys=True) != \
                json.dumps(_jsonable(c2["rankings"]["grey"]), sort_keys=True):
            changed_grey += 1
    ck.notes["inputs_whose_grey_ranking_changed_under_perturbation"] = changed_grey
    return ck


def adaptive_structure_problems(ad, fixed_order=None):
    """Structural invariants of one adaptive record, using its own numbers."""
    probs = []
    if not isinstance(ad, dict):
        return ["adaptive record absent"]
    if ad.get("status") == "incomplete":
        return probs
    states = ad.get("states")
    order = ad.get("order")
    if not isinstance(states, list) or len(states) != MAX_BUDGET:
        return ["expected %d states" % MAX_BUDGET]
    so_far = []
    prev_sel_score = None
    for idx, st in enumerate(states):
        if not isinstance(st, dict):
            probs.append("state %d is not an object" % (idx + 1))
            return probs
        if st.get("step") != idx + 1:
            probs.append("state %d has step %r" % (idx + 1, st.get("step")))
        if st.get("masked") != sorted(so_far):
            probs.append("state %d masked %r != previous selections %r" % (idx + 1, st.get("masked"), sorted(so_far)))
        cands = st.get("candidates") or []
        slots = sorted(cd.get("slot") for cd in cands if isinstance(cd, dict))
        remaining = [s for s in SLOTS if s not in so_far]
        if slots != remaining:
            probs.append("state %d candidates %r != every remaining slot %r" % (idx + 1, slots, remaining))
        for cd in cands:
            if isinstance(cd, dict) and cd.get("mask") != sorted(so_far + [cd.get("slot")]):
                probs.append("state %d candidate %r has mask %r" % (idx + 1, cd.get("slot"), cd.get("mask")))
        sel = st.get("selected")
        if sel in so_far:
            probs.append("state %d selected previously masked slot %r" % (idx + 1, sel))
        effs = [(cd.get("effect"), cd.get("slot")) for cd in cands
                if isinstance(cd, dict) and _is_num(cd.get("effect"))]
        if effs:
            best = max(e for e, _ in effs)
            tied = sorted(s for e, s in effs if e == best)
            if sel != tied[0]:
                probs.append("state %d selected %r but max-effect/lower-slot rule gives %r" % (idx + 1, sel, tied[0]))
            if sorted(st.get("tied_max") or []) != tied[1:]:
                probs.append("state %d tied_max %r != %r" % (idx + 1, st.get("tied_max"), tied[1:]))
        if idx > 0 and prev_sel_score is not None and not float_eq(st.get("state_score"), prev_sel_score):
            probs.append("state %d state_score %r != previous selected candidate score %r"
                         % (idx + 1, st.get("state_score"), prev_sel_score))
        prev_sel_score = next((cd.get("score") for cd in cands if isinstance(cd, dict) and cd.get("slot") == sel),
                              None)
        so_far.append(sel)
    if order != so_far:
        probs.append("order %r != selections %r" % (order, so_far))
    if fixed_order and so_far and so_far[0] != fixed_order[0]:
        probs.append("adaptive step 1 (%r) != fixed pi_1 (%r)" % (so_far[0], fixed_order[0]))
    return probs


def check_adaptive_invariants(calcs, theirs=None):
    ck = Check("adaptive_invariants", "adaptive trajectories (recomputed, and the analyzer's when present): every "
                                      "remaining slot is a candidate, no masked slot is re-selected, max-effect/lower-"
                                      "slot selection, step 1 equals fixed pi_1, prefixes and T_w follow the orders")
    for c in calcs:
        if not c["label_available"]:
            continue
        uid = c["window_uid"]
        fixed_order = c["rankings"]["blur"].get("order")
        for who, ad, rk_blur, pref, tw in (
                [("recomputed", c["adaptive"], c["rankings"]["blur"], c["prefixes"], c["T_w"])] +
                ([("analyzer",) + _their_parts(theirs[uid])] if theirs and uid in theirs else [])):
            ck.tick()
            fo = rk_blur.get("order") if isinstance(rk_blur, dict) else None
            for p in adaptive_structure_problems(ad, fo if who == "analyzer" else fixed_order):
                ck.fail(window_uid=uid, source=who, problem=p)
            if isinstance(ad, dict) and ad.get("status") != "incomplete" and isinstance(ad.get("order"), list):
                if who == "analyzer" and (ad.get("family") != "blur" or ad.get("readout") != "S_hi"):
                    ck.fail(window_uid=uid, source=who, problem="adaptive family/readout is not blur/S_hi",
                            family=ad.get("family"), readout=ad.get("readout"))
                exp = {str(k): sorted(ad["order"][:k]) for k in range(1, MAX_BUDGET + 1)}
                got = (pref or {}).get("adaptive") if isinstance(pref, dict) else None
                if got != exp:
                    ck.fail(window_uid=uid, source=who, problem="adaptive prefixes != sorted(order[:k])",
                            expected=exp, found=got)
            if isinstance(fo, list) and len(fo) == 8:
                exp = {str(k): sorted(fo[:k]) for k in range(1, MAX_BUDGET + 1)}
                got = (pref or {}).get("fixed") if isinstance(pref, dict) else None
                if got != exp:
                    ck.fail(window_uid=uid, source=who, problem="fixed prefixes != sorted(blur order[:k])",
                            expected=exp, found=got)
                if tw != exp["2"]:
                    ck.fail(window_uid=uid, source=who, problem="T_w != prefixes.fixed['2']", expected=exp["2"],
                            found=tw)
    return ck


def _their_parts(t):
    fixed = t.get("fixed") if isinstance(t.get("fixed"), dict) else {}
    return (t.get("adaptive"), fixed.get("blur"), t.get("prefixes"), t.get("T_w"))


# ----------------------------------------------------------------------------------------- comparisons

def _canon_ties(ties):
    if not isinstance(ties, list):
        return ties
    out = []
    for g in ties:
        if isinstance(g, list):
            if len(g) > 1:
                out.append(sorted(g))
        else:
            out.append(g)
    return sorted(out, key=lambda g: (g if isinstance(g, list) else [g]))


def compare_ranking(ck, ctx, field, mine, theirs):
    if not isinstance(theirs, dict):
        ck.tick()
        ck.fail(field=field, problem="ranking absent", **ctx)
        return
    if mine.get("status") == "incomplete":
        cmp_exact(ck, ctx, field + ".status", "incomplete", theirs.get("status"))
        cmp_exact(ck, ctx, field + ".missing", sorted(str(x) for x in mine["missing"]),
                  sorted(str(x) for x in (theirs.get("missing") or [])))
        return
    if theirs.get("status") not in (None, "ok", "complete"):
        ck.tick()
        ck.fail(field=field + ".status", expected="complete", found=theirs.get("status"), **ctx)
        return
    te = theirs.get("effects") if isinstance(theirs.get("effects"), dict) else {}
    for i in SLOTS:
        cmp_float(ck, ctx, "%s.effects.%d" % (field, i), mine["effects"][str(i)], te.get(str(i)))
    cmp_exact(ck, ctx, field + ".order", mine["order"], theirs.get("order"))
    cmp_exact(ck, ctx, field + ".ties", _canon_ties(mine["ties"]), _canon_ties(theirs.get("ties")))
    tsc = theirs.get("tie_spans_cut") if isinstance(theirs.get("tie_spans_cut"), dict) else {}
    for k in ("1", "2", "3", "4"):
        cmp_exact(ck, ctx, "%s.tie_spans_cut.%s" % (field, k), mine["tie_spans_cut"][k], tsc.get(k))
    cmp_float(ck, ctx, field + ".top2_margin", mine["top2_margin"], theirs.get("top2_margin"))
    cmp_exact(ck, ctx, field + ".all_tied", mine["all_tied"], theirs.get("all_tied"))
    cmp_exact(ck, ctx, field + ".all_nonpositive", mine["all_nonpositive"], theirs.get("all_nonpositive"))


def compare_adaptive(ck, ctx, mine, theirs):
    if not isinstance(theirs, dict):
        ck.tick()
        ck.fail(field="adaptive", problem="adaptive record absent", **ctx)
        return
    if mine.get("status") == "incomplete":
        cmp_exact(ck, ctx, "adaptive.status", "incomplete", theirs.get("status"))
        return
    cmp_exact(ck, ctx, "adaptive.family", "blur", theirs.get("family"))
    cmp_exact(ck, ctx, "adaptive.readout", "S_hi", theirs.get("readout"))
    cmp_exact(ck, ctx, "adaptive.max_budget", MAX_BUDGET, theirs.get("max_budget"))
    cmp_exact(ck, ctx, "adaptive.order", mine["order"], theirs.get("order"))
    ts = theirs.get("states") if isinstance(theirs.get("states"), list) else []
    if not cmp_exact(ck, ctx, "adaptive.n_states", len(mine["states"]), len(ts)):
        return
    for ms, tst in zip(mine["states"], ts):
        if not isinstance(tst, dict):
            ck.tick()
            ck.fail(field="adaptive.states", problem="state is not an object", **ctx)
            continue
        f = "adaptive.state%d" % ms["step"]
        cmp_exact(ck, ctx, f + ".step", ms["step"], tst.get("step"))
        cmp_exact(ck, ctx, f + ".masked", ms["masked"], tst.get("masked"))
        cmp_float(ck, ctx, f + ".state_score", ms["state_score"], tst.get("state_score"))
        tc = {cd.get("slot"): cd for cd in (tst.get("candidates") or []) if isinstance(cd, dict)}
        cmp_exact(ck, ctx, f + ".candidate_slots", sorted(cd["slot"] for cd in ms["candidates"]),
                  sorted(s for s in tc if _is_int(s)))
        for cd in ms["candidates"]:
            t = tc.get(cd["slot"])
            if t is None:
                continue
            cf = "%s.cand%d" % (f, cd["slot"])
            cmp_exact(ck, ctx, cf + ".mask", cd["mask"], t.get("mask"))
            cmp_float(ck, ctx, cf + ".score", cd["score"], t.get("score"))
            cmp_float(ck, ctx, cf + ".effect", cd["effect"], t.get("effect"))
        cmp_exact(ck, ctx, f + ".selected", ms["selected"], tst.get("selected"))
        cmp_exact(ck, ctx, f + ".tied_max", ms["tied_max"], sorted(tst.get("tied_max") or []))


def load_their_rankings(path):
    out, problems = {}, []
    for lineno, obj, err in read_jsonl(path):
        if err:
            problems.append({"line": lineno, "problem": err})
            continue
        uid = obj.get("window_uid")
        if uid in out:
            problems.append({"line": lineno, "problem": "duplicate window_uid", "window_uid": uid})
        out[uid] = obj
    return out, problems


def check_rankings_compare(run, calcs):
    ck = Check("rankings_compare", "derived/rankings.jsonl equals the recomputation: one record per label-available "
                                   "input; L0, q, baseline, three fixed RANKINGs, adaptive states, prefixes, T_w")
    path = run.path("derived", "rankings.jsonl")
    if not os.path.isfile(path):
        ck.skip("derived/rankings.jsonl absent")
        return ck, None
    run.note_file("derived/rankings.jsonl")
    theirs, problems = load_their_rankings(path)
    for p in problems:
        ck.fail(**p)
    mine = {c["window_uid"]: c for c in calcs if c["label_available"]}
    for uid in sorted(set(mine) - set(theirs)):
        ck.tick()
        ck.fail(window_uid=uid, problem="label-available input has no rankings record")
    for uid in sorted(set(theirs) - set(mine), key=str):
        ck.tick()
        ck.fail(window_uid=uid, problem="rankings record for an input without a parsed L0 (or unknown input)")
    for uid in sorted(set(mine) & set(theirs)):
        c, t = mine[uid], theirs[uid]
        ctx = {"window_uid": uid}
        cmp_exact(ck, ctx, "L0", c["L0"], t.get("L0"))
        cmp_exact(ck, ctx, "q", c["q"], t.get("q"))
        tb = t.get("baseline") if isinstance(t.get("baseline"), dict) else {}
        cmp_float(ck, ctx, "baseline.S_hi", c["baseline_S_hi"], tb.get("S_hi"))
        cmp_float(ck, ctx, "baseline.S", c["baseline_S"], tb.get("S"))
        tf = t.get("fixed") if isinstance(t.get("fixed"), dict) else {}
        for name in ("blur", "grey", "blur_native"):
            compare_ranking(ck, ctx, "fixed." + name, c["rankings"][name], tf.get(name))
        compare_adaptive(ck, ctx, c["adaptive"], t.get("adaptive"))
        tp = t.get("prefixes") if isinstance(t.get("prefixes"), dict) else {}
        for mode in ("fixed", "adaptive"):
            if c["prefixes"][mode] is None:
                continue
            tm = tp.get(mode) if isinstance(tp.get(mode), dict) else {}
            for k in ("1", "2", "3", "4"):
                cmp_exact(ck, ctx, "prefixes.%s.%s" % (mode, k), c["prefixes"][mode][k], tm.get(k))
        cmp_exact(ck, ctx, "T_w", c["T_w"], t.get("T_w"))
    return ck, theirs


def _cmp_bootstrap_ci(ck, ctx, field, mine, theirs):
    """Bootstrap intervals are compared endpoint by endpoint within FLOAT_TOL. PROTOCOL 7.1 fixes the draws
    (random.Random(seed).randrange(V)) and the percentile rule, not the floating-point summation order or
    where the pp scaling is applied, so last-bit differences are not discrepancies."""
    if mine is None or not (isinstance(theirs, list) and len(theirs) == 2):
        return cmp_exact(ck, ctx, field, mine, theirs)
    ok_lo = cmp_float(ck, ctx, field + "[0]", mine[0], theirs[0])
    ok_hi = cmp_float(ck, ctx, field + "[1]", mine[1], theirs[1])
    return ok_lo and ok_hi


def _compare_estimate(ck, ctx, mine, theirs, full):
    cmp_float(ck, ctx, "estimate_pp", mine["estimate_pp"], theirs.get("estimate_pp"))
    if full or "ci95_pp" in theirs:
        _cmp_bootstrap_ci(ck, ctx, "ci95_pp", mine["ci95_pp"], theirs.get("ci95_pp"))
    for f in ("n_inputs", "n_inputs_missing_dw"):
        if full or f in theirs:
            cmp_exact(ck, ctx, f, mine[f], theirs.get(f))
    if full or "n_videos" in theirs:
        if not exact_eq(mine["n_videos"], theirs.get("n_videos")):
            ck.tick()
            ck.fail(field="n_videos", expected=mine["n_videos"], found=theirs.get("n_videos"),
                    note="expected = videos with >= 1 non-missing d_w; videos in the population = %d"
                         % mine["n_videos_population"], **ctx)
        else:
            ck.tick()
    if full or "bounds_pp" in theirs:
        tb = theirs.get("bounds_pp")
        if not (isinstance(tb, list) and len(tb) == 2):
            ck.tick()
            ck.fail(field="bounds_pp", expected=mine["bounds_pp"], found=tb, **ctx)
        else:
            cmp_float(ck, ctx, "bounds_pp[0]", mine["bounds_pp"][0], tb[0])
            cmp_float(ck, ctx, "bounds_pp[1]", mine["bounds_pp"][1], tb[1])
    for f in ("video_means_pp", "d_w"):
        if not (full or f in theirs):
            continue
        tv = theirs.get(f)
        if not isinstance(tv, dict):
            ck.tick()
            ck.fail(field=f, problem="absent or not an object", **ctx)
            continue
        cmp_exact(ck, ctx, f + ".keys", sorted(mine[f]), sorted(tv))
        for k in sorted(set(mine[f]) & set(tv)):
            cmp_float(ck, ctx, "%s[%s]" % (f, k), mine[f][k], tv[k])


def check_primary_compare(run, primary):
    ck = Check("primary_compare", "results/primary.json equals the recomputed primary estimate, bootstrap interval, "
                                  "stability seeds, n's, video means, d_w, worst-case bounds and sensitivities")
    path = run.path("results", "primary.json")
    if not os.path.isfile(path):
        ck.skip("results/primary.json absent")
        return ck
    run.note_file("results/primary.json")
    try:
        t = read_json(path)
    except ValueError as exc:
        ck.tick()
        ck.fail(problem="primary.json does not parse: %s" % exc)
        return ck
    if not isinstance(t, dict):
        ck.tick()
        ck.fail(problem="primary.json is not an object")
        return ck
    ctx = {"block": "primary"}
    tb = t.get("bootstrap") if isinstance(t.get("bootstrap"), dict) else {}
    cmp_exact(ck, ctx, "bootstrap.seed", primary["bootstrap"]["seed"], tb.get("seed"))
    cmp_exact(ck, ctx, "bootstrap.B", primary["bootstrap"]["B"], tb.get("B"))
    _compare_estimate(ck, ctx, primary, t, full=True)
    tstab = {s.get("seed"): s for s in (tb.get("stability") or []) if isinstance(s, dict)}
    for s in primary["bootstrap"]["stability"]:
        ts = tstab.get(s["seed"])
        if ts is None:
            ck.tick()
            ck.fail(field="bootstrap.stability", problem="seed %d absent" % s["seed"], **ctx)
            continue
        _cmp_bootstrap_ci(ck, ctx, "bootstrap.stability[seed=%d].ci95_pp" % s["seed"], s["ci95_pp"],
                          ts.get("ci95_pp"))
    ts_all = t.get("sensitivity") if isinstance(t.get("sensitivity"), dict) else {}
    for name in SENSITIVITIES:
        ts = ts_all.get(name)
        if not isinstance(ts, dict):
            ck.tick()
            ck.fail(field="sensitivity." + name, problem="absent", **ctx)
            continue
        _compare_estimate(ck, {"block": "sensitivity." + name}, primary["sensitivity"][name], ts, full=False)
    return ck


def _row_map(rows, keyfun, ck, what):
    out = {}
    for i, r in enumerate(rows):
        k = keyfun(r)
        if k in out:
            ck.tick()
            ck.fail(problem="duplicate %s row" % what, key=list(k) if isinstance(k, tuple) else k, row=i + 2)
        out[k] = r
    return out


def check_inputs_metrics_compare(run, calcs):
    ck = Check("inputs_metrics_compare", "results/inputs_metrics.csv equals the recomputation (d_w, F_top2, "
                                         "F_random_mean, n_pair_labels, e_w, L0, q, T_w, tau_b and overlaps when present)")
    path = run.path("results", "inputs_metrics.csv")
    if not os.path.isfile(path):
        ck.skip("results/inputs_metrics.csv absent")
        return ck
    run.note_file("results/inputs_metrics.csv")
    headers, rows = read_csv(path)
    required = {"window_uid": find_col(headers, ("window_uid",)), "d_w": find_col(headers, ("d_w",)),
                "F_top2": find_col(headers, ("F_top2",)), "F_random_mean": find_col(headers, ("F_random_mean",)),
                "n_pair_labels": find_col(headers, ("n_pair_labels",)), "e_w": find_col(headers, ("e_w",))}
    absent = [k for k, v in required.items() if v is None]
    if absent:
        ck.tick()
        ck.fail(problem="required columns absent", columns=absent, headers=headers)
        return ck
    opt = {
        "L0": find_col(headers, ("L0",)),
        "q": find_col(headers, ("q",)),
        "T_w": find_col(headers, ("T_w", "T_w_slots", "top2_slots")),
        "video_id": find_col(headers, ("video_id",)),
        "nosound_label": find_col(headers, ("nosound_label", "L_nosound", "nosound_L", "L0_nosound", "nosound")),
        "tau_blur_grey": find_col(headers, (), lambda h: "tau" in h and ("grey" in h or "gray" in h)),
        "tau_native_hi": find_col(headers, (), lambda h: "tau" in h and "native" in h),
        "overlap_blur_grey": find_col(headers, (), lambda h: "overlap" in h and ("grey" in h or "gray" in h)),
        "overlap_native_hi": find_col(headers, (), lambda h: "overlap" in h and "native" in h),
    }
    ck.notes["optional_columns_matched"] = {k: v for k, v in opt.items() if v}
    ck.notes["optional_columns_unmatched"] = sorted(k for k, v in opt.items() if not v)
    theirs = _row_map(rows, lambda r: r.get(required["window_uid"]), ck, "inputs_metrics")
    mine = {c["window_uid"]: c for c in calcs}
    for uid in sorted(set(mine) - set(theirs)):
        ck.tick()
        ck.fail(window_uid=uid, problem="input has no inputs_metrics row")
    for uid in sorted(set(theirs) - set(mine), key=str):
        ck.tick()
        ck.fail(window_uid=uid, problem="inputs_metrics row for an unknown input")
    n_not_compared = 0
    for uid in sorted(set(mine) & set(theirs)):
        c, r = mine[uid], theirs[uid]
        ctx = {"window_uid": uid}
        if opt["L0"]:
            cmp_exact(ck, ctx, "L0", c["L0"], csv_int(r[opt["L0"]]))
        if opt["q"]:
            cmp_exact(ck, ctx, "q", c["q"], csv_int(r[opt["q"]]))
        if opt["video_id"]:
            cmp_exact(ck, ctx, "video_id", c["video_id"], r[opt["video_id"]])
        if opt["nosound_label"]:
            cmp_exact(ck, ctx, "nosound_label", c["nosound_label"], csv_int(r[opt["nosound_label"]]))
        if c["L0"] == 1:
            cmp_float(ck, ctx, "d_w", c["d_w"], csv_num(r[required["d_w"]]))
            cmp_float(ck, ctx, "F_top2", c["F_top2"], csv_num(r[required["F_top2"]]))
            got = csv_num(r[required["F_random_mean"]])
            if c["n_pair_labels"] == N_PAIRS:
                cmp_float(ck, ctx, "F_random_mean", c["F_random_mean"], got)
            else:
                # undefined by the protocol when labels are missing: accept blank or the available-label mean
                ck.tick()
                if not (got is None or float_eq(got, c["F_random_mean"])):
                    ck.fail(field="F_random_mean", expected=[None, c["F_random_mean"]], found=got, **ctx)
        elif c["L0"] is None:
            for f in ("d_w", "F_top2"):
                cmp_float(ck, ctx, f, None, csv_num(r[required[f]]))
        else:
            n_not_compared += 1
        if c["label_available"]:
            cmp_exact(ck, ctx, "n_pair_labels", c["n_pair_labels"], csv_int(r[required["n_pair_labels"]]))
        cmp_float(ck, ctx, "e_w", c["e_w"], csv_num(r[required["e_w"]]))
        if opt["T_w"] and c["label_available"]:
            cmp_exact(ck, ctx, "T_w", c["T_w"], csv_ints(r[opt["T_w"]]) or None)
        for f in ("tau_blur_grey", "tau_native_hi"):
            if opt[f] and c["label_available"]:
                cmp_float(ck, ctx, f, c[f], csv_num(r[opt[f]]))
        for f in ("overlap_blur_grey", "overlap_native_hi"):
            if opt[f] and c["label_available"]:
                cmp_exact(ck, ctx, f, c[f], csv_int(r[opt[f]]))
    ck.notes["L0_0_rows_d_w_F_not_compared"] = n_not_compared
    return ck


def check_pairs_compare(run, calcs):
    ck = Check("pairs_compare", "results/pairs.csv equals the recomputation: one row per (label-available input, "
                                "family, pair) with S_hi, S, D, J and, on grey, label and flip")
    path = run.path("results", "pairs.csv")
    if not os.path.isfile(path):
        ck.skip("results/pairs.csv absent")
        return ck
    run.note_file("results/pairs.csv")
    headers, rows = read_csv(path)
    cols = {"window_uid": find_col(headers, ("window_uid",)), "family": find_col(headers, ("family",)),
            "pair": find_col(headers, ("pair", "mask", "slots", "pair_slots", "pair_mask")),
            "S_hi": find_col(headers, ("S_hi",)), "S": find_col(headers, ("S", "S_native")),
            "D": find_col(headers, ("D",)), "J": find_col(headers, ("J",))}
    absent = [k for k, v in cols.items() if v is None]
    if absent:
        ck.tick()
        ck.fail(problem="required columns absent", columns=absent, headers=headers)
        return ck
    lab_col = find_col(headers, ("label", "original_label", "L"))
    flip_col = find_col(headers, ("flip", "flipped"))
    ck.notes["label_column"] = lab_col
    ck.notes["flip_column"] = flip_col

    def key(r):
        m = csv_ints(r.get(cols["pair"])) or []
        return (r.get(cols["window_uid"]), r.get(cols["family"]), tuple(sorted(m)))
    theirs = _row_map(rows, key, ck, "pairs")
    mine = {}
    for c in calcs:
        if c["label_available"]:
            for row in c["pairs"]:
                mine[(c["window_uid"], row["family"], row["pair"])] = row
    for k in sorted(set(mine) - set(theirs)):
        ck.tick()
        ck.fail(problem="pair row absent", key=[k[0], k[1], list(k[2])])
    for k in sorted(set(theirs) - set(mine), key=str):
        ck.tick()
        ck.fail(problem="unexpected pair row", key=[k[0], k[1], list(k[2])])
    for k in sorted(set(mine) & set(theirs)):
        m, r = mine[k], theirs[k]
        ctx = {"window_uid": k[0], "family": k[1], "pair": list(k[2])}
        for f in ("S_hi", "S", "D", "J"):
            cmp_float(ck, ctx, f, m[f], csv_num(r[cols[f]]))
        if k[1] == "grey":
            if lab_col:
                cmp_exact(ck, ctx, "label", m["label"], csv_int(r[lab_col]))
            if flip_col:
                cmp_exact(ck, ctx, "flip", m["flip"], csv_int(r[flip_col]))
    return ck


def check_trajectories_compare(run, calcs):
    ck = Check("trajectories_compare", "results/trajectories.csv equals the recomputation: per (input, mode, k) the "
                                       "grey-evaluated prefix mask, label, flip, q*dS_hi and q*dS")
    path = run.path("results", "trajectories.csv")
    if not os.path.isfile(path):
        ck.skip("results/trajectories.csv absent")
        return ck
    run.note_file("results/trajectories.csv")
    headers, rows = read_csv(path)
    cols = {"window_uid": find_col(headers, ("window_uid",)), "mode": find_col(headers, ("mode",)),
            "k": find_col(headers, ("k", "budget")), "mask": find_col(headers, ("mask", "prefix", "prefix_mask"))}
    absent = [k for k, v in cols.items() if v is None]
    if absent:
        ck.tick()
        ck.fail(problem="required columns absent", columns=absent, headers=headers)
        return ck
    lab_col = find_col(headers, ("label", "original_label", "L"))
    flip_col = find_col(headers, ("flip", "flipped"))
    dshi_col = find_col(headers, ("q_dS_hi", "q_delta_S_hi", "qΔS_hi", "q·ΔS_hi", "dS_hi"),
                        lambda h: re.fullmatch(r"q?(d|delta)?shi", h) is not None and h != "shi")
    ds_col = find_col(headers, ("q_dS", "q_delta_S", "qΔS", "q·ΔS", "q_dS_native", "dS"),
                      lambda h: re.fullmatch(r"(q(d|delta)?|(d|delta))s(native|bf16)?", h) is not None)
    ck.notes["columns_matched"] = {"label": lab_col, "flip": flip_col, "q_dS_hi": dshi_col, "q_dS": ds_col}

    def key(r):
        return (r.get(cols["window_uid"]), r.get(cols["mode"]), csv_int(r.get(cols["k"])))
    theirs = _row_map(rows, key, ck, "trajectories")
    mine = {}
    for c in calcs:
        for row in c["trajectories"]:
            mine[(c["window_uid"], row["mode"], row["k"])] = row
    for k in sorted(set(mine) - set(theirs)):
        ck.tick()
        ck.fail(problem="trajectory row absent", key=list(k))
    for k in sorted(set(theirs) - set(mine), key=str):
        ck.tick()
        ck.fail(problem="unexpected trajectory row (no defined prefix for it)", key=list(k))
    for k in sorted(set(mine) & set(theirs)):
        m, r = mine[k], theirs[k]
        ctx = {"window_uid": k[0], "mode": k[1], "k": k[2]}
        cmp_exact(ck, ctx, "mask", m["mask"], csv_ints(r[cols["mask"]]))
        if lab_col:
            cmp_exact(ck, ctx, "label", m["label"], csv_int(r[lab_col]))
        if flip_col:
            cmp_exact(ck, ctx, "flip", m["flip"], csv_int(r[flip_col]))
        if dshi_col:
            cmp_float(ck, ctx, "q_dS_hi", m["q_dS_hi"], csv_num(r[dshi_col]))
        if ds_col:
            cmp_float(ck, ctx, "q_dS", m["q_dS"], csv_num(r[ds_col]))
    return ck


# ----------------------------------------------------------------------------------------- erratum E1

def _bf16_margin(rec):
    """Native BF16 margin on its own grid: logit_1 - logit_0 (both exact BF16 values)."""
    a, b = _num(rec.get("logit_1")), _num(rec.get("logit_0"))
    return None if a is None or b is None else a - b


def e1_reproduce(exp02_dir, ck, expected=None):
    """Recompute Delta and Delta_bf16 for experiment02's effect rows from raw score records."""
    expected = E1_EXPECTED if expected is None else expected
    res = os.path.join(exp02_dir, "results")
    bpath = os.path.join(res, "baseline_responses.jsonl")
    ppath = os.path.join(res, "perturbation_responses.jsonl")
    cpath = os.path.join(res, "temporal_effects.csv")
    case_path = os.path.join(exp02_dir, "cases.json")
    for p in (bpath, ppath):
        if not os.path.isfile(p):
            ck.tick()
            ck.fail(problem="experiment02 raw record file absent", path=p)
            return None
    case_frames = {}
    if os.path.isfile(case_path):
        try:
            cobj = read_json(case_path)
            for cs in cobj.get("cases", []):
                case_frames[cs.get("case_id")] = cs.get("frame_indices")
        except (ValueError, AttributeError):
            ck.fail(problem="cases.json does not parse", path=case_path)
    base, base_repeats = {}, {}
    for lineno, rec, err in read_jsonl(bpath):
        if err or rec.get("target") != SCORE or not is_success(rec) or rec.get("mask") != []:
            continue
        cid = rec.get("case_id")
        base_repeats.setdefault(cid, set()).add((rec.get("S_hi"), _bf16_margin(rec)))
        if rec.get("repeat") == 0:
            base[cid] = rec
    rows = {}
    for lineno, rec, err in read_jsonl(ppath):
        if err or rec.get("stage") != "B" or rec.get("target") != SCORE or not is_success(rec):
            continue
        m = canon_mask(rec.get("mask"))
        if m is None:
            ck.fail(problem="invalid mask in experiment02 record", line=lineno)
            continue
        rows[(rec.get("case_id"), rec.get("family"), m)] = rec
    for cid, vals in base_repeats.items():
        ck.tick()
        if len(vals) > 1:
            ck.fail(problem="experiment02 baseline score repeats disagree", case_id=cid, values=sorted(vals, key=str))
    computed = {}
    n_sub = n_sub_changed = n_sub_raw_nonzero = 0
    by_family = {}

    def on_grid(r):
        """The stored native S must be the BF16 logit margin up to float noise."""
        mg, s = _bf16_margin(r), _num(r.get("S"))
        ok = not (mg is None or s is None or abs(s - mg) > E1_S_NOISE_TOL) and _num(r.get("S_hi")) is not None \
            and _num(r.get("S_resolution")) is not None
        if not ok:
            ck.fail(problem="native S is not the BF16 logit margin (logit_1 - logit_0)",
                    request_id=r.get("request_id"), S=r.get("S"), margin=mg, S_hi=r.get("S_hi"),
                    S_resolution=r.get("S_resolution"))
        return ok

    base_ok = {}
    for cid, b in sorted(base.items(), key=lambda kv: str(kv[0])):
        ck.tick()
        base_ok[cid] = on_grid(b)
    for k in sorted(rows, key=lambda x: (str(x[0]), str(x[1]), len(x[2]), x[2])):
        rec = rows[k]
        ck.tick()
        b = base.get(k[0])
        if b is None:
            ck.fail(problem="no repeat-0 baseline score for case", case_id=k[0])
            continue
        if case_frames.get(k[0]) is not None and rec.get("frame_indices") != case_frames[k[0]]:
            ck.fail(problem="record frame_indices differ from cases.json", case_id=k[0], mask=list(k[2]))
        if not (on_grid(rec) and base_ok.get(k[0])):
            continue
        d = b["S_hi"] - rec["S_hi"]
        d_bf16 = _bf16_margin(b) - _bf16_margin(rec)
        d_raw = b["S"] - rec["S"]
        s_res = rec["S_resolution"]
        sub = abs(d) < s_res
        computed[k] = {"Delta": d, "Delta_bf16": d_bf16, "S_resolution": s_res, "sub": sub,
                       "S_baseline": b["S_hi"], "S_perturbed": rec["S_hi"], "Delta_raw_S": d_raw}
        fam = by_family.setdefault(k[1], {"n_rows": 0, "n_substep": 0, "n_substep_bf16_changed": 0})
        fam["n_rows"] += 1
        if sub:
            n_sub += 1
            fam["n_substep"] += 1
            if d_bf16 != 0:
                n_sub_changed += 1
                fam["n_substep_bf16_changed"] += 1
            if d_raw != 0:
                n_sub_raw_nonzero += 1
    summary = {"n_rows": len(computed), "n_substep": n_sub, "n_substep_bf16_changed": n_sub_changed,
               "by_family": by_family,
               "n_substep_raw_S_field_nonzero": n_sub_raw_nonzero,
               "native_difference_definition": "Delta_bf16 = (logit_1 - logit_0)(base) - (logit_1 - logit_0)(mask); "
                                               "the stored S field carries <= 1.2e-7 float noise from log-prob "
                                               "subtraction, so its literal difference is reported separately "
                                               "(n_substep_raw_S_field_nonzero); temporal_effects.csv prints "
                                               "Delta_bf16 to 6 decimals (agrees with the margin difference) but "
                                               "computed its Delta_bf16_zero flag from the raw S fields",
               "expected": expected}
    for f, v in expected.items():
        ck.tick()
        if summary[f] != v:
            ck.fail(problem="E1 count differs from the documented value", field=f, expected=v, found=summary[f])
    # cross-check against temporal_effects.csv (Delta, Delta_bf16, S_resolution per row)
    if os.path.isfile(cpath):
        _, crow = read_csv(cpath)
        seen = set()
        for r in crow:
            try:
                m = tuple(int(x) for x in r["mask_slots"].split("-"))
            except (KeyError, ValueError):
                ck.fail(problem="unparseable mask_slots in temporal_effects.csv", row=r.get("mask_slots"))
                continue
            k = (r.get("case_id"), r.get("family"), m)
            seen.add(k)
            ck.tick()
            mine = computed.get(k)
            if mine is None:
                ck.fail(problem="temporal_effects.csv row has no raw record", key=[k[0], k[1], list(m)])
                continue
            for f in ("Delta", "Delta_bf16", "S_resolution", "S_baseline", "S_perturbed"):
                got = csv_num(r.get(f))
                if not float_eq(mine[f], got, E2_RESULT_TOL):
                    ck.fail(problem="temporal_effects.csv value differs", key=[k[0], k[1], list(m)], field=f,
                            expected=mine[f], found=got)
            # experiment02 computed its Delta_bf16_zero flag from the stored S fields literally
            flags = {"Delta_below_resolution": int(mine["sub"]), "Delta_bf16_zero": int(mine["Delta_raw_S"] == 0)}
            for f, v in flags.items():
                if csv_int(r.get(f)) != v:
                    ck.fail(problem="temporal_effects.csv flag differs", key=[k[0], k[1], list(m)], field=f,
                            expected=v, found=r.get(f))
        for k in sorted(set(computed) - seen, key=str):
            ck.fail(problem="raw effect row absent from temporal_effects.csv", key=[k[0], k[1], list(k[2])])
        summary["temporal_effects_csv_rows"] = len(crow)
    else:
        ck.notes["temporal_effects_csv"] = "absent; per-row cross-check not run"
    return summary


def check_e1(exp02_dir, expected=None):
    ck = Check("erratum_e1", "experiment02 erratum E1 from raw score records: |Delta S_hi| < S_resolution rows whose "
                             "native BF16 margin still changes (expected 36 of 109 of 168), cross-checked against "
                             "temporal_effects.csv")
    if not exp02_dir:
        ck.skip("--exp02-run-dir not given")
        return ck, None
    if not os.path.isdir(exp02_dir):
        ck.tick()
        ck.fail(problem="experiment02 run dir absent", path=exp02_dir)
        return ck, None
    summary = e1_reproduce(os.path.abspath(exp02_dir), ck, expected)
    return ck, summary


# ----------------------------------------------------------------------------------------- orchestration

AMBIGUITIES = [
    "latest line: request lines are ordered by request file name (sorted) and then line number",
    "parsed label = parse_status 'ok' with an integer label 0/1; repeats 1/2 are expected only after an unparsed "
    "repeat r-1 of the same generation; score requests have repeat 0 only",
    "native readout = logit_1 - logit_0 (protocol/RECORDS v1.1; the stored S field carries ~1e-7 log-softmax noise)",
    "development = video_id in the protocol's two development videos (or tagged so in inputs.json)",
    "primary population ignores inputs.json status (any input with parsed L0 = 1); planned-request checks cover "
    "status 'ok' inputs only",
    "a successful line with a non-finite readout is treated as a missing value",
    "a RANKING whose baseline score is missing is incomplete with 'baseline' in its missing list",
    "ties lists only groups of size >= 2 and is compared as an unordered set of sorted groups",
    "d_w, F_top2 and F_random_mean are defined for L0 = 1 inputs only (F = 1 iff label 0); L0 = 0 rows are not "
    "compared for them; F_random_mean with missing labels is accepted blank or as the mean of available labels",
    "if T_w is undefined (incomplete blur ranking) d_w is missing and the bound uses selected = 0 / 1",
    "e_w is missing unless the baseline score, T_w and all 28 grey-pair S_hi values are present",
    "n_videos = videos with at least one non-missing d_w (the bootstrap V)",
    "bootstrap results (ci95_pp and stability ci95_pp) are compared endpoint by endpoint within 1e-9 pp (the "
    "protocol fixes the draws and the percentile rule, not float summation order); the validator sums with "
    "Python's built-in sum in input order within a video and video_id order across videos; percentile = "
    "s[lo] + (s[hi]-s[lo])*frac; pp = 100 * value at the end",
    "sensitivity estimates use the primary seed and B; their CIs are compared only when the analyzer reports them",
    "E1 native difference uses the BF16 logit margin (logit_1 - logit_0); the stored S field differs from it by "
    "<= 1.2e-7 and its literal difference is reported separately",
    "a retry is any attempt logged after an error attempt of the same request_id (global line order); more than "
    "two retries or more than three error attempts per request_id fail retry_limit",
    "an unparsed successful generation at repeat r < 2 must be followed by a logged line (any cache_status) for "
    "repeat r + 1; an errored repeat does not oblige a further repeat",
    "a replaced slot's tensor hash depends only on (input, family, slot), since every mask is rebuilt from the "
    "original frames; two hashes for one (input, family, slot) fail slot_identity",
    "manifest window records with status 'unavailable' have no window and are not checked against VERA arithmetic",
]


def validate(run_dir, exp02_dir=None, seed=None, B=None):
    run = load_run(run_dir)
    checks = [check_run_files(run)]
    if not run.inputs and not run.lines:
        return _report(run, checks, {}, {})
    auth = build_authoritative(run.lines)
    store = Store(auth)
    calcs = [compute_input(inp, store) for inp in run.inputs if isinstance(inp.get("window_uid"), str)]
    seen, uniq = set(), []
    for c in calcs:
        if c["window_uid"] not in seen:
            seen.add(c["window_uid"])
            uniq.append(c)
    calcs = uniq
    cfg_seed, cfg_B, bp_info = bootstrap_params(run.config)
    seed = cfg_seed if seed is None else seed
    B = cfg_B if B is None else B
    primary = compute_primary(calcs, seed, B)

    checks += [check_records_schema(run), check_result_key_consistency(run), check_repeat_rule(run, auth),
               check_retry_limit(run), check_score_rounding(run), check_inputs(run), check_window_arithmetic(run),
               check_slot_hashes_present(run), check_slot_identity(run), check_mask_hash_consistency(run),
               check_planned_requests(run, calcs),
               check_cross_family_isolation(calcs, auth, run.inputs_by_uid)]
    rk_check, theirs = check_rankings_compare(run, calcs)
    checks.append(check_adaptive_invariants(calcs, theirs))
    checks.append(rk_check)
    checks += [check_primary_compare(run, primary), check_inputs_metrics_compare(run, calcs),
               check_pairs_compare(run, calcs), check_trajectories_compare(run, calcs)]
    e1_check, e1_summary = check_e1(exp02_dir)
    checks.append(e1_check)

    counts = {"n_inputs": len(calcs),
              "n_inputs_status_ok": sum(1 for c in calcs if c["status"] == "ok"),
              "n_label_available": sum(1 for c in calcs if c["label_available"]),
              "n_L0_1": sum(1 for c in calcs if c["L0"] == 1),
              "n_L0_0": sum(1 for c in calcs if c["L0"] == 0),
              "n_L0_missing": sum(1 for c in calcs if c["L0"] is None),
              "n_development_inputs": sum(1 for c in calcs if c["development"]),
              "n_request_lines": len(run.lines),
              "n_successful_lines": sum(1 for r in run.lines if is_success(r)),
              "n_authoritative_tuples": len(auth),
              "cache_status": {s: sum(1 for r in run.lines if r.get("cache_status") == s) for s in CACHE_STATUSES},
              "all_tied_blur": sum(1 for c in calcs if c["label_available"]
                                   and ranking_complete(c["rankings"]["blur"]) and c["rankings"]["blur"]["all_tied"]),
              "all_nonpositive_blur": sum(1 for c in calcs if c["label_available"]
                                          and ranking_complete(c["rankings"]["blur"])
                                          and c["rankings"]["blur"]["all_nonpositive"]),
              "tie_spans_cut2_blur": sum(1 for c in calcs if c["label_available"]
                                         and ranking_complete(c["rankings"]["blur"])
                                         and c["rankings"]["blur"]["tie_spans_cut"]["2"])}
    per_input = [{k: c[k] for k in ("window_uid", "video_id", "development", "L0", "q", "T_w", "d_w", "d_lower",
                                    "d_upper", "F_top2", "n_pair_labels", "e_w", "tau_blur_grey", "tau_native_hi",
                                    "overlap_blur_grey", "overlap_native_hi")}
                 for c in calcs]
    for row, c in zip(per_input, calcs):
        row["order_blur"] = c["rankings"]["blur"].get("order") if c["rankings"] else None
        row["order_adaptive"] = c["adaptive"].get("order") if c["adaptive"] else None
    recomputed = {"primary": primary, "bootstrap_params": dict(bp_info, seed=seed, B=B), "counts": counts,
                  "e1": e1_summary, "inputs": per_input}
    return _report(run, checks, recomputed, {})


def _report(run, checks, recomputed, extra):
    cj = [c.to_json() for c in checks]
    statuses = [c["status"] for c in cj]
    report = {"validator": VALIDATOR_VERSION, "protocol_version": PROTOCOL_VERSION, "run_dir": run.run_dir,
              "passed": all(s == "passed" for s in statuses),
              "summary": {s: statuses.count(s) for s in ("passed", "failed", "skipped")},
              "checks": cj, "recomputed": recomputed, "readings": AMBIGUITIES,
              "provenance": {"sha256": run.provenance}}
    report.update(extra)
    return _jsonable(report)


def exit_code(report):
    if report["passed"]:
        return 0
    if report["summary"].get("failed"):
        return 1
    return 3


def main(argv=None):
    ap = argparse.ArgumentParser(description="experiment03B independent validator (standard library only)")
    ap.add_argument("--run-dir", required=True, help="runs/<run_id>")
    ap.add_argument("--out", default=None, help="output file (default <run-dir>/validation/validation.json)")
    ap.add_argument("--exp02-run-dir", default=None, help="experiments/experiment02/runs/run_002 for erratum E1")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.run_dir):
        print("validate3b: run dir not found: %s" % args.run_dir, file=sys.stderr)
        return 2
    report = validate(args.run_dir, args.exp02_run_dir)
    out = args.out or os.path.join(args.run_dir, "validation", "validation.json")
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=False, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, out)
    s = report["summary"]
    print("validate3b: passed=%s  passed:%d failed:%d skipped:%d  -> %s"
          % (report["passed"], s.get("passed", 0), s.get("failed", 0), s.get("skipped", 0), out))
    for c in report["checks"]:
        if c["status"] != "passed":
            print("  %-26s %-7s n_checked=%d n_discrepancies=%d%s" % (
                c["id"], c["status"], c["n_checked"], c["n_discrepancies"],
                ("  (" + c.get("skip_reason", "") + ")") if c["status"] == "skipped" else ""))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())

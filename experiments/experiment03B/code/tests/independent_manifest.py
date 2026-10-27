#!/usr/bin/env python3
"""Independent re-derivation of the experiment03B window manifest (PROTOCOL.md section 3).

Standard library only. Written from PROTOCOL.md section 3 and RECORDS.md alone, without reading the runner or the
analyzer, so its output can be cross-checked against the manifest they produce.

    python3 -I -B independent_manifest.py --ucf-eval <UCF_Eval.json> --listings <tsv>... --out <json>
        [--exp02-cases <experiment02 cases.json>]
        [--expect-total-frames N] [--expect-total-anchors N]

    python3 -I -B independent_manifest.py --self-test

Preflight failures (video/class/category/interval counts, clipped-interval count, development videos, malformed
labels) print every problem and exit with status 2 without writing the output file. Later problems (window-arithmetic
disagreements, invariant violations, videos missing from or duplicated across the listings, experiment02 mismatches)
are written into the output's "problems" list, printed, and also exit with status 2.
"""

import argparse
import collections
import fractions
import hashlib
import json
import math
import os
import sys

# ---------------------------------------------------------------- frozen constants (PROTOCOL.md section 3)

EXPECTED_VIDEOS = 290
EXPECTED_ANOMALOUS = 140
EXPECTED_NORMAL = 150
EXPECTED_INTERVALS = 162
EXPECTED_CLIPPED_INTERVALS = 5          # "end is clipped to the manifest length (this affects 5 intervals)"
EXPECTED_ANOMALY_CATEGORIES = frozenset([
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
])
EXPECTED_SMALL_CLASSES = {"Abuse": 2, "Assault": 3}   # PROTOCOL.md section 13
NORMAL_CATEGORY = "Normal"
DEVELOPMENT_VIDEOS = ("RoadAccidents127_x264", "Normal_Videos_018_x264")
MAX_WINDOW_RECORDS = 452

SAMPLING_RATE = 16      # vera_pipeline.anchors
FPS = 30                # hard-coded in VERA
T_WINDOW = 10           # seconds
HALF_WINDOW = FPS * T_WINDOW // 2   # 150 frames
SNIPPET_LEN = 8

STRATA = ("event", "negative", "normal")

# Archive name for each listing's top-level directory. The test-normal archive's name differs from its folder
# (provenance: experiment02 run_002 cases.json, source_url ".../Testing_Normal_Videos.zip" with member
# "Testing_Normal_Videos_Anomaly/...").
ZIP_FOR_TOPDIR = {
    "Anomaly-Videos-Part-1": "Anomaly-Videos-Part-1.zip",
    "Anomaly-Videos-Part-2": "Anomaly-Videos-Part-2.zip",
    "Anomaly-Videos-Part-3": "Anomaly-Videos-Part-3.zip",
    "Anomaly-Videos-Part-4": "Anomaly-Videos-Part-4.zip",
    "Testing_Normal_Videos_Anomaly": "Testing_Normal_Videos.zip",
}


class PreflightError(Exception):
    pass


# ---------------------------------------------------------------- VERA arithmetic

def anchors_for(length):
    """vera_pipeline.anchors: range(0, length, 16)."""
    return list(range(0, length, SAMPLING_RATE))


def numpy_style_window(a, length):
    """Pure-Python float reproduction of vera_pipeline.window_for_anchor.

    snippet_start = max(a - 150.0, 0); snippet_end = min(a + 150.0, length);
    np.arange(s, e, (e - s)/8).astype(int).

    numpy's float arange takes len = ceil((stop - start)/step), writes buffer[0] = start, buffer[1] = start + step,
    then fills buffer[i] = start + i*delta with delta = buffer[1] - buffer[0] for i >= 2. That exact sequence is
    reproduced here with Python floats (IEEE-754 binary64, the same as float64). astype(int) truncates toward zero.
    """
    s = max(a - 1 / 2 * FPS * T_WINDOW, 0)
    e = min(a + 1 / 2 * FPS * T_WINDOW, length)
    s = float(s)
    e = float(e)
    step = (e - s) / SNIPPET_LEN
    if step == 0:
        return s, e, []
    n = math.ceil((e - s) / step)
    if n <= 0:
        return s, e, []
    vals = [s]
    if n >= 2:
        vals.append(s + step)
    if n >= 3:
        delta = vals[1] - vals[0]
        for i in range(2, n):
            vals.append(s + i * delta)
    return s, e, [int(v) for v in vals]


def simple_float_window(a, length):
    """The arithmetic as literally stated in the task: frames = [int(s + i*step) for i in range(ceil((e-s)/step))]."""
    s = float(max(a - 150, 0))
    e = float(min(a + 150, length))
    step = (e - s) / 8
    n = math.ceil((e - s) / step)
    return [int(s + i * step) for i in range(n)]


def exact_window(a, length):
    """Exact rational reference: s + i*(e-s)/8 for i in 0..7, floored (all values are non-negative)."""
    s = max(a - HALF_WINDOW, 0)
    e = min(a + HALF_WINDOW, length)
    step = fractions.Fraction(e - s, SNIPPET_LEN)
    n = math.ceil(fractions.Fraction(e - s) / step)
    return s, e, [math.floor(s + i * step) for i in range(n)]


def window_for(a, length):
    """Return the window dict for anchor a, verifying that float and exact arithmetic agree on exactly 8 frames."""
    fs, fe, f_np = numpy_style_window(a, length)
    f_simple = simple_float_window(a, length)
    s, e, f_exact = exact_window(a, length)
    problems = []
    if len(f_np) != SNIPPET_LEN:
        problems.append("numpy-style float arange gives %d frames" % len(f_np))
    if len(f_simple) != SNIPPET_LEN:
        problems.append("simple float formula gives %d frames" % len(f_simple))
    if len(f_exact) != SNIPPET_LEN:
        problems.append("exact rational arithmetic gives %d frames" % len(f_exact))
    if f_np != f_exact or f_simple != f_exact:
        problems.append("float/exact disagreement np=%s simple=%s exact=%s" % (f_np, f_simple, f_exact))
    if fs != float(s) or fe != float(e):
        problems.append("window bound disagreement")
    if any(not (0 <= f < length) for f in f_exact):
        problems.append("frame index outside [0, length)")
    if any(f_exact[i] >= f_exact[i + 1] for i in range(len(f_exact) - 1)):
        problems.append("frame indices not strictly increasing")
    if any(not (s <= f < e) for f in f_exact):
        problems.append("frame index outside window [s, e)")
    clipped_start = a - HALF_WINDOW < 0
    clipped_end = a + HALF_WINDOW > length
    return {
        "window": [s, e],
        "frame_indices": f_exact,
        "clipped": bool(clipped_start or clipped_end),
        "clipped_start": bool(clipped_start),
        "clipped_end": bool(clipped_end),
        "float_frame_count": len(f_np),
        "float_simple_frame_count": len(f_simple),
        "arith_problems": problems,
    }


# ---------------------------------------------------------------- population

def load_population(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    data = json.loads(raw.decode("utf-8"))
    problems = []
    if not isinstance(data, list):
        raise PreflightError("UCF_Eval.json is not a JSON list")
    videos = []
    seen = set()
    for pos, entry in enumerate(data):
        vpath = entry.get("video")
        tl = entry.get("temporal_label")
        length = entry.get("length")
        qa = entry.get("QA")
        if not isinstance(vpath, str) or not isinstance(tl, list) or not isinstance(length, int) or isinstance(length, bool):
            problems.append("entry %d malformed" % pos)
            continue
        parts = vpath.split("/")
        if len(parts) != 3 or not parts[2].endswith(".mp4"):
            problems.append("entry %d: unexpected video path %r" % (pos, vpath))
            continue
        category = parts[1]
        video_id = parts[2][: -len(".mp4")]
        if video_id in seen:
            problems.append("duplicate video_id %s" % video_id)
        seen.add(video_id)
        if length <= 0:
            problems.append("%s: non-positive length %d" % (video_id, length))
        if len(tl) % 2 != 0 or any(not isinstance(t, int) or isinstance(t, bool) for t in tl):
            problems.append("%s: malformed temporal_label %r" % (video_id, tl))
            continue
        intervals = []
        raw_pairs = []
        n_clipped_end = 0
        for k in range(0, len(tl), 2):
            start, end = tl[k], tl[k + 1]
            if start == -1:
                if end != -1:
                    problems.append("%s: half-missing pair %r" % (video_id, (start, end)))
                continue
            raw_pairs.append([start, end])
            clipped_end = min(end, length)
            if clipped_end != end:
                n_clipped_end += 1
            if not (0 <= start < clipped_end):
                problems.append("%s: empty or negative interval %r (length %d)" % (video_id, (start, end), length))
            intervals.append({"start": start, "end": clipped_end, "raw_end": end,
                              "end_clipped": clipped_end != end})
        for i in range(len(intervals) - 1):
            if intervals[i]["end"] > intervals[i + 1]["start"]:
                problems.append("%s: overlapping or unordered intervals" % video_id)
        answer = None
        if isinstance(qa, list) and qa and isinstance(qa[0], dict):
            answer = qa[0].get("a")
        is_normal = category == NORMAL_CATEGORY
        if is_normal and intervals:
            problems.append("%s: normal video with intervals" % video_id)
        if not is_normal and not intervals:
            problems.append("%s: anomalous video without intervals" % video_id)
        if answer not in ("Yes.", "No."):
            problems.append("%s: unexpected QA answer %r" % (video_id, answer))
        elif (answer == "No.") != is_normal:
            problems.append("%s: QA answer %r disagrees with category %s" % (video_id, answer, category))
        videos.append({
            "manifest_position": pos,
            "video_id": video_id,
            "video_path": vpath,
            "category": category,
            "label": "normal" if is_normal else "anomalous",
            "length": length,
            "temporal_label": tl,
            "intervals": intervals,
            "n_clipped_end": n_clipped_end,
        })
    return videos, problems, hashlib.sha256(raw).hexdigest()


def preflight_counts(videos, args):
    problems = []
    n_anom = sum(1 for v in videos if v["label"] == "anomalous")
    n_norm = sum(1 for v in videos if v["label"] == "normal")
    cats = collections.Counter(v["category"] for v in videos)
    anomaly_cats = set(cats) - {NORMAL_CATEGORY}
    n_intervals = sum(len(v["intervals"]) for v in videos)
    n_clipped = sum(v["n_clipped_end"] for v in videos)
    total_frames = sum(v["length"] for v in videos)
    total_anchors = sum(len(range(0, v["length"], SAMPLING_RATE)) for v in videos)
    checks = [
        ("videos", len(videos), EXPECTED_VIDEOS),
        ("anomalous videos", n_anom, EXPECTED_ANOMALOUS),
        ("normal videos", n_norm, EXPECTED_NORMAL),
        ("anomaly categories", len(anomaly_cats), len(EXPECTED_ANOMALY_CATEGORIES)),
        ("Normal category videos", cats.get(NORMAL_CATEGORY, 0), EXPECTED_NORMAL),
        ("intervals", n_intervals, EXPECTED_INTERVALS),
        ("intervals with end clipped to length", n_clipped, EXPECTED_CLIPPED_INTERVALS),
    ]
    for cat, n in EXPECTED_SMALL_CLASSES.items():
        checks.append(("%s videos" % cat, cats.get(cat, 0), n))
    if args.expect_total_frames is not None:
        checks.append(("total frames", total_frames, args.expect_total_frames))
    if args.expect_total_anchors is not None:
        checks.append(("total anchors", total_anchors, args.expect_total_anchors))
    for name, got, want in checks:
        if got != want:
            problems.append("PREFLIGHT %s: got %d, expected %d" % (name, got, want))
    if anomaly_cats != EXPECTED_ANOMALY_CATEGORIES:
        problems.append("PREFLIGHT anomaly category set differs: extra %s missing %s" % (
            sorted(anomaly_cats - EXPECTED_ANOMALY_CATEGORIES), sorted(EXPECTED_ANOMALY_CATEGORIES - anomaly_cats)))
    ids = {v["video_id"] for v in videos}
    for dev in DEVELOPMENT_VIDEOS:
        if dev not in ids:
            problems.append("PREFLIGHT development video %s not in manifest" % dev)
    counts = {
        "videos": len(videos),
        "anomalous_videos": n_anom,
        "normal_videos": n_norm,
        "anomaly_categories": len(anomaly_cats),
        "categories_including_normal": len(cats),
        "videos_per_category": dict(sorted(cats.items())),
        "intervals": n_intervals,
        "intervals_end_clipped_to_length": n_clipped,
        "videos_with_2_intervals": sum(1 for v in videos if len(v["intervals"]) == 2),
        "total_frames": total_frames,
        "total_frames_anomalous": sum(v["length"] for v in videos if v["label"] == "anomalous"),
        "total_frames_normal": sum(v["length"] for v in videos if v["label"] == "normal"),
        "total_anchors": total_anchors,
        "total_anchors_anomalous": sum(len(range(0, v["length"], SAMPLING_RATE))
                                       for v in videos if v["label"] == "anomalous"),
        "total_anchors_normal": sum(len(range(0, v["length"], SAMPLING_RATE))
                                    for v in videos if v["label"] == "normal"),
        "total_annotated_frames": sum(iv["end"] - iv["start"] for v in videos for iv in v["intervals"]),
    }
    return counts, problems


# ---------------------------------------------------------------- listings

def load_listings(paths):
    by_video = {}
    duplicates = []
    per_listing = []
    problems = []
    for path in paths:
        with open(path, "rb") as fh:
            raw = fh.read()
        text = raw.decode("utf-8")
        n_members = 0
        n_mp4 = 0
        topdirs = set()
        for lineno, line in enumerate(text.split("\n"), 1):
            if line == "":
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                problems.append("%s:%d: expected 2 tab-separated fields" % (path, lineno))
                continue
            size_s, member = fields
            try:
                size = int(size_s)
            except ValueError:
                problems.append("%s:%d: non-integer size %r" % (path, lineno, size_s))
                continue
            n_members += 1
            topdirs.add(member.split("/")[0])
            if member.endswith("/"):
                continue
            if not member.endswith(".mp4"):
                continue
            n_mp4 += 1
            topdir = member.split("/")[0]
            zipname = ZIP_FOR_TOPDIR.get(topdir)
            if zipname is None:
                problems.append("%s:%d: unknown archive top-level directory %r" % (path, lineno, topdir))
            vid = member.rsplit("/", 1)[-1][: -len(".mp4")]
            rec = {"zip": zipname, "zip_member": member, "video_bytes": size,
                   "listing": os.path.basename(path), "listing_sha256": hashlib.sha256(raw).hexdigest()}
            if vid in by_video:
                duplicates.append({"video_id": vid, "first": by_video[vid], "second": rec})
            else:
                by_video[vid] = rec
        per_listing.append({"listing": os.path.basename(path), "sha256": hashlib.sha256(raw).hexdigest(),
                            "members": n_members, "mp4_members": n_mp4, "top_level": sorted(topdirs)})
    return by_video, duplicates, per_listing, problems


# ---------------------------------------------------------------- window selection

def overlaps(ws, we, iv):
    return ws < iv["end"] and iv["start"] < we


def gap_to(ws, we, iv):
    """Gap between the half-open window [ws, we) and the half-open interval; only called when they do not overlap."""
    if we <= iv["start"]:
        return iv["start"] - we
    return ws - iv["end"]


def nearest_anchor_twice_target(anchors, twice_target):
    """Anchor nearest target (given as 2*target to stay exact); on a tie, the earlier anchor."""
    best = None
    best_d = None
    for a in anchors:
        d = abs(2 * a - twice_target)
        if best is None or d < best_d:      # strict '<' keeps the earlier anchor on ties
            best, best_d = a, d
    return best


def slots_in(frames, intervals):
    return [i for i, f in enumerate(frames) if any(iv["start"] <= f < iv["end"] for iv in intervals)]


def build_windows(videos):
    records = []
    arith_problems = []
    for v in videos:
        anc = anchors_for(v["length"])
        v["n_anchors"] = len(anc)
        if v["label"] != "anomalous":
            continue
        for k, iv in enumerate(v["intervals"]):
            a = nearest_anchor_twice_target(anc, iv["start"] + iv["end"])
            w = window_for(a, v["length"])
            cov = [i for i, f in enumerate(w["frame_indices"]) if iv["start"] <= f < iv["end"]]
            records.append(dict(
                window_record_id="event:%s:%d" % (v["video_id"], k),
                stratum="event", video_id=v["video_id"], category=v["category"], anchor=a,
                interval=[iv["start"], iv["end"]], interval_index=k,
                interval_coverage=len(cov), interval_coverage_slots=cov,
                annotated_slots=slots_in(w["frame_indices"], v["intervals"]),
                gap_to_annotation=None, **w))
    for v in videos:
        if v["label"] != "anomalous":
            continue
        L = v["length"]
        best = None
        best_gap = None
        n_eligible = 0
        for a in anchors_for(L):
            ws, we = a - HALF_WINDOW, a + HALF_WINDOW
            if ws < 0 or we > L:
                continue
            if any(overlaps(ws, we, iv) for iv in v["intervals"]):
                continue
            n_eligible += 1
            g = min(gap_to(ws, we, iv) for iv in v["intervals"])
            if best is None or g > best_gap:   # strict '>' keeps the earlier anchor on ties
                best, best_gap = a, g
        rid = "negative:%s" % v["video_id"]
        if best is None:
            records.append(dict(window_record_id=rid, stratum="negative", video_id=v["video_id"],
                                category=v["category"], anchor=None, window=None, frame_indices=None,
                                clipped=None, clipped_start=None, clipped_end=None, interval=None,
                                interval_index=None, interval_coverage=None, interval_coverage_slots=None,
                                annotated_slots=None, gap_to_annotation=None, n_eligible_anchors=0,
                                float_frame_count=None, float_simple_frame_count=None, arith_problems=[],
                                status="unavailable"))
            continue
        w = window_for(best, L)
        records.append(dict(window_record_id=rid, stratum="negative", video_id=v["video_id"],
                            category=v["category"], anchor=best, interval=None, interval_index=None,
                            interval_coverage=None, interval_coverage_slots=None,
                            annotated_slots=slots_in(w["frame_indices"], v["intervals"]),
                            gap_to_annotation=best_gap, n_eligible_anchors=n_eligible, **w))
    for v in videos:
        if v["label"] != "normal":
            continue
        a = nearest_anchor_twice_target(anchors_for(v["length"]), v["length"])
        w = window_for(a, v["length"])
        records.append(dict(window_record_id="normal:%s" % v["video_id"], stratum="normal",
                            video_id=v["video_id"], category=v["category"], anchor=a, interval=None,
                            interval_index=None, interval_coverage=None, interval_coverage_slots=None,
                            annotated_slots=slots_in(w["frame_indices"], v["intervals"]),
                            gap_to_annotation=None, **w))
    for r in records:
        r["development"] = r["video_id"] in DEVELOPMENT_VIDEOS
        if r.get("status") == "unavailable":
            r["window_uid"] = None
            continue
        r["window_uid"] = "%s@%d" % (r["video_id"], r["anchor"])
        if r["arith_problems"]:
            arith_problems.append({"window_record_id": r["window_record_id"], "problems": r["arith_problems"]})
    return records, arith_problems


def deduplicate(records):
    unique = collections.OrderedDict()
    for r in records:
        uid = r["window_uid"]
        if uid is None:
            continue
        if uid not in unique:
            unique[uid] = {"window_uid": uid, "video_id": r["video_id"], "category": r["category"],
                           "anchor": r["anchor"], "frame_indices": r["frame_indices"], "window": r["window"],
                           "clipped": r["clipped"], "annotated_slots": r["annotated_slots"],
                           "development": r["development"], "window_record_ids": [], "strata": []}
            r["status"] = "selected"
        else:
            u = unique[uid]
            if u["frame_indices"] != r["frame_indices"] or u["window"] != r["window"]:
                raise PreflightError("window_uid %s maps to two different frame sets" % uid)
            r["status"] = "duplicate_input"
        u = unique[uid]
        u["window_record_ids"].append(r["window_record_id"])
        if r["stratum"] not in u["strata"]:
            u["strata"].append(r["stratum"])
    for u in unique.values():
        u["strata"] = [s for s in STRATA if s in u["strata"]]
    # Inputs are (video_id, frame_indices) pairs; flag distinct window_uids that sample identical frames.
    by_frames = collections.defaultdict(list)
    for u in unique.values():
        by_frames[(u["video_id"], tuple(u["frame_indices"]))].append(u["window_uid"])
    identical_frames = [uids for uids in by_frames.values() if len(uids) > 1]
    return list(unique.values()), identical_frames


# ---------------------------------------------------------------- invariants and summaries

def check_invariants(videos, records, unique):
    problems = []
    vid = {v["video_id"]: v for v in videos}
    for r in records:
        if r["status"] == "unavailable":
            continue
        v = vid[r["video_id"]]
        a = r["anchor"]
        if a % SAMPLING_RATE != 0 or not (0 <= a < v["length"]):
            problems.append("%s: anchor %d not a VERA anchor" % (r["window_record_id"], a))
        if r["stratum"] == "negative":
            ws, we = r["window"]
            if r["clipped"] or ws != a - HALF_WINDOW or we != a + HALF_WINDOW:
                problems.append("%s: negative window not full length" % r["window_record_id"])
            if any(overlaps(ws, we, iv) for iv in v["intervals"]):
                problems.append("%s: negative window overlaps an interval" % r["window_record_id"])
            if r["annotated_slots"]:
                problems.append("%s: negative window has annotated slots" % r["window_record_id"])
            if r["gap_to_annotation"] is None or r["gap_to_annotation"] < 0:
                problems.append("%s: bad gap" % r["window_record_id"])
        if r["stratum"] == "normal" and r["annotated_slots"]:
            problems.append("%s: normal window has annotated slots" % r["window_record_id"])
        if r["stratum"] in ("event", "normal"):
            # Nearest VERA anchor to the target (interval midpoint / length/2), earlier anchor on a tie. |2a - t2| is
            # convex over the anchor grid, so it suffices that the previous anchor is strictly farther and the next
            # anchor is not nearer. (A "within 8 frames" bound is wrong after the last anchor, which can be up to 15
            # frames before the end, and it cannot see the tie-break.)
            t2 = r["interval"][0] + r["interval"][1] if r["stratum"] == "event" else v["length"]
            d2 = abs(2 * a - t2)
            prev_a, next_a = a - SAMPLING_RATE, a + SAMPLING_RATE
            if (prev_a >= 0 and abs(2 * prev_a - t2) <= d2) or (next_a < v["length"] and abs(2 * next_a - t2) < d2):
                problems.append("%s: anchor %d is not the nearest VERA anchor to the target (earlier on a tie)" % (
                    r["window_record_id"], a))
    n_by = collections.Counter(r["stratum"] for r in records)
    if n_by["event"] != EXPECTED_INTERVALS:
        problems.append("event window records %d != %d" % (n_by["event"], EXPECTED_INTERVALS))
    if n_by["negative"] != EXPECTED_ANOMALOUS:
        problems.append("negative window records %d != %d" % (n_by["negative"], EXPECTED_ANOMALOUS))
    if n_by["normal"] != EXPECTED_NORMAL:
        problems.append("normal window records %d != %d" % (n_by["normal"], EXPECTED_NORMAL))
    if len(records) > MAX_WINDOW_RECORDS:
        problems.append("window records %d > %d" % (len(records), MAX_WINDOW_RECORDS))
    if sum(len(u["window_record_ids"]) for u in unique) != sum(1 for r in records if r["window_uid"]):
        problems.append("unique-input links do not cover every available record")
    # An input is a (video_id, frame_indices) pair (PROTOCOL.md section 3); two window_uids sampling the same frames
    # would be one input counted twice, which window_uid-keyed deduplication cannot resolve.
    first_uid = {}
    for u in unique:
        key = (u["video_id"], tuple(u["frame_indices"]))
        if key in first_uid:
            problems.append("window_uids %s and %s sample identical frames (one input)" % (first_uid[key], u["window_uid"]))
        first_uid.setdefault(key, u["window_uid"])
    return problems


def check_exp02(cases_path, videos):
    """Compare this file's window arithmetic with frame indices recorded by experiment02's VERA (numpy) pipeline."""
    with open(cases_path, "rb") as fh:
        data = json.loads(fh.read().decode("utf-8"))
    vid = {v["video_id"]: v for v in videos}
    out = []
    for c in data.get("cases", []):
        v = vid.get(c.get("video_id"))
        if v is None:
            out.append({"case_id": c.get("case_id"), "match": False, "reason": "video not in manifest"})
            continue
        w = window_for(c["anchor"], v["length"])
        recorded = [c.get("window_start"), c.get("window_end")]
        match = (w["frame_indices"] == c.get("frame_indices")
                 and [float(x) for x in w["window"]] == [float(x) for x in recorded]
                 and c.get("decoded_frames", v["length"]) == v["length"])
        out.append({"case_id": c.get("case_id"), "video_id": c["video_id"], "anchor": c["anchor"],
                    "recorded_frames": c.get("frame_indices"), "derived_frames": w["frame_indices"],
                    "recorded_window": recorded, "derived_window": w["window"],
                    "recorded_decoded_frames": c.get("decoded_frames"), "manifest_length": v["length"],
                    "match": bool(match)})
    return out


def all_anchor_arithmetic(videos):
    """Run the float-vs-exact window check on every anchor of every video (not only the selected ones)."""
    n = 0
    bad = []
    for v in videos:
        for a in anchors_for(v["length"]):
            n += 1
            w = window_for(a, v["length"])
            if w["arith_problems"]:
                bad.append({"video_id": v["video_id"], "anchor": a, "problems": w["arith_problems"]})
    return n, bad


def summarise(videos, records, unique, identical_frames, listing_info):
    by_stratum = collections.defaultdict(list)
    for r in records:
        by_stratum[r["stratum"]].append(r)
    uniq_by_stratum = {s: sum(1 for u in unique if s in u["strata"]) for s in STRATA}
    sharing = {}
    for s in STRATA:
        recs = [r for r in by_stratum[s] if r["window_uid"]]
        cnt = collections.Counter(r["window_uid"] for r in recs)
        sharing[s] = {
            "records": len(by_stratum[s]),
            "available_records": len(recs),
            "unique_inputs": len(cnt),
            "records_sharing_an_input": sum(n for n in cnt.values() if n > 1),
            "inputs_shared_by_2plus_records": sum(1 for n in cnt.values() if n > 1),
            "duplicate_input_records": sum(1 for r in recs if r["status"] == "duplicate_input"),
        }
    cnt_all = collections.Counter(r["window_uid"] for r in records if r["window_uid"])
    sharing["overall"] = {
        "records": len(records),
        "available_records": sum(cnt_all.values()),
        "unique_inputs": len(unique),
        "records_sharing_an_input": sum(n for n in cnt_all.values() if n > 1),
        "inputs_shared_by_2plus_records": sum(1 for n in cnt_all.values() if n > 1),
        "duplicate_input_records": sum(1 for r in records if r["status"] == "duplicate_input"),
        "inputs_in_more_than_one_stratum": sum(1 for u in unique if len(u["strata"]) > 1),
    }
    ev = by_stratum["event"]
    coverage_hist = collections.Counter(r["interval_coverage"] for r in ev)
    annotated_hist = collections.Counter(len(r["annotated_slots"]) for r in ev)
    ev_unique = [u for u in unique if "event" in u["strata"]]
    windows = {
        "window_records": len(records),
        "records_by_stratum": {s: len(by_stratum[s]) for s in STRATA},
        "unique_inputs": len(unique),
        "unique_inputs_by_stratum": uniq_by_stratum,
        "unique_inputs_excluding_development": sum(1 for u in unique if not u["development"]),
        "unique_inputs_development": sum(1 for u in unique if u["development"]),
        "negative_unavailable": sum(1 for r in by_stratum["negative"] if r["status"] == "unavailable"),
        "negative_unavailable_videos": [r["video_id"] for r in by_stratum["negative"] if r["status"] == "unavailable"],
        "clipped_records_by_stratum": {s: sum(1 for r in by_stratum[s] if r["clipped"]) for s in STRATA},
        "clipped_unique_inputs_by_stratum": {s: sum(1 for u in unique if s in u["strata"] and u["clipped"])
                                             for s in STRATA},
        "clipped_start_records_by_stratum": {s: sum(1 for r in by_stratum[s] if r["clipped_start"]) for s in STRATA},
        "clipped_end_records_by_stratum": {s: sum(1 for r in by_stratum[s] if r["clipped_end"]) for s in STRATA},
        "event_records_zero_annotated_slots": sum(1 for r in ev if not r["annotated_slots"]),
        "event_unique_inputs_zero_annotated_slots": sum(1 for u in ev_unique if not u["annotated_slots"]),
        "event_records_zero_own_interval_coverage": sum(1 for r in ev if r["interval_coverage"] == 0),
        "event_records_full_own_interval_coverage": sum(1 for r in ev if r["interval_coverage"] == SNIPPET_LEN),
        "event_own_interval_coverage_histogram": {str(k): coverage_hist[k] for k in sorted(coverage_hist)},
        "event_annotated_slot_count_histogram": {str(k): annotated_hist[k] for k in sorted(annotated_hist)},
        "event_records_with_annotated_slots_from_other_interval": sum(
            1 for r in ev if set(r["annotated_slots"]) != set(r["interval_coverage_slots"])),
        "negative_gap_min_max": ([min(r["gap_to_annotation"] for r in by_stratum["negative"] if r["window_uid"]),
                                  max(r["gap_to_annotation"] for r in by_stratum["negative"] if r["window_uid"])]
                                 if any(r["window_uid"] for r in by_stratum["negative"]) else None),
        "float_arithmetic_non8_windows": sum(1 for r in records if r["window_uid"] and (
            r["float_frame_count"] != SNIPPET_LEN or r["float_simple_frame_count"] != SNIPPET_LEN)),
        "distinct_uids_with_identical_frames": identical_frames,
        "sharing": sharing,
    }
    return windows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ucf-eval", required=True)
    ap.add_argument("--listings", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp02-cases", default=None,
                    help="optional experiment02 cases.json: recorded VERA frame indices to cross-check")
    ap.add_argument("--expect-total-frames", type=int, default=None)
    ap.add_argument("--expect-total-anchors", type=int, default=None)
    args = ap.parse_args(argv)

    problems = []
    videos, p, eval_sha = load_population(args.ucf_eval)
    problems += p
    counts, p = preflight_counts(videos, args)
    problems += p
    if problems:
        for msg in problems:
            print("FAIL", msg, file=sys.stderr)
        print("preflight failed: %d problem(s); no output written" % len(problems), file=sys.stderr)
        return 2

    listing, dup_listing, per_listing, lp = load_listings(args.listings)
    problems += lp
    records, arith = build_windows(videos)
    unique, identical_frames = deduplicate(records)
    problems += check_invariants(videos, records, unique)
    if arith:
        problems.append("window arithmetic problems on %d selected window(s): %s" % (len(arith), arith[:5]))
    n_all_anchor, bad_all_anchor = all_anchor_arithmetic(videos)
    if bad_all_anchor:
        problems.append("window arithmetic problems on %d of %d anchors overall: %s" % (
            len(bad_all_anchor), n_all_anchor, bad_all_anchor[:5]))
    exp02 = check_exp02(args.exp02_cases, videos) if args.exp02_cases else None
    if exp02 is not None and not all(c["match"] for c in exp02):
        problems.append("experiment02 recorded windows disagree: %s" % [c for c in exp02 if not c["match"]])

    # Per-video records.
    out_videos = []
    missing = []
    for v in videos:
        li = listing.get(v["video_id"])
        if li is None:
            missing.append(v["video_id"])
        elif li["zip"] is not None and v["label"] == "normal" and not li["zip"].startswith("Testing_Normal"):
            problems.append("%s: normal video found in anomaly archive %s" % (v["video_id"], li["zip"]))
        elif li["zip"] is not None and v["label"] == "anomalous" and "/%s/" % v["category"] not in li["zip_member"]:
            problems.append("%s: archive member %s not under category %s" % (
                v["video_id"], li["zip_member"], v["category"]))
        out_videos.append({
            "video_id": v["video_id"], "category": v["category"], "label": v["label"], "length": v["length"],
            "manifest_position": v["manifest_position"], "video_path": v["video_path"],
            "temporal_label": v["temporal_label"],
            "intervals": [[iv["start"], iv["end"]] for iv in v["intervals"]],
            "intervals_end_clipped": [iv["end_clipped"] for iv in v["intervals"]],
            "n_anchors": v["n_anchors"],
            "zip": li["zip"] if li else None, "zip_member": li["zip_member"] if li else None,
            "video_bytes": li["video_bytes"] if li else None, "listing": li["listing"] if li else None,
            "development": v["video_id"] in DEVELOPMENT_VIDEOS,
            "window_record_ids": [r["window_record_id"] for r in records if r["video_id"] == v["video_id"]],
            "status": "selected",
        })
    if missing:
        problems.append("%d manifest video(s) missing from the listings: %s" % (len(missing), missing))
    manifest_ids = {v["video_id"] for v in videos}
    dup_manifest = sorted({dd["video_id"] for dd in dup_listing if dd["video_id"] in manifest_ids})
    if dup_manifest:
        problems.append("manifest video(s) listed more than once across listings (archive ambiguous): %s" % dup_manifest)
    present = [ov for ov in out_videos if ov["video_bytes"] is not None]
    largest = max(present, key=lambda ov: ov["video_bytes"]) if present else None
    downloads = {
        "listings": per_listing,
        "videos_found_in_listings": len(present),
        "videos_missing_from_listings": missing,
        "duplicate_basenames_across_listings": dup_listing,
        "total_bytes_all_videos": sum(ov["video_bytes"] for ov in present),
        "total_bytes_excluding_development": sum(ov["video_bytes"] for ov in present if not ov["development"]),
        "development_bytes": {ov["video_id"]: ov["video_bytes"] for ov in present if ov["development"]},
        "largest_video": ({"video_id": largest["video_id"], "zip": largest["zip"],
                           "zip_member": largest["zip_member"], "video_bytes": largest["video_bytes"]}
                          if largest else None),
        "smallest_video_bytes": min(ov["video_bytes"] for ov in present) if present else None,
        "bytes_note": "uncompressed member sizes from the listings; zip transfer sizes are not in the listings",
    }
    by_zip = collections.Counter()
    n_zip = collections.Counter()
    for ov in present:
        by_zip[ov["zip"]] += ov["video_bytes"]
        n_zip[ov["zip"]] += 1
    downloads["bytes_by_zip"] = dict(sorted(by_zip.items()))
    downloads["videos_by_zip"] = dict(sorted(n_zip.items()))

    window_summary = summarise(videos, records, unique, identical_frames, listing)
    out_windows = []
    for r in records:
        out_windows.append({k: r.get(k) for k in (
            "window_record_id", "stratum", "video_id", "category", "anchor", "window", "frame_indices",
            "clipped", "clipped_start", "clipped_end", "interval", "interval_index", "interval_coverage",
            "interval_coverage_slots", "annotated_slots", "gap_to_annotation", "n_eligible_anchors",
            "window_uid", "development", "float_frame_count", "status")})

    result = {
        "generator": "experiment03B/code/tests/independent_manifest.py",
        "protocol": "experiment03B PROTOCOL.md v1.0 section 3",
        "inputs_sha256": {"ucf_eval": eval_sha, **{pl["listing"]: pl["sha256"] for pl in per_listing}},
        "constants": {"sampling_rate": SAMPLING_RATE, "fps": FPS, "t_window_s": T_WINDOW,
                      "half_window_frames": HALF_WINDOW, "snippet_len": SNIPPET_LEN,
                      "development_videos": list(DEVELOPMENT_VIDEOS)},
        "counts": counts,
        "window_summary": window_summary,
        "arithmetic_check": {"selected_windows_checked": sum(1 for r in records if r["window_uid"]),
                             "selected_window_problems": arith,
                             "all_anchors_checked": n_all_anchor, "all_anchor_problems": bad_all_anchor},
        "exp02_crosscheck": exp02,
        "downloads": downloads,
        "problems": problems,
        "videos": out_videos,
        "windows": out_windows,
        "unique_inputs": [{"window_uid": u["window_uid"], "video_id": u["video_id"], "category": u["category"],
                           "anchor": u["anchor"], "window": u["window"], "frame_indices": u["frame_indices"],
                           "clipped": u["clipped"], "annotated_slots": u["annotated_slots"],
                           "development": u["development"], "window_record_ids": u["window_record_ids"],
                           "strata": u["strata"]} for u in unique],
    }
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, args.out)

    c = counts
    w = window_summary
    print("independent manifest (experiment03B PROTOCOL v1.0 section 3)")
    print("  videos %d (anomalous %d, normal %d); categories %d anomaly + Normal; intervals %d (%d end-clipped)" % (
        c["videos"], c["anomalous_videos"], c["normal_videos"], c["anomaly_categories"], c["intervals"],
        c["intervals_end_clipped_to_length"]))
    print("  total frames %d; total anchors %d" % (c["total_frames"], c["total_anchors"]))
    print("  window records %d: event %d, negative %d (unavailable %d), normal %d" % (
        w["window_records"], w["records_by_stratum"]["event"], w["records_by_stratum"]["negative"],
        w["negative_unavailable"], w["records_by_stratum"]["normal"]))
    print("  unique inputs %d: event %d, negative %d, normal %d; excluding development %d" % (
        w["unique_inputs"], w["unique_inputs_by_stratum"]["event"], w["unique_inputs_by_stratum"]["negative"],
        w["unique_inputs_by_stratum"]["normal"], w["unique_inputs_excluding_development"]))
    sh = w["sharing"]
    print("  records sharing an input: event %d, negative %d, normal %d, overall %d (duplicate_input records %d)" % (
        sh["event"]["records_sharing_an_input"], sh["negative"]["records_sharing_an_input"],
        sh["normal"]["records_sharing_an_input"], sh["overall"]["records_sharing_an_input"],
        sh["overall"]["duplicate_input_records"]))
    print("  clipped records: event %d, negative %d, normal %d" % tuple(
        w["clipped_records_by_stratum"][s] for s in STRATA))
    print("  event windows with zero annotated frames: %d records (%d unique); zero own-interval coverage %d" % (
        w["event_records_zero_annotated_slots"], w["event_unique_inputs_zero_annotated_slots"],
        w["event_records_zero_own_interval_coverage"]))
    print("  event own-interval coverage histogram: %s" % w["event_own_interval_coverage_histogram"])
    print("  window arithmetic: %d selected windows, %d with float count != 8; %d anchors overall, %d problems" % (
        result["arithmetic_check"]["selected_windows_checked"], w["float_arithmetic_non8_windows"],
        n_all_anchor, len(bad_all_anchor)))
    print("  distinct window_uids with identical frames: %d" % len(w["distinct_uids_with_identical_frames"]))
    if exp02 is not None:
        print("  experiment02 recorded windows: %d/%d match" % (sum(cc["match"] for cc in exp02), len(exp02)))
    d = downloads
    print("  listings: %d/%d videos found; missing %s; duplicate basenames %d" % (
        d["videos_found_in_listings"], c["videos"], d["videos_missing_from_listings"] or "none",
        len(d["duplicate_basenames_across_listings"])))
    print("  bytes: all %d (%.3f GB); excluding development %d (%.3f GB); largest %s %d (%.1f MB)" % (
        d["total_bytes_all_videos"], d["total_bytes_all_videos"] / 1e9, d["total_bytes_excluding_development"],
        d["total_bytes_excluding_development"] / 1e9, d["largest_video"]["video_id"] if d["largest_video"] else None,
        d["largest_video"]["video_bytes"] if d["largest_video"] else 0,
        (d["largest_video"]["video_bytes"] if d["largest_video"] else 0) / 1e6))
    print("  wrote %s" % args.out)
    if problems:
        for msg in problems:
            print("FAIL", msg, file=sys.stderr)
        print("%d problem(s) found" % len(problems), file=sys.stderr)
        return 2
    print("  OK: no problems")
    return 0


# ---------------------------------------------------------------- self-test (hand-worked fixtures and regressions)

def _fixture_video(video_id, category, length, intervals):
    return {"video_id": video_id, "category": category,
            "label": "normal" if category == NORMAL_CATEGORY else "anomalous", "length": length,
            "intervals": [{"start": s, "end": e, "raw_end": e, "end_clipped": False} for s, e in intervals],
            "n_clipped_end": 0}


def _synthetic_eval():
    """290 entries that pass the preflight: 140 anomalous (Abuse 2, Assault 3), 150 normal, 162 intervals (22 videos
    with two), 5 intervals with end > length, both development videos present."""
    cats = sorted(EXPECTED_ANOMALY_CATEGORIES)
    per_cat = dict(EXPECTED_SMALL_CLASSES)
    rest = [c for c in cats if c not in per_cat]
    left = EXPECTED_ANOMALOUS - sum(per_cat.values())
    for i, c in enumerate(rest):
        per_cat[c] = left // len(rest) + (1 if i < left % len(rest) else 0)
    entries = []
    k = 0
    for c in cats:
        for j in range(per_cat[c]):
            num = 127 if (c == "RoadAccidents" and j == 0) else j + 1
            tl = [200, 300, 600, 700] if k < 22 else ([800, 1005, -1, -1] if k < 27 else [400, 500, -1, -1])
            entries.append({"video": "frames/%s/%s%03d_x264.mp4" % (c, c, num), "temporal_label": tl, "length": 1000,
                            "QA": [{"q": "Is there any anomaly in the video?", "a": "Yes."}]})
            k += 1
    for j in range(EXPECTED_NORMAL):
        entries.append({"video": "frames/Normal/Normal_Videos_%03d_x264.mp4" % (18 + j), "temporal_label": [-1] * 4,
                        "length": 900, "QA": [{"q": "Is there any anomaly in the video?", "a": "No."}]})
    return entries


def _synthetic_listings(entries):
    anom = ["0\tAnomaly-Videos-Part-1/"]
    norm = ["0\tTesting_Normal_Videos_Anomaly/"]
    for n, e in enumerate(entries):
        _, cat, base = e["video"].split("/")
        if cat == NORMAL_CATEGORY:
            norm.append("%d\tTesting_Normal_Videos_Anomaly/%s" % (1000 + n, base))
        else:
            anom.append("%d\tAnomaly-Videos-Part-1/%s/%s" % (1000 + n, cat, base))
    return [("part1.tsv", anom), ("normal.tsv", norm)]


def self_test():
    import contextlib
    import io
    import tempfile

    results = []

    def check(name, got, want):
        results.append((name, got == want, got, want))

    def rec(recs, rid):
        return [r for r in recs if r["window_record_id"] == rid][0]

    # 1. Window arithmetic against frame indices recorded by VERA's numpy pipeline in experiment02 run_002.
    for a, length, frames in ((0, 2580, [0, 18, 37, 56, 75, 93, 112, 131]),
                              (496, 2580, [346, 383, 421, 458, 496, 533, 571, 608]),
                              (2576, 2580, [2426, 2445, 2464, 2483, 2503, 2522, 2541, 2560]),
                              (592, 1181, [442, 479, 517, 554, 592, 629, 667, 704])):
        w = window_for(a, length)
        check("exp02 frames anchor %d" % a, (w["frame_indices"], w["arith_problems"]), (frames, []))
    check("clipped iff a-150<0 or a+150>length", [window_for(a, 438)["clipped"] for a in (144, 160, 288, 304)],
          [True, False, False, True])   # 288 + 150 == 438 is full length, not clipped

    # 2. Hand-worked selection rules and tie-breaks (expected values computed by hand from PROTOCOL.md section 3).
    videos = [
        _fixture_video("Tie_event", "Arson", 1000, [(0, 16)]),       # midpoint 8: anchors 0 and 16 tie -> 0
        _fixture_video("Tie_neg", "Arson", 1158, [(574, 594)]),      # gap 264 at anchors 160 and 1008 -> 160
        _fixture_video("Max_neg", "Arson", 1000, [(480, 520)]),      # best before: 160 (gap 170); after: 848 (178)
        _fixture_video("Touch_neg", "Arson", 600, [(310, 320)]),     # only 160: [10, 310) touches [310, 320), gap 0
        _fixture_video("Same_anchor", "Arson", 1000, [(100, 110), (112, 118)]),   # both midpoints -> anchor 112
        _fixture_video("End_event", "Arson", 112, [(110, 112)]),     # midpoint 111, last anchor 96 (15 frames)
        _fixture_video("Normal_tie", "Normal", 48, []),              # 24: anchors 16 and 32 tie -> 16
        _fixture_video("Normal_300", "Normal", 300, []),             # 150: anchor 144, window [0, 294)
    ]
    recs, _ = build_windows(videos)
    uniq, _ = deduplicate(recs)
    check("event tie -> earlier anchor", rec(recs, "event:Tie_event:0")["anchor"], 0)
    check("negative gap tie -> earlier anchor", (rec(recs, "negative:Tie_neg")["anchor"],
                                                 rec(recs, "negative:Tie_neg")["gap_to_annotation"]), (160, 264))
    check("negative maximises gap", (rec(recs, "negative:Max_neg")["anchor"],
                                     rec(recs, "negative:Max_neg")["gap_to_annotation"]), (848, 178))
    check("negative half-open touch is eligible", (rec(recs, "negative:Touch_neg")["anchor"],
                                                   rec(recs, "negative:Touch_neg")["gap_to_annotation"]), (160, 0))
    check("no full-length window -> unavailable", (rec(recs, "negative:End_event")["status"],
                                                   rec(recs, "negative:End_event")["window_uid"]), ("unavailable", None))
    check("shared input statuses", [rec(recs, "event:Same_anchor:%d" % i)["status"] for i in (0, 1)],
          ["selected", "duplicate_input"])
    check("shared input counted once", [u["window_record_ids"] for u in uniq if u["window_uid"] == "Same_anchor@112"],
          [["event:Same_anchor:0", "event:Same_anchor:1"]])
    check("normal tie -> earlier anchor", rec(recs, "normal:Normal_tie")["anchor"], 16)
    check("normal window/frames", (rec(recs, "normal:Normal_300")["window"], rec(recs, "normal:Normal_300")["frame_indices"]),
          ([0, 294], [0, 36, 73, 110, 147, 183, 220, 257]))
    check("event coverage slots", (rec(recs, "event:Max_neg:0")["anchor"], rec(recs, "event:Max_neg:0")["interval_coverage_slots"]),
          (496, [4]))   # midpoint 500 -> 496; frames 346 + 37.5i = 346 383 421 458 496 533 571 608; only 496 in [480, 520)
    check("half-open coverage excludes the interval end", (rec(recs, "event:Touch_neg:0")["anchor"],
                                                           rec(recs, "event:Touch_neg:0")["interval_coverage_slots"],
                                                           rec(recs, "event:Touch_neg:0")["annotated_slots"]),
          (320, [], []))   # midpoint 315 -> 320; frames 170 207 245 282 320 357 395 432; 320 is the excluded end

    # 3. Regression: the nearest anchor after the last grid point is legitimately > 8 frames from the midpoint.
    check("end-of-video event anchor", rec(recs, "event:End_event:0")["anchor"], 96)
    inv = [p for p in check_invariants(videos, recs, uniq) if p.split(":")[0] in STRATA or "identical" in p]
    check("no spurious per-record invariant problems", inv, [])

    # 4. Regression: the invariant rejects a non-nearest anchor and the wrong tie-break (event and normal).
    for rid, bad in (("event:Tie_event:0", 16), ("event:Max_neg:0", 512), ("normal:Normal_tie", 32)):
        mut = [dict(r) for r in recs]
        rec(mut, rid)["anchor"] = bad
        check("invariant flags anchor %d for %s" % (bad, rid),
              any(p.startswith(rid + ":") for p in check_invariants(videos, mut, uniq)), True)

    # 5. Regression: two window_uids sampling identical frames are flagged as one input.
    v2 = [_fixture_video("Ident", "Arson", 200, [(60, 70), (76, 86)])]   # anchors 64 and 80 both give [0, 200)
    r2, _ = build_windows(v2)
    u2, _ = deduplicate(r2)
    check("identical-frame inputs", [r["frame_indices"] == r2[0]["frame_indices"] for r in r2 if r["stratum"] == "event"],
          [True, True])
    check("identical-frame inputs flagged", sum("identical frames" in p for p in check_invariants(v2, r2, u2)), 1)

    # 6. End-to-end main() on a synthetic population that passes the preflight.
    def run(entries, listings):
        with tempfile.TemporaryDirectory() as tmp:
            ucf = os.path.join(tmp, "ucf.json")
            with open(ucf, "w", encoding="utf-8") as fh:
                json.dump(entries, fh)
            paths = []
            for name, lines in listings:
                paths.append(os.path.join(tmp, name))
                with open(paths[-1], "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            out = os.path.join(tmp, "out.json")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main(["--ucf-eval", ucf, "--listings"] + paths + ["--out", out])
            res = None
            if os.path.exists(out):
                with open(out, encoding="utf-8") as fh:
                    res = json.load(fh)
            return rc, res

    base = _synthetic_eval()
    rc, res = run(base, _synthetic_listings(base))
    check("synthetic population: clean run", (rc, res and res["problems"]), (0, []))
    check("synthetic population: counts", res and (res["counts"]["intervals"], res["window_summary"]["window_records"]),
          (162, 452))

    ents = json.loads(json.dumps(base))
    single = next(e for e in ents if e["temporal_label"] == [400, 500, -1, -1])
    single["length"], single["temporal_label"] = 112, [110, 112, -1, -1]
    rc, res = run(ents, _synthetic_listings(ents))
    check("end-of-video event interval: main exit 0", (rc, res and res["problems"]), (0, []))

    lst = _synthetic_listings(base)
    dropped = lst[0][1].pop(5)
    rc, res = run(base, lst)
    check("missing listing entry flagged (output written, exit 2)",
          (rc, res is not None and any("missing from the listings" in p for p in res["problems"])), (2, True))

    lst = _synthetic_listings(base)
    lst.append(("dup.tsv", ["7\t" + dropped.split("\t")[1]]))
    rc, res = run(base, lst)
    check("duplicate listing entry flagged",
          (rc, res is not None and any("listed more than once" in p for p in res["problems"])), (2, True))

    ents = json.loads(json.dumps(base))
    double = next(e for e in ents if e["temporal_label"] == [200, 300, 600, 700])
    double["length"], double["temporal_label"] = 200, [60, 70, 76, 86]
    rc, res = run(ents, _synthetic_listings(ents))
    check("identical-frame inputs flagged by main",
          (rc, res is not None and any("identical frames" in p for p in res["problems"])), (2, True))

    n_fail = 0
    for name, ok, got, want in results:
        if not ok:
            n_fail += 1
            print("FAIL %s: got %r, expected %r" % (name, got, want))
    print("self-test: %d passed, %d failed" % (len(results) - n_fail, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    try:
        sys.exit(main())
    except PreflightError as exc:
        print("FAIL", exc, file=sys.stderr)
        sys.exit(2)

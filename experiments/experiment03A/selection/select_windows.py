#!/usr/bin/env python3
"""Deterministic window selection for experiment03 (protocol draft v0.3). Standard library only; no model, no video.

Inputs (metadata only; sha256 recorded in selection_log.json):
  VERA Data/UCF_Eval.json, read from the local VERA checkout at 008ed401 (= base 15b8bcb8 + experiment01 patch, which
      does not touch Data/; the validator re-checks the hash against `git show 008ed401:Data/UCF_Eval.json` on oz)
  VERA_prediction_scores_UCF_Crime.zip     the authors' released Step-1 labels (per anchor, 0/1) for all 290 test videos
  setup/logs/ucf_part{1,2,3,4}_listing.tsv, ucf_testing_normal_listing.tsv   official zip directories (sizes, members)

Definitions (VERA arithmetic): anchors range(0, n, 16); window [a-150, a+150); 8 frames np.arange(s, e, (e-s)/8).
"Unclipped" means a-150 >= 0 and a+150 <= n. Intervals are the manifest's (start, min(end, n)) pairs that are not -1.

Strata:
  S1 true alerts   anomaly video that has BOTH an S1-eligible and an S2-eligible anchor (joint eligibility).
                   S1-eligible anchor: released label exactly 1, unclipped, inside some annotated interval.
                   Choice: maximise overlap = sum over intervals of |window intersect interval|; ties: smallest distance
                   from the anchor to any interval midpoint; then the smallest anchor.
                   Classes: 10 in hash order of the class key; within a class, the first jointly eligible video in hash order.
  S2 alerts outside the annotation (benchmark-negative alerts), always from the S1 video:
                   released label 1, unclipped, window at least 150 frames from every annotated interval.
                   Preference: side 'pre' (window ends >= 150 frames before the first interval) before any other side;
                   within a side, hash order. Never from another video (the script fails instead).
  S3 normal        normal test video with an unclipped anchor; the unclipped anchor closest to n/2, ties to the earlier.
Ordering keys (ascending sha256 hexdigest of f"{SEED}:{key}"): "class:{class}", "S1video:{video}", "S2:{video}:{anchor}",
"S3video:{video}".
Replacement: rerun with --exclude VIDEO_ID ... . Window IDs are slot-stable: slot k keeps its ID whatever else is excluded.
  S1/S2 slot k (k = 1..10) belongs to the k-th class in hash order. Its candidates are the first 1 + N_BACKUPS jointly
  eligible videos of that class, in hash order; the first one not excluded is used. If none is left, the slot takes the
  first not-yet-used next class (classes 11, 12, ... in hash order) that still has a candidate, from that class's own
  first 1 + N_BACKUPS videos. If no next class has one, the slot is dropped and logged.
  S3 slot k (k = 1..10) is the k-th normal video; an excluded one takes the first unused, non-excluded of the next
  N_BACKUPS normal videos, in order; if none is left, the slot is dropped and logged.
  Every video the rerun can ever pick is pre-listed with its full windows in backup_windows.json.

Usage: python3 -I -B select_windows.py --root ~/Documents/research/vera_research --out <dir> [--exclude ID ...]
"""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

SEED = "experiment03-20261023"
MAX_MB = 100
HOLD_OUT = {"RoadAccidents127_x264", "Normal_Videos_018_x264"}  # experiment02's videos (hypothesis-forming)
N_CLASSES_S1 = 10
N_NORMAL = 10
N_BACKUPS = 5
HALF = 150      # 1/2 * fps(30) * t_window(10), VERA constants
STEP = 16       # VERA sampling_rate
MARGIN = 150    # S2: minimum gap (frames) between the window and every annotated interval
KEY_TEMPLATES = ["class:{class}", "S1video:{video}", "S2:{video}:{anchor}", "S3video:{video}"]


def h(key):
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vid(e):
    return e["video"].split("/")[-1][:-4]


def cls(e):
    return e["video"].split("/")[1]


def intervals(e):
    tl, n = e["temporal_label"], e["length"]
    return [(tl[i], min(tl[i + 1], n)) for i in range(0, len(tl), 2) if tl[i] != -1]


def window_frames(a, n):
    """numpy.arange(s, e, step).astype(int): length ceil((e-s)/step), values s + i*step (checked against
    vera_pipeline.window_for_anchor on oz; see CHECKS.md)."""
    import math
    s, e = max(a - 0.5 * 30 * 10, 0), min(a + 0.5 * 30 * 10, n)
    step = (e - s) / 8
    return s, e, [int(s + i * step) for i in range(math.ceil((e - s) / step))]


def unclipped(x, n):
    return x - HALF >= 0 and x + HALF <= n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="video ids that failed acquisition/decoding checks")
    a = ap.parse_args()
    root, out = Path(a.root).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    excluded_by_user = set(a.exclude)
    man_p = root / "setup/source/VERA/Data/UCF_Eval.json"
    rel_p = root / "shared/reference/vera_official_ucf_initial_scores/VERA_prediction_scores_UCF_Crime.zip"
    lst = [root / f"setup/logs/ucf_part{k}_listing.tsv" for k in (1, 2, 3, 4)] + [root / "setup/logs/ucf_testing_normal_listing.tsv"]
    man = json.loads(man_p.read_text())
    z = zipfile.ZipFile(rel_p)
    rel = {i.filename.split("/")[-1][:-5]: json.loads(z.read(i)) for i in z.infolist()
           if i.filename.endswith(".json") and not i.filename.startswith("__MACOSX")}
    member = {}
    for p in lst:
        for line in p.read_text().splitlines():
            b, m = line.split("\t")
            if m.endswith(".mp4"):
                member[m.split("/")[-1][:-4]] = {"bytes": int(b), "zip": p.name.replace("_listing.tsv", "").replace("ucf_", ""), "member": m}
    log = {"seed": SEED, "key_templates": KEY_TEMPLATES, "order": "ascending sha256 hexdigest of SEED:key",
           "max_mb": MAX_MB, "hold_out": sorted(HOLD_OUT), "excluded_by_replacement": sorted(excluded_by_user),
           "vera_source": {"checkout_head": "008ed4019781c39e7cb16244b7500244f3986f92",
                           "base_commit": "15b8bcb8574a977c229c577f50bfe6f06d07106e",
                           "note": "Data/UCF_Eval.json is not modified by the experiment01 patch"},
           "inputs": {"ucf_eval_json": {"path": str(man_p.relative_to(root)), "sha256": sha_file(man_p)},
                      "released_scores_zip": {"path": str(rel_p.relative_to(root)), "sha256": sha_file(rel_p)},
                      "zip_listings": {p.name: sha_file(p) for p in lst}},
           "excluded": {"hold_out": [], "too_large": [], "no_unclipped_anchor": [], "replacement": []}}

    def usable(e):
        # user exclusions are applied per slot below, so that slots and their candidate lists never shift
        v = vid(e)
        if v in HOLD_OUT:
            log["excluded"]["hold_out"].append(v)
            return False
        if member[v]["bytes"] > MAX_MB * 1e6:
            log["excluded"]["too_large"].append(v)
            return False
        if not any(unclipped(x, e["length"]) for x in range(0, e["length"], STEP)):
            log["excluded"]["no_unclipped_anchor"].append(v)
            return False
        return True

    anom = [e for e in man if intervals(e) and usable(e)]
    norm = [e for e in man if not intervals(e) and usable(e)]

    def s1_anchor(e):
        n, iv, best = e["length"], intervals(e), None
        for k, r in rel[vid(e)].items():
            x = int(k)
            if r["score"] != 1 or not unclipped(x, n) or not any(s <= x < b for s, b in iv):
                continue
            ov = sum(max(0, min(x + HALF, b) - max(x - HALF, s)) for s, b in iv)
            mid = min(abs(x - (s + b) / 2) for s, b in iv)
            key = (-ov, mid, x)
            if best is None or key < best[0]:
                best = (key, x, ov)
        return best[1] if best else None

    def side(x, iv):
        if x + HALF + MARGIN <= min(s for s, _ in iv):
            return "pre"
        if x - HALF - MARGIN >= max(b for _, b in iv):
            return "post"
        return "between"

    def s2_anchors(e):
        n, iv, out_ = e["length"], intervals(e), []
        for k, r in rel[vid(e)].items():
            x = int(k)
            if r["score"] != 1 or not unclipped(x, n):
                continue
            if all(x + HALF + MARGIN <= s or x - HALF - MARGIN >= b for s, b in iv):
                out_.append(x)
        return sorted(out_, key=lambda x: (side(x, iv) != "pre", h(f"S2:{vid(e)}:{x}")))

    def jointly_eligible(e):
        return s1_anchor(e) is not None and bool(s2_anchors(e))

    by_class = {}
    for e in anom:
        if jointly_eligible(e):
            by_class.setdefault(cls(e), []).append(e)
    for c in by_class:
        by_class[c].sort(key=lambda e: h(f"S1video:{vid(e)}"))
    classes = sorted(by_class, key=lambda c: h(f"class:{c}"))
    log["jointly_eligible_s1_videos_by_class"] = {c: len(v) for c, v in sorted(by_class.items())}
    log["class_order"] = classes

    def record(stratum, e, x, why, wid, extra=None):
        n, iv = e["length"], intervals(e)
        s, en, fr = window_frames(x, n)
        r = rel[vid(e)][str(x)]["score"]
        ov = sum(max(0, min(en, b) - max(s, a_)) for a_, b in iv)
        gap = min((a_ - en if a_ >= en else s - b) for a_, b in iv) if iv and ov == 0 else (0 if iv else None)
        ann_slots = [i for i, f in enumerate(fr) if any(a_ <= f < b for a_, b in iv)]
        localized = any(s <= a_ and b <= en and a_ <= x < b for a_, b in iv) and 2 <= len(ann_slots) <= 4
        labs = [rel[vid(e)][str(y)]["score"] for y in range(0, n, STEP)]
        ai = list(range(0, n, STEP)).index(x)
        run = 1
        for d in (-1, 1):
            j = ai + d
            while 0 <= j < len(labs) and labs[j] == labs[ai]:
                run += 1
                j += d
        return {"window_id": wid, "stratum": stratum, "video_id": vid(e), "class": cls(e), "manifest_entry": e["video"],
                "manifest_length": n, "anchor": x, "window": [s, en], "window_s_at_30fps": [round(s / 30, 2), round(en / 30, 2)],
                "frame_indices": fr, "released_label": r, "annotation": e["temporal_label"], "overlap_frames": ov,
                "gap_to_annotation_frames": gap, "annotated_slots": ann_slots, "localized_event": localized,
                "side": side(x, iv) if (iv and ov == 0) else None,
                "released_run_length_anchors": run, "released_label1_fraction_video": round(sum(labs) / len(labs), 3),
                "selection_rule": why, "zip": member[vid(e)]["zip"], "zip_member": member[vid(e)]["member"],
                "video_bytes": member[vid(e)]["bytes"], **(extra or {})}

    windows, backups = [], []
    s1_rule = "max overlap with annotation; ties: distance to nearest interval midpoint, then earliest"
    s2_rule = "same video as S1; released label 1; >=150 frames from every interval; 'pre' side first, then hash order"
    used_classes = classes[:N_CLASSES_S1]          # slot k belongs to the k-th class, whatever is excluded
    next_classes = classes[N_CLASSES_S1:]
    cand = {c: by_class[c][:1 + N_BACKUPS] for c in classes}   # every S1 video the selection can ever use, per class
    nv = sorted(norm, key=lambda e: h(f"S3video:{vid(e)}"))
    s3_backup_pool = nv[N_NORMAL:N_NORMAL + N_BACKUPS]
    known = {vid(e) for c in classes for e in cand[c]} | {vid(e) for e in nv[:N_NORMAL + N_BACKUPS]}
    unknown = sorted(excluded_by_user - known)
    assert not unknown, f"--exclude ids that no slot could ever use: {unknown}"
    log["excluded"]["replacement"] = sorted(excluded_by_user)
    log["slot_classes"] = used_classes
    log["next_class_order"] = next_classes
    log["next_class_candidates"] = {c: len(cand[c]) for c in next_classes}
    replacements, dropped, used_next = [], [], []
    for k, c in enumerate(used_classes, start=1):
        pick_c = c
        pick = next((e for e in cand[c] if vid(e) not in excluded_by_user), None)
        if pick is None:
            for c2 in next_classes:
                p2 = next((e for e in cand[c2] if vid(e) not in excluded_by_user), None)
                if c2 not in used_next and p2 is not None:
                    used_next.append(c2)
                    pick, pick_c = p2, c2
                    break
        if pick is None:
            dropped.append(f"S1-{k:02d}/S2-{k:02d}")
            continue
        s2 = s2_anchors(pick)
        assert s2, f"no same-video S2 anchor for {vid(pick)}"
        extra = {}
        if pick is not cand[c][0]:
            i = cand[pick_c].index(pick)
            src = f"{k:02d}b{i}" if pick_c == c else f"next{next_classes.index(pick_c) + 1}b{i + 1}"
            extra = {"replaces_video": vid(cand[c][0]), "backup_record": [f"S1-{src}", f"S2-{src}"]}
            replacements.append({"slot": f"S1-{k:02d}/S2-{k:02d}", "excluded": vid(cand[c][0]), "class": pick_c,
                                 "video_id": vid(pick), "backup_record": extra["backup_record"]})
        windows.append(record("S1", pick, s1_anchor(pick), s1_rule, f"S1-{k:02d}", extra or None))
        windows.append(record("S2", pick, s2[0], s2_rule, f"S2-{k:02d}", {"paired_with": f"S1-{k:02d}", **extra}))
    for k, c in enumerate(used_classes, start=1):
        for i, b in enumerate(cand[c][1:], start=1):
            backups.append({"replaces": f"S1-{k:02d}/S2-{k:02d}", "class": c, "order": i,
                            "S1": record("S1", b, s1_anchor(b), s1_rule, f"S1-{k:02d}b{i}"),
                            "S2": record("S2", b, s2_anchors(b)[0], s2_rule, f"S2-{k:02d}b{i}", {"paired_with": f"S1-{k:02d}b{i}"})})
    for j, c in enumerate(next_classes, start=1):
        for i, b in enumerate(cand[c], start=1):
            backups.append({"replaces": "the first S1/S2 slot (in slot order) whose class has no candidate left",
                            "class": c, "next_class_rank": j, "order": i,
                            "S1": record("S1", b, s1_anchor(b), s1_rule, f"S1-next{j}b{i}"),
                            "S2": record("S2", b, s2_anchors(b)[0], s2_rule, f"S2-next{j}b{i}", {"paired_with": f"S1-next{j}b{i}"})})

    def s3_anchor(e):
        n = e["length"]
        return min((y for y in range(0, n, STEP) if unclipped(y, n)), key=lambda y: (abs(y - n / 2), y))
    s3_rule = "unclipped anchor closest to n/2 (ties earlier); label-agnostic"
    used_s3 = set()
    for k, e in enumerate(nv[:N_NORMAL], start=1):
        pick, extra = e, None
        if vid(e) in excluded_by_user:
            pick = next((b for b in s3_backup_pool if vid(b) not in excluded_by_user and vid(b) not in used_s3), None)
            if pick is None:
                dropped.append(f"S3-{k:02d}")
                continue
            used_s3.add(vid(pick))
            extra = {"replaces_video": vid(e), "backup_record": [f"S3-b{s3_backup_pool.index(pick) + 1}"]}
            replacements.append({"slot": f"S3-{k:02d}", "excluded": vid(e), "class": "Normal", "video_id": vid(pick),
                                 "backup_record": extra["backup_record"]})
        windows.append(record("S3", pick, s3_anchor(pick), s3_rule, f"S3-{k:02d}", extra))
    for i, b in enumerate(s3_backup_pool, start=1):
        backups.append({"replaces": "the first excluded S3 slot (in slot order) not yet replaced", "class": "Normal",
                        "order": i, "S3": record("S3", b, s3_anchor(b), s3_rule, f"S3-b{i}")})
    log["replacements"] = replacements
    log["dropped_slots"] = dropped

    counts = {s: sum(w["stratum"] == s for w in windows) for s in ("S1", "S2", "S3")}
    n_pairs = 10 - sum(d.startswith("S1-") for d in dropped)
    assert counts == {"S1": n_pairs, "S2": n_pairs, "S3": 10 - sum(d.startswith("S3-") for d in dropped)}, counts
    for s1 in [w for w in windows if w["stratum"] == "S1"]:
        s2 = next(w for w in windows if w["window_id"] == "S2-" + s1["window_id"][3:])
        assert s2["paired_with"] == s1["window_id"] and s2["video_id"] == s1["video_id"]
    for w in windows + [r for b in backups for r in b.values() if isinstance(r, dict)]:
        assert len(w["frame_indices"]) == 8 and w["window"][0] >= 0 and w["window"][1] <= w["manifest_length"]
    log["counts"] = counts
    log["distinct_videos"] = len({w["video_id"] for w in windows})
    log["download_mb_primary"] = round(sum({w["video_id"]: w["video_bytes"] for w in windows}.values()) / 1e6, 1)
    log["backup_counts_by_class"] = {c: sum(1 for b in backups if b.get("class") == c and isinstance(b["order"], int))
                                     for c in used_classes}
    for k in log["excluded"]:
        log["excluded"][k] = sorted(set(log["excluded"][k]))
    (out / "windows.json").write_text(json.dumps({"seed": SEED, "windows": windows}, indent=1) + "\n")
    (out / "backup_windows.json").write_text(json.dumps({"seed": SEED, "backups": backups}, indent=1) + "\n")
    (out / "selection_log.json").write_text(json.dumps(log, indent=1) + "\n")
    with open(out / "windows.csv", "w", newline="") as f:
        cols = ["window_id", "stratum", "video_id", "class", "anchor", "window", "window_s_at_30fps", "released_label",
                "overlap_frames", "gap_to_annotation_frames", "side", "annotated_slots", "localized_event",
                "released_run_length_anchors", "released_label1_fraction_video", "annotation", "manifest_length",
                "video_bytes", "paired_with"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for w in windows:
            wr.writerow({k: (json.dumps(w.get(k)) if isinstance(w.get(k), list) else w.get(k)) for k in cols})
    print(json.dumps({k: log[k] for k in ("counts", "distinct_videos", "download_mb_primary", "backup_counts_by_class")}, indent=1))
    print("excluded:", {k: len(v) for k, v in log["excluded"].items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the experiment03B window manifest from metadata only (PROTOCOL.md §3). No model, no video.

Inputs: VERA Data/UCF_Eval.json at 008ed401 (sha256 checked) and the official zip listings in setup/logs/
(member path and byte size per video; the UCF part 1, 2 and 4 listings were made for experiment03A's selection).
Frame indices come from setup/scripts/vera_pipeline.window_for_anchor, the expression experiment01 ran.

    setup/environment/env/bin/python experiments/experiment03B/code/manifest3b.py --out experiments/experiment03B/manifest
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", CODE.parents[2])).resolve()
sys.path.insert(0, str(ROOT / "setup" / "scripts"))
sys.path.insert(0, str(CODE))
import vera_pipeline as vp  # noqa: E402
import xb_core as xc  # noqa: E402

MANIFEST_VERSION = "exp03b-manifest-1"
UCF_EVAL_SHA256 = "f3857cddd623f767e7b86bd50a4aeb122445fc6e241ddd7130b7e3ec0c4d8b9f"
EXPECTED = {"videos": 290, "anomalous": 140, "normal": 150, "anomaly_categories": 13, "intervals": 162}
STEP, HALF = 16, 150
LISTINGS = ["ucf_part1_listing.tsv", "ucf_part2_listing.tsv", "ucf_part3_listing.tsv", "ucf_part4_listing.tsv",
            "ucf_testing_normal_listing.tsv"]
ZIP_NAMES = {"ucf_part1_listing.tsv": "Anomaly-Videos-Part-1.zip", "ucf_part2_listing.tsv": "Anomaly-Videos-Part-2.zip",
             "ucf_part3_listing.tsv": "Anomaly-Videos-Part-3.zip", "ucf_part4_listing.tsv": "Anomaly-Videos-Part-4.zip",
             "ucf_testing_normal_listing.tsv": "Testing_Normal_Videos.zip"}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def video_id(e):
    return e["video"].split("/")[-1][:-4]


def category(e):
    return e["video"].split("/")[1]


def intervals(e):
    tl, n = e["temporal_label"], e["length"]
    return [(tl[i], min(tl[i + 1], n)) for i in range(0, len(tl), 2) if tl[i] != -1]


def window(a, n):
    s, e, fr = vp.window_for_anchor(a, n)
    return int(s), int(e), [int(f) for f in fr]


def annotated_slots(frames, ivs):
    return [i for i, f in enumerate(frames) if any(s <= f < e for s, e in ivs)]


def nearest_anchor(target, n):
    return min(range(0, n, STEP), key=lambda a: (abs(a - target), a))


def negative_anchor(n, ivs):
    best = None
    for a in range(0, n, STEP):
        if a - HALF < 0 or a + HALF > n:
            continue
        ws, we = a - HALF, a + HALF
        if any(ws < e and s < we for s, e in ivs):
            continue
        gap = min((s - we) if we <= s else (ws - e) for s, e in ivs)
        if best is None or gap > best[0]:
            best = (gap, a)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ucf-eval", default=str(ROOT / "setup/source/VERA/Data/UCF_Eval.json"))
    ap.add_argument("--listings-dir", default=str(ROOT / "setup/logs"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_sha = sha_file(a.ucf_eval)
    if ev_sha != UCF_EVAL_SHA256:
        sys.exit(f"UCF_Eval.json sha256 {ev_sha} != pinned {UCF_EVAL_SHA256}")
    man = json.loads(Path(a.ucf_eval).read_text(encoding="utf-8"))
    member = {}
    listing_sha = {}
    for name in LISTINGS:
        p = Path(a.listings_dir) / name
        listing_sha[name] = sha_file(p)
        for line in p.read_text(encoding="utf-8").splitlines():
            b, m = line.split("\t")
            if m.endswith(".mp4"):
                member[m.split("/")[-1][:-4]] = {"zip": ZIP_NAMES[name], "zip_member": m, "video_bytes": int(b)}

    anom = [e for e in man if intervals(e)]
    norm = [e for e in man if not intervals(e)]
    counts = {"videos": len(man), "anomalous": len(anom), "normal": len(norm),
              "anomaly_categories": len({category(e) for e in anom}),
              "intervals": sum(len(intervals(e)) for e in man),
              "source_frames": sum(e["length"] for e in man),
              "stride16_anchors": sum(len(range(0, e["length"], STEP)) for e in man),
              "interval_ends_clipped_to_length": sum(1 for e in man for i in range(0, 4, 2)
                                                     if e["temporal_label"][i] != -1 and e["temporal_label"][i + 1] > e["length"]),
              "videos_by_category": {}}
    for e in man:
        counts["videos_by_category"][category(e)] = counts["videos_by_category"].get(category(e), 0) + 1
    bad = {k: (counts[k], v) for k, v in EXPECTED.items() if counts[k] != v}
    if bad:
        sys.exit(f"preflight recount differs from the expected manifest: {bad}")
    missing_members = sorted(video_id(e) for e in man if video_id(e) not in member)
    if missing_members:
        sys.exit(f"videos missing from the zip listings: {missing_members}")

    videos, windows = [], []
    for e in sorted(man, key=video_id):
        vid, n, ivs = video_id(e), e["length"], intervals(e)
        dev = vid in xc.DEVELOPMENT_VIDEOS
        videos.append({"video_id": vid, "category": category(e), "manifest_entry": e["video"], "length": n,
                       "intervals": [list(iv) for iv in ivs], "development": dev, **member[vid], "status": "selected"})

        def rec(stratum, idx, anchor, extra):
            s, en, fr = window(anchor, n)
            if len(fr) != 8:
                sys.exit(f"{vid}@{anchor}: VERA arithmetic gave {len(fr)} frames")
            return {"window_record_id": f"{stratum}:{vid}:{idx}", "stratum": stratum, "video_id": vid,
                    "category": category(e), "anchor": anchor, "window": [s, en], "frame_indices": fr,
                    "clipped": s > anchor - HALF or en < anchor + HALF, "annotated_slots": annotated_slots(fr, ivs),
                    "window_uid": f"{vid}@{anchor}", "development": dev, "status": "selected", **extra}
        if ivs:
            for k, (s, en) in enumerate(ivs):
                anc = nearest_anchor((s + en) / 2, n)
                r = rec("event", k, anc, {"interval": [s, en], "interval_index": k, "interval_midpoint": (s + en) / 2,
                                          "gap_to_annotation": None})
                r["interval_coverage_slots"] = [i for i, f in enumerate(r["frame_indices"]) if s <= f < en]
                windows.append(r)
            neg = negative_anchor(n, ivs)
            if neg is None:
                windows.append({"window_record_id": f"negative:{vid}:0", "stratum": "negative", "video_id": vid,
                                "category": category(e), "anchor": None, "window": None, "frame_indices": None,
                                "clipped": None, "annotated_slots": None, "window_uid": None, "development": dev,
                                "status": "unavailable", "interval": None, "interval_index": None,
                                "gap_to_annotation": None,
                                "reason": "no full 10-second window that avoids every annotated interval"})
            else:
                windows.append(rec("negative", 0, neg[1], {"interval": None, "interval_index": None,
                                                           "gap_to_annotation": neg[0]}))
        else:
            windows.append(rec("normal", 0, nearest_anchor(n / 2, n), {"interval": None, "interval_index": None,
                                                                        "gap_to_annotation": None}))

    uniq = {}
    for w in windows:
        if w["window_uid"] is None:
            continue
        u = uniq.setdefault(w["window_uid"], {"window_uid": w["window_uid"], "video_id": w["video_id"],
                                               "category": w["category"], "anchor": w["anchor"],
                                               "frame_indices": w["frame_indices"], "window": w["window"],
                                               "clipped": w["clipped"], "annotated_slots": w["annotated_slots"],
                                               "development": w["development"], "window_record_ids": [], "strata": []})
        u["window_record_ids"].append(w["window_record_id"])
        if w["stratum"] not in u["strata"]:
            u["strata"].append(w["stratum"])
    seen_per_stratum = {}
    for w in windows:
        if w["window_uid"] is None:
            continue
        key = (w["stratum"], w["window_uid"])
        if key in seen_per_stratum:
            w["status"] = "duplicate_input"
            w["duplicate_of"] = seen_per_stratum[key]
        else:
            seen_per_stratum[key] = w["window_record_id"]
    by = lambda st: [w for w in windows if w["stratum"] == st]  # noqa: E731
    counts["window_records"] = len(windows)
    counts["window_records_by_stratum"] = {st: len(by(st)) for st in ("event", "negative", "normal")}
    counts["unavailable_negative"] = sum(w["status"] == "unavailable" for w in windows)
    counts["duplicate_input_records"] = sum(w["status"] == "duplicate_input" for w in windows)
    counts["unique_inputs"] = len(uniq)
    counts["unique_inputs_by_stratum"] = {st: len({w["window_uid"] for w in by(st) if w["window_uid"]})
                                          for st in ("event", "negative", "normal")}
    counts["unique_inputs_non_development"] = sum(not u["development"] for u in uniq.values())
    counts["clipped_unique_inputs"] = sum(bool(u["clipped"]) for u in uniq.values())
    counts["event_records_with_no_annotated_slot"] = sum(1 for w in by("event") if not w["interval_coverage_slots"])
    counts["download_bytes_all_videos"] = sum(v["video_bytes"] for v in videos)
    counts["download_bytes_non_development"] = sum(v["video_bytes"] for v in videos if not v["development"])
    counts["largest_video"] = max(((v["video_bytes"], v["video_id"]) for v in videos))
    counts["per_input_upper_bound"] = xc.upper_bound_counts()
    counts["frames_needed"] = sum(len(set(f for u in uniq.values() if u["video_id"] == v["video_id"]
                                          for f in u["frame_indices"])) for v in videos)
    mani = {"version": MANIFEST_VERSION, "protocol_version": "1.0",
            "inputs_sha256": {"ucf_eval_json": ev_sha, "listings": listing_sha,
                              "vera_pipeline.py": sha_file(ROOT / "setup/scripts/vera_pipeline.py"),
                              "manifest3b.py": sha_file(Path(__file__))},
            "rules": {"anchors": "range(0, length, 16)", "window": "vera_pipeline.window_for_anchor (30 fps, 10 s, 8 frames)",
                      "intervals": "temporal_label pairs with start != -1, end clipped to length, half-open [start, end)",
                      "event": "anchor nearest (start + end) / 2, earlier on ties",
                      "negative": "full window [a-150, a+150) inside [0, length) that overlaps no interval; max gap to the "
                                  "nearest interval, earlier on ties; else unavailable",
                      "normal": "anchor nearest length / 2, earlier on ties",
                      "dedup": "an input is (video_id, anchor); records sharing it are linked; each stratum counts it once"},
            "counts": counts, "development_videos": list(xc.DEVELOPMENT_VIDEOS), "videos": videos, "windows": windows,
            "unique_inputs": sorted(uniq.values(), key=lambda u: u["window_uid"])}
    text = json.dumps(mani, indent=1) + "\n"
    (out / "manifest_v1.json").write_text(text, encoding="utf-8")
    (out / "manifest_v1.json.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  manifest_v1.json\n")
    with open(out / "windows_v1.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["window_record_id", "stratum", "status", "video_id", "category", "development", "anchor", "window",
                "clipped", "frame_indices", "interval", "interval_coverage_slots", "annotated_slots", "gap_to_annotation",
                "window_uid", "duplicate_of"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for w in windows:
            wr.writerow({k: (json.dumps(w.get(k)) if isinstance(w.get(k), list) else w.get(k)) for k in cols})
    print(json.dumps({k: v for k, v in counts.items() if k != "videos_by_category"}, indent=1))
    print("sha256", hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()

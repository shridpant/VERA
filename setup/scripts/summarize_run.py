#!/usr/bin/env python3
"""Collect the numbers a run report needs into <run-dir>/results/summary.json (no model calls).

Example windows are chosen by a fixed rule, never by reading the text first:
five evenly spaced windows (positions round(q*(K-1)) for q in 0, .25, .5, .75, 1), plus the window with the
highest temporally refined score (earliest on ties) and the first window whose raw label is 1 (if any).
Usage: python setup/scripts/summarize_run.py --run-dir experiments/experiment01/runs/run_001
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)).resolve()


def runs_of_ones(anchors, labels):
    out, cur = [], None
    for a, l in zip(anchors, labels):
        if l == 1:
            cur = [a, a] if cur is None else [cur[0], a]
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    rd = Path(ap.parse_args().run_dir)
    rd = rd if rd.is_absolute() else ROOT / rd
    res = rd / "results"
    recs = [json.loads(l) for l in (res / "raw_responses.jsonl").read_text(encoding="utf-8").splitlines()]
    scene = json.loads((res / "scene_refined_scores.json").read_text(encoding="utf-8"))
    temporal = json.loads((res / "temporal_refined_scores.json").read_text(encoding="utf-8"))
    diag = json.loads((res / "diagnostics.json").read_text(encoding="utf-8"))
    ident = json.loads((res / "frames_identity.json").read_text(encoding="utf-8"))
    man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
    fs = np.load(res / "frame_scores.npy")
    fps = ident["video_metadata_cv2"]["container_fps"]
    anchors = [r["anchor_frame"] for r in recs]
    labels = [r["label"] for r in recs]
    tm = np.array(temporal["temporal_smoothed"])
    sc = np.array(scene["scene_refined"])
    K = len(recs)

    pick = {}
    for q in (0, 0.25, 0.5, 0.75, 1):
        pick.setdefault(int(round(q * (K - 1))), f"evenly spaced q={q}")
    top = int(np.argmax(tm))
    pick.setdefault(top, "highest temporally refined score")
    first_one = next((i for i, l in enumerate(labels) if l == 1), None)
    if first_one is not None:
        pick.setdefault(first_one, "first raw label 1")
    examples = []
    for i in sorted(pick):
        r = recs[i]
        examples.append({"rule": pick[i], "window_index": i, "anchor_frame": r["anchor_frame"],
                         "anchor_time_s": round(r["anchor_time_s"], 2),
                         "window_frames": [r["window_start"], r["window_end"]],
                         "window_time_s": [round(r["window_start_time_s"], 2), round(r["window_end_time_s"], 2)],
                         "raw_label": r["label"], "scene_refined": round(float(sc[i]), 4),
                         "temporal_refined": round(float(tm[i]), 4), "raw_response": r["raw_response"]})

    rt = [r["runtime_s"] for r in recs]
    tok = [r["n_new_tokens"] for r in recs]
    vlm_stage = [s for s in man["stages"]["vlm"] if s["status"] == "ok"][-1]
    status_counts = {}
    for r in recs:
        status_counts[r["parse_status"]] = status_counts.get(r["parse_status"], 0) + 1
    gt = json.loads((res / "benchmark_annotation.json").read_text(encoding="utf-8"))
    hi = np.where(fs >= 0.5 * fs.max())[0] if fs.max() > 0 else np.array([], int)
    summary = {
        "video": recs[0]["video_id"], "decoded_frames": ident["decoded_frame_count"], "fps": fps,
        "duration_s": ident["decoded_frame_count"] / fps, "video_metadata": ident["video_metadata_cv2"],
        "windows": K, "parse_status_counts": status_counts,
        "raw_label_1_count": sum(l == 1 for l in labels),
        "raw_label_1_runs_anchor_frames": runs_of_ones(anchors, labels),
        "upstream_root_parser_disagreements": sum(r["label"] != r["label_upstream_root_parser"] for r in recs),
        "upstream_inference_variant_parser_disagreements": sum(r["label"] != r["label_upstream_inference_variant_parser"] for r in recs),
        "responses_mentioning_sound": sum(r["mentions_sound"] for r in recs),
        "runtime_s": {"total_generation": round(sum(rt), 1), "mean": round(statistics.mean(rt), 2),
                      "median": round(statistics.median(rt), 2), "max": round(max(rt), 2)},
        "new_tokens": {"mean": round(statistics.mean(tok), 1), "max": max(tok), "hit_limit": sum(t >= 1024 for t in tok)},
        "model_load_s": vlm_stage.get("model_load_s"), "attention": vlm_stage.get("attention"),
        "peak_gpu_mem_allocated_gib": round(max(r["peak_gpu_mem_allocated_bytes"] for r in recs) / 2**30, 2),
        "scene_refined_range": [round(float(sc.min()), 4), round(float(sc.max()), 4)],
        "temporal_refined_range": [round(float(tm.min()), 4), round(float(tm.max()), 4)],
        "temporal_refined_argmax_anchor": anchors[top],
        "frame_score_range": [round(float(fs.min()), 4), round(float(fs.max()), 4)],
        "frame_score_argmax": int(np.argmax(fs)),
        "frames_at_or_above_half_max": [int(hi.min()), int(hi.max())] if len(hi) else None,
        "benchmark_annotation": gt["temporal_label"], "diagnostics": diag,
        "example_selection_rule": __doc__.split("\n\n")[1].strip(),
        "examples": examples,
    }
    with open(res / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Config-driven one-video VERA runner: frames -> VLM decisions (Step 1) -> scene/temporal refinement (Steps 2-3).

Usage (on oz, through setup/scripts/launch.sh for the GPU stage):
  python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml --new-run
  python setup/scripts/run_experiment.py --config ... --run-dir experiments/experiment01/runs/run_001 --stages refine

Stages: frames (shared frame cache), vlm (InternVL2 segment decisions + raw text), refine (compute_auc.py Steps 2-3).
Each run directory freezes its resolved config; a later invocation on the same run directory must use an identical
config. The vlm stage appends one JSON line per window as it finishes and skips windows already recorded, so a
crash or a refine-only rerun never repeats finished VLM work. Nothing is written into the source checkout.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import yaml  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import vera_pipeline as vp  # noqa: E402

ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", SCRIPTS.parent.parent)).resolve()
STAGES = ("frames", "vlm", "refine")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- run directory and manifest

def resolve_run_dir(cfg, args):
    runs = ROOT / cfg["output"]["runs_dir"]
    if args.run_dir:
        rd = (ROOT / args.run_dir).resolve() if not os.path.isabs(args.run_dir) else Path(args.run_dir)
        if not rd.exists() and not args.new_run:
            sys.exit(f"{rd} does not exist (pass --new-run to create it)")
        return rd
    if not args.new_run:
        sys.exit("pass --new-run (next run_NNN) or --run-dir")
    runs.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for d in runs.iterdir() if (m := re.fullmatch(r"run_(\d{3})", d.name))]
    return runs / f"run_{(max(taken) + 1 if taken else 1):03d}"


def freeze_config(cfg_path, run_dir):
    raw = Path(cfg_path).read_text(encoding="utf-8")
    frozen = run_dir / "resolved_config.yaml"
    if frozen.exists():
        if frozen.read_text(encoding="utf-8") != raw:
            sys.exit(f"{cfg_path} differs from the config frozen in {frozen}; start a new run instead")
    else:
        frozen.write_text(raw, encoding="utf-8")
    return hashlib.sha256(raw.encode()).hexdigest()


def load_manifest(run_dir):
    p = run_dir / "run_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def environment_report(run_dir):
    py = sys.executable
    lines = [f"recorded_at: {now()}", f"host: {socket.getfqdn()}", f"platform: {platform.platform()}",
             f"python: {sys.version.split()[0]} ({py})", f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}",
             f"HF_HOME: {os.environ.get('HF_HOME')}", f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}"]
    try:
        import torch
        lines += [f"torch: {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()}",
                  f"cuda_available: {torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                lines.append(f"device {i}: {p.name}, {p.total_memory // 2**20} MiB, capability {p.major}.{p.minor}")
    except Exception as e:  # environment.txt must still be written
        lines.append(f"torch import failed: {e!r}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used",
                          "--format=csv"], capture_output=True, text=True)
    lines += ["", "nvidia-smi:", smi.stdout.strip(), "", "pip freeze:"]
    lines += subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.strip().splitlines()
    with open(run_dir / "environment.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


# ---------------------------------------------------------------- stage: frames

def stage_frames(cfg, run_dir, man):
    video = ROOT / cfg["video"]["path"]
    vsha = sha256_file(video)
    if vsha != cfg["video"]["sha256"]:
        sys.exit(f"video checksum {vsha} != config {cfg['video']['sha256']}")
    src = ROOT / cfg["source"]["checkout"]
    extractor = src / "Preprocessing" / "extract_frames.py"
    import cv2
    cache_root = ROOT / cfg["frames"]["cache_root"]
    stem = video.stem
    frames_dir = cache_root / stem
    ident_path = cache_root / f"{stem}.frames.json"
    shalist_path = cache_root / f"{stem}.frames.sha256"
    want = {"source_sha256": vsha, "extractor_sha256": sha256_file(extractor), "cv2_version": cv2.__version__}

    def frame_digest():
        names = sorted(os.listdir(frames_dir))
        lines = [f"{sha256_file(frames_dir / n)}  {n}" for n in names]
        return names, "\n".join(lines) + "\n"

    reused = False
    if ident_path.exists() and frames_dir.is_dir():
        ident = json.loads(ident_path.read_text(encoding="utf-8"))
        if all(ident.get(k) == v for k, v in want.items()):
            names, digest_text = frame_digest()
            if hashlib.sha256(digest_text.encode()).hexdigest() == ident["frames_digest_sha256"]:
                reused = True
    if not reused:
        sys.path.insert(0, str(extractor.parent))
        import extract_frames as vera_extract  # VERA's own extractor (cv2 decode, {:06d}.jpg, zero-based)
        tmp_parent = cache_root / f".tmp_{stem}_{os.getpid()}"
        shutil.rmtree(tmp_parent, ignore_errors=True)
        tmp_parent.mkdir(parents=True)
        t0 = time.time()
        _, decoded = vera_extract.extract_frames(str(video), str(tmp_parent))
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        os.replace(tmp_parent / stem, frames_dir)
        tmp_parent.rmdir()
        names, digest_text = frame_digest()
        if len(names) != decoded or names != [f"{i:06d}.jpg" for i in range(decoded)]:
            sys.exit(f"frame files do not match the decoded count {decoded}")
        cap = cv2.VideoCapture(str(video))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        meta = {"container_fps": cap.get(cv2.CAP_PROP_FPS), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
                "container_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        shalist_path.write_text(digest_text, encoding="utf-8")
        ident = {**want, "video": rel(video), "frames_dir": rel(frames_dir), "decoded_frame_count": decoded,
                 "naming": "{:06d}.jpg, zero-based, written by cv2.imwrite (default JPEG quality 95)",
                 "extractor": f"{rel(extractor)} extract_frames() at VERA {git('rev-parse', 'HEAD', cwd=src)}",
                 "video_metadata_cv2": meta, "duration_s_from_decoded": decoded / meta["container_fps"],
                 "frames_digest_sha256": hashlib.sha256(digest_text.encode()).hexdigest(),
                 "per_frame_sha256_file": rel(shalist_path), "extracted_at": now(),
                 "extract_seconds": round(time.time() - t0, 2)}
        write_json(ident_path, ident)
    n = ident["decoded_frame_count"]
    expected = cfg["video"]["expected_length"]
    if n != expected:
        sys.exit(f"decoded {n} frames but the VERA manifest says {expected}; investigate before inference")
    write_json(run_dir / "results" / "frames_identity.json", {**ident, "reused_cache": reused})
    return {"decoded_frame_count": n, "reused_cache": reused, "frames_identity": rel(ident_path),
            "video_sha256": vsha}


# ---------------------------------------------------------------- stage: vlm (VERA Step 1)

def one_video_manifest(cfg, run_dir):
    src = ROOT / cfg["source"]["checkout"]
    full = json.loads((src / "Data" / "UCF_Eval.json").read_text(encoding="utf-8"))
    hits = [e for e in full if e["video"] == cfg["video"]["manifest_entry"]]
    if len(hits) != 1:
        sys.exit(f"expected exactly one UCF_Eval.json entry for {cfg['video']['manifest_entry']}, found {len(hits)}")
    entry = hits[0]
    # Step 1 sees only the path and length; benchmark labels are kept apart for diagnostics.
    write_json(run_dir / "results" / "eval_manifest_one_video.json", [{"video": entry["video"], "length": entry["length"]}])
    write_json(run_dir / "results" / "benchmark_annotation.json", {
        "note": "UCF-Crime benchmark annotation copied from VERA Data/UCF_Eval.json. NOT used by Step 1 or the "
                "refinement; diagnostics only. compute_auc.py treats it as frames [start, end) (zero-based).",
        "video": entry["video"], "length": entry["length"], "temporal_label": entry["temporal_label"],
        "video_level_answer": entry["QA"][0]["a"]})
    return run_dir / "results" / "eval_manifest_one_video.json", entry


def stage_vlm(cfg, run_dir, man):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer

    src = ROOT / cfg["source"]["checkout"]
    sys.path.insert(0, str(src))
    import generate_initial_score as gis  # pinned VERA + vera-research patch (importable, no side effects)

    questions = "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"])
    if questions != gis.new_qs:
        sys.exit("config guiding_questions differ from the pinned source's new_qs")
    model_instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
    gen_cfg = dict(cfg["generation"])

    ann_path, entry = one_video_manifest(cfg, run_dir)
    frames_ident = json.loads((run_dir / "results" / "frames_identity.json").read_text(encoding="utf-8"))
    n_decoded = frames_ident["decoded_frame_count"]
    actual_fps = frames_ident["video_metadata_cv2"]["container_fps"]
    test_dataset = gis.Video_Instruct_Dataset_Inference(vis_root=str(ROOT / cfg["frames"]["cache_root"]) + "/",
                                                        ann_root=str(ann_path))
    if len(test_dataset) != 1:
        sys.exit("one-video manifest must have exactly one entry")
    video_path, video_name, vlen = test_dataset[0]
    if video_name != cfg["video"]["id"] or vlen != n_decoded:
        sys.exit(f"dataset returned {video_name}/{vlen}, expected {cfg['video']['id']}/{n_decoded}")

    s = cfg["sampling"]
    fps, t_window, sampling_rate, snippet_len = s["fps_for_windows"], s["t_window_s"], s["sampling_rate"], s["snippet_len"]
    anchor_list = vp.anchors(vlen, sampling_rate)

    out_path = run_dir / "results" / "raw_responses.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["anchor_frame"]] = rec
    todo = [a for a in anchor_list if a not in done]
    if cfg.get("_max_windows") is not None:  # setup smoke tests only; leaves Step 1 incomplete on purpose
        todo = todo[:cfg["_max_windows"]]
    print(f"{len(anchor_list)} anchors, {len(done)} already recorded, {len(todo)} to run", flush=True)

    load_t0 = time.time()
    model_dir = ROOT / cfg["model"]["local_dir"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = AutoModel.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      trust_remote_code=True, use_flash_attn=cfg["model"]["use_flash_attn"]).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    vera = gis.VideoAnomalyDetectionModelInference(model=model, tokenizer=tokenizer, optimizer_instruct=None,
                                                   model_instruct=model_instruct, new_qs=gis.new_qs,
                                                   generation_config=gen_cfg).eval().cuda()
    load_s = time.time() - load_t0
    flash = {"vision_use_flash_attn": bool(model.vision_model.encoder.layers[0].attn.use_flash_attn),
             "llm_attn_implementation": getattr(model.language_model.config, "attn_implementation", None),
             "llm_attention_class": type(model.language_model.model.layers[0].attention).__name__}
    print(f"model loaded in {load_s:.1f}s; {flash}", flush=True)

    captured = {}
    orig_generate = model.generate

    def generate_capture(*a, **kw):  # instrumentation only: record the exact token ids fed to the model
        captured["input_ids"] = kw["input_ids"].detach().cpu()
        out = orig_generate(*a, **kw)
        captured["n_new_tokens"] = int(out.shape[1])
        return out
    model.generate = generate_capture

    image_transform = test_dataset.transform
    vsha = frames_ident["source_sha256"]
    template_path = run_dir / "results" / "model_input_rendered.txt"
    with open(out_path, "a", encoding="utf-8") as fout:
        for ind in todo:
            k = anchor_list.index(ind)
            snippet_start, snippet_end, snippet_indices = vp.window_for_anchor(ind, vlen, fps, t_window, snippet_len)
            frame_no_list = ['{:06d}'.format(i) + '.jpg' for i in snippet_indices]
            pixel_values_list, num_patches_list = [], []
            for frame_name in frame_no_list:
                frame_path = video_path + '/' + frame_name
                frame = Image.open(frame_path).convert('RGB')
                stitched_image = gis.dynamic_preprocess([frame], grid_size=1)
                pixel_values = [image_transform(stitched_image)]
                pixel_values = torch.stack(pixel_values)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.shape[0])
            pixel_values = torch.cat(pixel_values_list).unsqueeze(0)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            predict_labels = vera.forward(pixel_values, num_patches_list)
            torch.cuda.synchronize()
            runtime = time.time() - t0
            if not (isinstance(predict_labels, list) and len(predict_labels) == 1):
                raise RuntimeError(f"forward() returned {predict_labels!r}; expected a one-element list")
            raw = vera.last_responses[0]
            label, status = vp.strict_parse_label(raw)
            ids = captured["input_ids"][0]
            rendered = tokenizer.decode(ids, skip_special_tokens=False)
            rendered = re.sub(r"(?:<IMG_CONTEXT>)+", lambda m: f"<IMG_CONTEXT>x{len(m.group(0)) // 13}", rendered)
            if not template_path.exists():
                template_path.write_text(rendered, encoding="utf-8")
            rec = {
                "run_id": run_dir.name, "experiment_id": cfg["experiment_id"], "video_id": video_name,
                "video_sha256": vsha, "window_index": k, "anchor_frame": ind,
                "anchor_time_s": ind / actual_fps,
                "window_start": snippet_start, "window_end": snippet_end,
                "window_start_time_s": snippet_start / actual_fps, "window_end_time_s": snippet_end / actual_fps,
                "frame_indices": snippet_indices, "frame_times_s": [i / actual_fps for i in snippet_indices],
                "frame_files": frame_no_list,
                "prompt_question": vera.last_questions[0],
                "model_input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "model_input_tokens": int(ids.shape[0]),
                "model_input_rendered_matches_file": rendered == template_path.read_text(encoding="utf-8"),
                "raw_response": raw,
                "label": label, "parse_status": status,
                "label_upstream_root_parser": predict_labels[0],
                "label_upstream_inference_variant_parser": vp.upstream_inference_variant_parse(raw),
                "mentions_sound": vp.mentions_sound(raw), "audio_supplied": False,
                "n_new_tokens": captured["n_new_tokens"], "runtime_s": round(runtime, 3),
                "peak_gpu_mem_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "generation_config": gen_cfg, "recorded_at": now(),
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            done[ind] = rec
            print(f"[{k + 1}/{len(anchor_list)}] anchor {ind}: label={label} ({status}) upstream={predict_labels[0]} "
                  f"{runtime:.1f}s {captured['n_new_tokens']} tok", flush=True)

    if any(a not in done for a in anchor_list):
        return {"n_anchors_recorded": len(done), "n_anchors_expected": len(anchor_list), "incomplete": True,
                "windows_run_this_invocation": len(todo), "model_load_s": round(load_s, 1), "attention": flash,
                "peak_gpu_mem_allocated_bytes_max": max(r["peak_gpu_mem_allocated_bytes"] for r in done.values())}
    recs = [done[a] for a in anchor_list]
    seg = {str(r["anchor_frame"]): {"start": r["window_start"], "end": r["window_end"], "score": r["label"],
                                    "parse_status": r["parse_status"],
                                    "label_upstream_root_parser": r["label_upstream_root_parser"]} for r in recs}
    n_bad = sum(r["label"] is None for r in recs)
    write_json(run_dir / "results" / "segment_scores.json", {
        "video_id": video_name, "video_sha256": vsha, "n_anchors": len(recs), "n_unparsed": n_bad,
        "note": "score = strict-parsed raw VLM label (0 normal, 1 anomaly) per anchor; null if the output field was "
                "not a single 0/1. label_upstream_root_parser is what the original parser would have produced.",
        "anchors": seg})
    if n_bad == 0:
        write_json(run_dir / "results" / "segment_scores_vera_format.json",
                   {str(r["anchor_frame"]): {"start": r["window_start"], "end": r["window_end"], "score": r["label"]}
                    for r in recs})
    return {"n_anchors": len(recs), "n_unparsed": n_bad, "windows_run_this_invocation": len(todo),
            "model_load_s": round(load_s, 1), "attention": flash,
            "vlm_seconds_this_invocation": round(sum(done[a]["runtime_s"] for a in todo), 1),
            "peak_gpu_mem_allocated_bytes_max": max(r["peak_gpu_mem_allocated_bytes"] for r in recs)}


# ---------------------------------------------------------------- stage: refine (VERA Steps 2-3)

def stage_refine(cfg, run_dir, man):
    res = run_dir / "results"
    vf = res / "segment_scores_vera_format.json"
    if not vf.exists():
        sys.exit("segment_scores_vera_format.json missing: Step 1 incomplete or some outputs unparsed; refusing to "
                 "refine with missing or imputed labels")
    scores = json.loads(vf.read_text(encoding="utf-8"))
    feat_path = ROOT / cfg["features"]["path"]
    fsha = sha256_file(feat_path)
    if fsha != cfg["features"]["sha256"]:
        sys.exit(f"feature checksum {fsha} != config {cfg['features']['sha256']}")
    feats = np.load(feat_path)
    frames_ident = json.loads((res / "frames_identity.json").read_text(encoding="utf-8"))
    n = frames_ident["decoded_frame_count"]
    fps_actual = frames_ident["video_metadata_cv2"]["container_fps"]
    out = vp.vera_refine(scores, feats, cfg["sampling"]["sampling_rate"])
    if len(out["frame_final"]) != n:
        sys.exit(f"frame scores have {len(out['frame_final'])} entries, video has {n} frames")

    seg_common = {"video_id": cfg["video"]["id"], "video_sha256": frames_ident["source_sha256"],
                  "anchors": out["anchors"]}
    write_json(res / "scene_refined_scores.json", {
        **seg_common, "features": {"path": cfg["features"]["path"], "sha256": fsha, "shape": list(feats.shape),
                                   "dtype": str(feats.dtype)},
        "top_n_neighbors": out["top_n"],
        "note": "Step 2 (compute_auc.py): each segment's score is the softmax(10*cos)-weighted mean of the raw "
                "labels of its top-n most similar segments by ImageBind cosine similarity (itself included).",
        "raw_labels": out["raw_labels"].tolist(), "scene_refined": out["scene_refined"].tolist(),
        "neighbor_indices": out["neighbor_indices"].tolist(),
        "neighbor_weights": np.round(out["neighbor_weights"], 6).tolist()})
    np.save(res / "scene_refined_scores.npy", out["scene_refined"])
    write_json(res / "temporal_refined_scores.json", {
        **seg_common,
        "note": "Step 3a (compute_auc.py): Gaussian smoothing over segments (size 15, sigma 10, np.convolve "
                "mode 'same'), then rounding to one decimal before expansion to frames.",
        "temporal_smoothed": out["temporal_smoothed"].tolist(), "temporal_rounded": out["temporal_rounded"].tolist()})
    np.save(res / "temporal_refined_scores.npy", out["temporal_smoothed"])
    np.save(res / "frame_scores.npy", out["frame_final"])
    np.save(res / "frame_scores_before_position_weighting.npy", out["frame_before_position_weighting"])
    with open(res / "frame_scores.csv", "w", encoding="utf-8") as f:
        f.write("frame_index,time_s,frame_score,score_before_position_weighting,position_weight\n")
        for i in range(n):
            f.write(f"{i},{i / fps_actual:.4f},{out['frame_final'][i]:.8f},"
                    f"{out['frame_before_position_weighting'][i]:.1f},{out['position_weights'][i]:.8f}\n")

    # Diagnostics: this video only; benchmark labels are used here and nowhere upstream of this point.
    ann = json.loads((res / "benchmark_annotation.json").read_text(encoding="utf-8"))
    gt = vp.gt_frame_mask(n, ann["temporal_label"])
    diag = {"scope": "single video only; not a benchmark AUC/AP and not comparable to the paper's numbers",
            "gt_positive_frames": int(gt.sum()), "gt_negative_frames": int(n - gt.sum())}
    if 0 < gt.sum() < n:
        from sklearn.metrics import auc, roc_curve
        fpr, tpr, _ = roc_curve(gt, out["frame_final"])
        diag["frame_level_roc_auc_this_video"] = float(auc(fpr, tpr))
        inside = gt == 1
        diag["mean_score_inside_annotation"] = float(out["frame_final"][inside].mean())
        diag["mean_score_outside_annotation"] = float(out["frame_final"][~inside].mean())
    else:
        diag["frame_level_roc_auc_this_video"] = None
        diag["why_no_auc"] = "only one class present in this video's frame labels"

    ref = cfg.get("reference", {}).get("official_initial_scores")
    if ref:
        ref_scores = json.loads((ROOT / ref).read_text(encoding="utf-8"))
        ours = {str(a): scores[str(a)]["score"] for a in out["anchors"]}
        same_keys = list(ref_scores.keys()) == list(scores.keys())
        diag["reference_official_initial_scores"] = {
            "label": "REFERENCE ONLY: labels released by the VERA authors, not generated in this run",
            "path": ref, "sha256": sha256_file(ROOT / ref), "same_anchor_keys": same_keys,
            "agreement_with_fresh_labels": (sum(int(ref_scores[k]["score"] == ours[k]) for k in ours) / len(ours))
            if same_keys else None,
            "official_anchors_labelled_1": [int(k) for k in ref_scores if ref_scores[k]["score"] == 1],
        }
        if same_keys:
            ref_out = vp.vera_refine(ref_scores, feats, cfg["sampling"]["sampling_rate"])
            np.save(res / "reference_official_scores_frame_scores.npy", ref_out["frame_final"])
            if 0 < gt.sum() < n:
                fpr, tpr, _ = roc_curve(gt, ref_out["frame_final"])
                diag["reference_official_initial_scores"]["frame_level_roc_auc_this_video"] = float(auc(fpr, tpr))
    write_json(res / "diagnostics.json", diag)
    return {"frame_scores": len(out["frame_final"]), "features_sha256": fsha, "features_shape": list(feats.shape),
            "top_n": out["top_n"], "diagnostics": diag}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--new-run", action="store_true")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--max-windows", type=int, help="setup smoke tests only: stop Step 1 after N new windows")
    args = ap.parse_args()
    stages = [s for s in args.stages.split(",") if s]
    if any(s not in STAGES for s in stages):
        sys.exit(f"unknown stage in {stages}; choose from {STAGES}")

    cfg_path = (ROOT / args.config) if not os.path.isabs(args.config) else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["_max_windows"] = args.max_windows
    run_dir = resolve_run_dir(cfg, args)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    cfg_sha = freeze_config(cfg_path, run_dir)

    src = ROOT / cfg["source"]["checkout"]
    head = git("rev-parse", "HEAD", cwd=src)
    if head != cfg["source"]["expected_head"]:
        sys.exit(f"source HEAD {head} != config expected_head {cfg['source']['expected_head']}")
    if git("status", "--porcelain", "--untracked-files=no", cwd=src):
        sys.exit("source checkout has uncommitted changes; commit them as a new revision first")

    man = load_manifest(run_dir) or {
        "run_id": run_dir.name, "experiment_id": cfg["experiment_id"], "created_at": now(),
        "host": socket.getfqdn(), "config": rel(cfg_path), "config_sha256": cfg_sha,
        "source": {"repo": cfg["source"]["repo"], "base_commit": cfg["source"]["base_commit"], "head": head,
                   "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=src),
                   "local_patch_vs_base": git("diff", "--stat", cfg["source"]["base_commit"], head, cwd=src),
                   "executed_inference_code": "generate_initial_score.py (root; classes and helpers imported) + "
                                              "setup/scripts/run_experiment.py main loop; "
                                              "Inference/generate_intial_score.py is not used",
                   "refinement_code": "setup/scripts/vera_pipeline.py vera_refine() (transcription of compute_auc.py, "
                                      "checked against the unmodified script by validate_run.py)"},
        "model": cfg["model"], "video": cfg["video"], "sampling": cfg["sampling"], "generation": cfg["generation"],
        "features": cfg["features"], "seed": cfg["seed"], "deviations": cfg.get("deviations", []),
        "runner_sha256": {p.name: sha256_file(p) for p in sorted(SCRIPTS.glob("*.py"))},
        "invocations": [], "stages": {}}
    if man["config_sha256"] != cfg_sha or man["source"]["head"] != head:
        sys.exit("run manifest was created with a different config or source revision")
    inv = {"started_at": now(), "argv": sys.argv, "python": sys.executable, "stages": stages,
           "cwd": os.getcwd(), "job_dir": os.environ.get("JOB_DIR"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    man["invocations"].append(inv)
    write_json(run_dir / "run_manifest.json", man)
    environment_report(run_dir)

    for st in stages:
        t0 = time.time()
        print(f"== stage {st} ({now()})", flush=True)
        try:
            result = {"frames": stage_frames, "vlm": stage_vlm, "refine": stage_refine}[st](cfg, run_dir, man)
            status = "ok"
        except SystemExit as e:
            result, status = {"error": str(e)}, "failed"
        except Exception as e:
            result, status = {"error": repr(e)}, "failed"
            import traceback
            traceback.print_exc()
        man["stages"].setdefault(st, []).append({"status": status, "finished_at": now(),
                                                  "wall_s": round(time.time() - t0, 1), **result})
        write_json(run_dir / "run_manifest.json", man)
        print(f"== stage {st}: {status} {json.dumps(result)[:400]}", flush=True)
        if status != "ok":
            inv["finished_at"] = now()
            write_json(run_dir / "run_manifest.json", man)
            sys.exit(1)
    inv["finished_at"] = now()
    write_json(run_dir / "run_manifest.json", man)
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()

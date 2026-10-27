"""experiment03B preflight and retrieval (PROTOCOL.md §3, §9).

preflight(): read-only checks of source, model, manifest, the GPU this run is pinned to, and (fetch mode) scratch space.
retrieve(): makes every input's eight frames available with recorded identities.
  existing_only  inputs carry frames_dir + frame_sha256 (e.g. experiment02 cases); the JPEGs are re-hashed.
  fetch          per video: use a local copy whose sha256 matches its provenance (unless retrieval.prefer_local is
                 false), or range-fetch the official zip member into scratch (setup/scripts/fetch_zip_member.py);
                 check the member name and byte size against the frozen manifest; probe container fps and frame count;
                 decode the whole video with VERA's loop (cv2 sequential read) and write ONLY the needed frames with
                 cv2.imwrite defaults; write <frames_root>/<video>.frames.json; delete the scratch copy (ledgered).
                 A passing fidelity check (sparse extraction byte-identical to a full VERA extraction of a development
                 video) is required before any extraction.
  none           mock engine only.
Files under shared/videos are never deleted; deletions are limited to files this module created in scratch.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", CODE.parents[2])).resolve()
sys.path.insert(0, str(CODE))
from io3b import append_jsonl, write_json  # noqa: E402

VERA_EXTRACTOR = "setup/source/VERA/Preprocessing/extract_frames.py"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p):
    """Git's object id of a file's content (the hub records this for non-LFS files)."""
    data = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _git(*a, cwd):
    """Read-only git call (no optional index locks). Returns (returncode, stdout, stderr)."""
    r = subprocess.run(["git", "--no-optional-locks", *a], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


# ---------------------------------------------------------------- preflight

def preflight(cfg, inputs, dry=False, need_gpu=True):
    checks, failures, warnings = {}, [], []
    src = ROOT / cfg["source"]["checkout"]
    if src.exists():
        rc1, head, e1 = _git("rev-parse", "--verify", "HEAD", cwd=src)
        rc2, dirty, e2 = _git("status", "--porcelain", "--untracked-files=no", cwd=src)
        clean = None if rc2 else not dirty
        checks["source"] = {"checkout": cfg["source"]["checkout"], "head": head if not rc1 else None, "clean": clean}
        if rc1 or rc2:
            failures.append(f"git failed in {cfg['source']['checkout']}: {(e1 or e2)[-200:]}")
        elif head != cfg["source"]["expected_head"] or not clean:
            failures.append(f"source checkout {cfg['source']['checkout']} not at the clean expected head")
    else:
        failures.append(f"source checkout {cfg['source']['checkout']} missing")
    if cfg["engine"] == "real":
        md = ROOT / cfg["model"]["local_dir"]
        prov = md / "MODEL_PROVENANCE.json"
        ok = prov.exists()
        if ok:
            p = json.loads(prov.read_text())
            files = p.get("files", {})
            small = ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json")
            blob_ok = {f: (md / f).exists() and git_blob_sha1(md / f) == (files.get(f) or {}).get("git_blob_sha1")
                       for f in small}
            sizes_ok = {f: (md / f).exists() and (md / f).stat().st_size == meta.get("bytes")
                        for f, meta in files.items() if f.endswith(".safetensors")}
            ok = p.get("revision") == cfg["model"]["revision"] and all(blob_ok.values()) and all(sizes_ok.values()) \
                and len(sizes_ok) > 0
            checks["model"] = {"local_dir": cfg["model"]["local_dir"], "provenance_revision": p.get("revision"),
                               "git_blob_sha1_match": blob_ok, "weight_shard_sizes_match": sizes_ok}
        if not ok:
            failures.append("model directory/provenance does not match the configured revision")
        g = gpus(cfg["gpu"]["min_free_mib"])
        checks["gpus"] = g
        target = target_gpu(g)
        checks["gpu_target"] = target
        if target.get("problem"):  # only invocations that run model stages need a pinned, idle GPU
            (warnings if (dry or not need_gpu) else failures).append(f"GPU: {target['problem']}")
    for name, path in (("home", Path.home()), ("scratch", Path(cfg["retrieval"]["scratch_dir"]))):
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            st = os.statvfs(probe)
            checks[f"free_bytes_{name}"] = st.f_bavail * st.f_frsize
            checks[f"{name}_probe_path"] = str(probe)
        except OSError as e:
            checks[f"free_bytes_{name}"] = repr(e)
    q = Path("/usr/local/pub/jmc/home2_quota")
    if q.exists():
        user = os.environ.get("USER", "")
        checks["home_quota_line"] = next((l.strip() for l in q.read_text(errors="replace").splitlines()
                                          if l.split()[:1] == [user]), None)
    checks["inputs"] = {"n": len(inputs), "videos": len({i["video_id"] for i in inputs}),
                        "development_inputs": sum(bool(i.get("development")) for i in inputs)}
    if cfg["population"]["kind"] == "manifest":
        man = json.loads((ROOT / cfg["population"]["manifest"]).read_text())
        c = man["counts"]
        want = {"videos": 290, "anomalous": 140, "normal": 150, "anomaly_categories": 13, "intervals": 162}
        checks["manifest_counts"] = {k: c[k] for k in want}
        if any(c[k] != v for k, v in want.items()):
            failures.append(f"manifest counts differ: {checks['manifest_counts']}")
        need = {i["video_id"] for i in inputs}
        to_fetch = [v for v in man["videos"] if v["video_id"] in need and not (
            cfg["retrieval"].get("prefer_local", True) and (ROOT / local_video(cfg, v["video_id"])).exists())]
        checks["download_bytes_needed"] = sum(v["video_bytes"] for v in to_fetch)
        checks["largest_video_bytes_needed"] = max((v["video_bytes"] for v in to_fetch), default=0)
        if cfg["retrieval"]["mode"] == "fetch" and to_fetch:
            free = checks.get("free_bytes_scratch")
            big = checks["largest_video_bytes_needed"]
            lim = cfg["limits"]["max_scratch_mib"]
            if not isinstance(free, int):
                failures.append(f"scratch filesystem unreadable: {free}")
            elif free < big + (64 << 20):
                failures.append(f"scratch has {free} bytes free; the largest needed video is {big} bytes")
            if lim is not None and lim * 2 ** 20 < big:
                failures.append(f"limits.max_scratch_mib ({lim}) is smaller than the largest needed video ({big} bytes)")
    return {"at": now(), "dry_run": dry, "checks": checks, "failures": failures, "warnings": warnings}


def gpus(min_free_mib):
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": repr(e), "all": [], "eligible": [], "min_free_mib": min_free_mib}
    rows = []
    for line in r.stdout.strip().splitlines():
        idx, uuid, name, used, total = [x.strip() for x in line.split(",")]
        free = int(total) - int(used)
        rows.append({"index": int(idx), "uuid": uuid, "name": name, "free_mib": free, "busy": uuid in apps})
    return {"all": rows, "eligible": [g["index"] for g in rows if g["free_mib"] >= min_free_mib and not g["busy"]],
            "min_free_mib": min_free_mib}


def target_gpu(g):
    """The device the engine will use: the first entry of CUDA_VISIBLE_DEVICES, which launch.sh pins by UUID."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return {"cuda_visible_devices": None, "problem": "CUDA_VISIBLE_DEVICES is unset; launch through setup/scripts/launch.sh, "
                                                         "which pins one idle GPU by UUID"}
    first = cvd.split(",")[0].strip()
    if not first:
        return {"cuda_visible_devices": cvd, "problem": "CUDA_VISIBLE_DEVICES is empty (no device)"}
    if not first.startswith(("GPU-", "MIG-")):
        return {"cuda_visible_devices": cvd, "problem": "pin the GPU by UUID (launch.sh does this); an index cannot be "
                                                       "matched reliably to nvidia-smi"}
    rows = [r for r in g.get("all", []) if r["uuid"].startswith(first)]
    if len(rows) != 1:
        return {"cuda_visible_devices": cvd, "problem": f"{first} matches {len(rows)} GPUs"}
    r = rows[0]
    problem = None
    if r["busy"]:
        problem = f"{r['uuid']} already has compute processes"
    elif r["free_mib"] < g["min_free_mib"]:
        problem = f"{r['uuid']} has {r['free_mib']} MiB free < {g['min_free_mib']}"
    return {"cuda_visible_devices": cvd, "device": r, "problem": problem}


# ---------------------------------------------------------------- retrieval

def local_video(cfg, vid):
    return f"shared/videos/ucf_crime/{vid}.mp4"


def sparse_extract(video_path, out_dir, needed):
    """VERA's extract_frames loop (cv2.VideoCapture, sequential read, cv2.imwrite defaults, {:06d}.jpg), writing only
    the frame indices in `needed`. Decodes the whole video to count frames. Returns metadata and per-frame sha256
    (None for an index the video does not decode)."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted(set(int(i) for i in needed))
    cap = cv2.VideoCapture(str(video_path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    meta = {"container_fps": cap.get(cv2.CAP_PROP_FPS), "container_frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)), "cv2_version": cv2.__version__}
    want = set(needed)
    frame_count = 0
    written = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in want:
            p = out_dir / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(p), frame)
            written[frame_count] = sha256_file(p)
        frame_count += 1
    cap.release()
    meta["decoded_frame_count"] = frame_count
    meta["frames"] = {str(i): written.get(i) for i in needed}
    meta["missing_indices"] = [i for i in needed if i not in written]
    return meta


def _ledger(rd, tag, rec):
    append_jsonl(Path(rd) / f"retrieval{tag}.jsonl", {"at": now(), **rec})


def _purge(rd, tag, paths):
    """Delete files this module created in scratch (never anything under shared/videos)."""
    for p in paths:
        p = Path(p)
        if p.exists() and p.is_file() and "shared/videos" not in str(p.resolve()):
            size = p.stat().st_size
            p.unlink()
            _ledger(rd, tag, {"event": "purge", "path": str(p), "bytes": size})


def _fetch_member(cfg, rd, tag, video, scratch):
    url = cfg["retrieval"]["archives"][video["zip"]]
    scratch = Path(scratch)
    out = scratch / f"{video['video_id']}.mp4"
    part, provp = Path(str(out) + ".part"), Path(str(out) + ".provenance.json")
    _purge(rd, tag, [out, part, provp])  # stale leftovers of an earlier, interrupted attempt
    lim = cfg["limits"]["max_scratch_mib"]
    used = sum(f.stat().st_size for f in scratch.rglob("*") if f.is_file()) if scratch.exists() else 0
    if lim is not None and used + video["video_bytes"] > lim * 2 ** 20:
        raise RuntimeError(f"scratch limit: {used + video['video_bytes']} bytes > {lim} MiB")
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-I", str(ROOT / "setup/scripts/fetch_zip_member.py"), "--url", url,
           "--member", video["zip_member"], "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    _ledger(rd, tag, {"event": "fetch", "video_id": video["video_id"], "path": str(out), "exit": r.returncode,
                      "stderr_tail": r.stderr[-300:]})
    try:
        if r.returncode != 0:
            raise RuntimeError(f"fetch failed: {r.stderr[-300:]}")
        prov = json.loads(provp.read_text())
    except Exception:
        _purge(rd, tag, [out, part, provp])
        raise
    return out, prov


def retrieve(cfg, rd, inputs, dry=False, tag=""):
    mode = cfg["retrieval"]["mode"]
    summary = {"mode": mode, "ok": 0, "failed": 0, "planned_fetch_bytes": 0, "videos": {}}
    by_video = {}
    for i in inputs:
        by_video.setdefault(i["video_id"], []).append(i)
    man = None
    if cfg["population"]["kind"] == "manifest":
        man = {v["video_id"]: v for v in json.loads((ROOT / cfg["population"]["manifest"]).read_text())["videos"]}
    if mode == "fetch" and not dry:  # the sparse extractor must reproduce VERA's full extraction on this host first
        summary["fidelity_check"] = fidelity_check(cfg, rd, tag)
        if not summary["fidelity_check"]["identical"]:
            raise RuntimeError(f"sparse extraction differs from the full frame cache: {summary['fidelity_check']}")
    for vid, inps in sorted(by_video.items()):
        if mode == "none":  # mock engine only (config check enforces it): no frames are read
            for inp in inps:
                inp["status"] = "ok"
            summary["videos"][vid] = ["ok"]
            continue
        if mode == "existing_only":
            for inp in inps:
                fdir = ROOT / inp["frames_dir"]
                got = [sha256_file(fdir / f"{fi:06d}.jpg") if (fdir / f"{fi:06d}.jpg").exists() else None
                       for fi in inp["frame_indices"]]
                if None in got:
                    inp["status"] = "fetch_failed"
                    inp["status_detail"] = "frames missing"
                elif inp.get("frame_sha256") and got != inp["frame_sha256"]:
                    inp["status"] = "fetch_failed"
                    inp["status_detail"] = "frame hashes differ from the recorded ones"
                else:
                    inp["frame_sha256"], inp["status"] = got, "ok"
            summary["videos"][vid] = sorted({i["status"] for i in inps})
            continue
        v = man[vid]
        fdir = ROOT / cfg["retrieval"]["frames_root"] / vid
        ident_p = ROOT / cfg["retrieval"]["frames_root"] / f"{vid}.frames.json"
        needed = sorted({f for i in inps for f in i["frame_indices"]})
        if ident_p.exists():
            ident = json.loads(ident_p.read_text())
            same_tools = (ident.get("extractor_sha256") == sha256_file(Path(__file__))
                          and ident.get("vera_extractor_sha256") == sha256_file(ROOT / VERA_EXTRACTOR))
            covered = all(str(f) in ident["frames"] for f in needed)

            def frame_ok(f):
                h, p = ident["frames"].get(str(f)), fdir / f"{f:06d}.jpg"
                return h is None or (p.exists() and sha256_file(p) == h)  # None = index never decoded (terminal)
            if same_tools and covered and all(frame_ok(f) for f in needed):
                _apply_identity(inps, ident, cfg)
                summary["videos"][vid] = sorted({i["status"] for i in inps})
                continue
            _ledger(rd, tag, {"event": "identity_stale", "video_id": vid, "same_tools": same_tools, "covered": covered})
        local = ROOT / local_video(cfg, vid)
        use_local = cfg["retrieval"].get("prefer_local", True) and local.exists()
        if dry:
            summary["planned_fetch_bytes"] += 0 if use_local else v["video_bytes"]
            continue
        scratch = Path(cfg["retrieval"]["scratch_dir"])
        fetched = []
        try:
            if use_local:
                video_path, prov = local, json.loads(Path(str(local) + ".provenance.json").read_text())
                if sha256_file(local) != prov["sha256"]:
                    raise RuntimeError("local video sha256 differs from its provenance")
            else:
                video_path, prov = _fetch_member(cfg, rd, tag, v, scratch)
                fetched = [video_path, Path(str(video_path) + ".provenance.json")]
            if prov.get("member") != v["zip_member"] or prov.get("member_bytes") != v["video_bytes"]:
                raise RuntimeError(f"source identity differs from the manifest: {prov.get('member')}/"
                                   f"{prov.get('member_bytes')} vs {v['zip_member']}/{v['video_bytes']}")
            meta = sparse_extract(video_path, fdir, needed)
            flags = []
            if abs(meta["container_fps"] - 30.0) > 1e-3:
                flags.append("fps_not_30")
            if meta["decoded_frame_count"] != v["length"]:
                flags.append("decoded_count_mismatch")
            if meta["container_frame_count"] != meta["decoded_frame_count"]:
                flags.append("container_count_mismatch")
            ident = {"video_id": vid, "source_sha256": prov["sha256"], "zip": v["zip"], "zip_member": v["zip_member"],
                     "manifest_length": v["length"], "extractor": "experiments/experiment03B/code/acquire3b.py::sparse_extract "
                     "(VERA extract_frames loop; writes only the needed indices)", "extractor_sha256": sha256_file(Path(__file__)),
                     "vera_extractor_sha256": sha256_file(ROOT / VERA_EXTRACTOR), **meta, "metadata_flags": flags,
                     "source": "local copy" if use_local else "range-fetched member",
                     "provenance": {k: prov.get(k) for k in ("member", "member_bytes", "member_crc32", "sha256", "resolved_url_host")},
                     "extracted_at": now()}
            write_json(ident_p, ident)
            _apply_identity(inps, ident, cfg)
            _ledger(rd, tag, {"event": "extracted", "video_id": vid, "flags": flags, "n_frames": len(needed),
                              "missing_indices": meta["missing_indices"]})
        except Exception as e:  # recorded; the video's inputs stay visible with a failure status
            for i in inps:
                i["status"] = "fetch_failed"
                i["status_detail"] = repr(e)[:300]
            _ledger(rd, tag, {"event": "failed", "video_id": vid, "error": repr(e)})
        finally:
            _purge(rd, tag, fetched)
        summary["videos"][vid] = sorted({i["status"] for i in inps})
    summary["ok"] = sum(i.get("status") == "ok" for i in inputs)
    summary["failed"] = sum(i.get("status") not in ("ok", "selected") for i in inputs)
    if mode != "fetch" and cfg["retrieval"].get("fidelity_check") and not dry:
        summary["fidelity_check"] = fidelity_check(cfg, rd, tag)
        if not summary["fidelity_check"]["identical"]:
            raise RuntimeError(f"sparse extraction differs from the full frame cache: {summary['fidelity_check']}")
    return summary


def _apply_identity(inps, ident, cfg):
    for i in inps:
        shas = [ident["frames"].get(str(f)) for f in i["frame_indices"]]
        i["frame_sha256"] = shas
        i["video_sha256"] = ident["source_sha256"]
        i["metadata_flags"] = list(ident.get("metadata_flags", []))
        i["frames_dir"] = f"{cfg['retrieval']['frames_root']}/{i['video_id']}"
        i["status"] = "ok" if all(shas) else "invalid_index"


def fidelity_check(cfg, rd, tag):
    """Fetch (or use) one development video, sparse-extract its needed frames into scratch and compare them with the
    full VERA frame cache. Scratch copies are purged afterwards. A passing result is reused within the run."""
    p = Path(rd) / "fidelity_check.json"
    if p.exists():
        prev = json.loads(p.read_text())
        if prev.get("identical"):
            return prev
    fc = cfg["retrieval"]["fidelity_check"]
    scratch = Path(cfg["retrieval"]["scratch_dir"]) / "fidelity"
    out = {"video_id": fc["video_id"], "fetched": False, "at": now()}
    fetched = []
    try:
        if fc.get("fetch"):
            v = {"video_id": fc["video_id"], "zip": fc["zip"], "zip_member": fc["zip_member"], "video_bytes": fc["video_bytes"]}
            video_path, prov = _fetch_member(cfg, rd, tag, v, scratch)
            fetched = [video_path, Path(str(video_path) + ".provenance.json")]
            out["fetched"] = True
            out["fetched_sha256"] = prov["sha256"]
            out["fetched_sha256_matches_recorded"] = prov["sha256"] == fc["expected_sha256"]
            out["bytes_fetched_over_http"] = prov.get("bytes_fetched_over_http")
        else:
            video_path = ROOT / fc["video_path"]
        frames_dir = scratch / "frames"
        meta = sparse_extract(video_path, frames_dir, fc["frame_indices"])
        full = ROOT / fc["full_cache_dir"]
        cmp_ = {str(i): (meta["frames"][str(i)], sha256_file(full / f"{i:06d}.jpg")) for i in fc["frame_indices"]}
        out.update({"decoded_frame_count": meta["decoded_frame_count"], "container_fps": meta["container_fps"],
                    "cv2_version": meta["cv2_version"], "n_frames": len(cmp_), "frames": cmp_,
                    "identical": all(a == b for a, b in cmp_.values()) and out.get("fetched_sha256_matches_recorded", True)})
        fetched += [frames_dir / f"{i:06d}.jpg" for i in fc["frame_indices"]]
    finally:
        _purge(rd, tag, fetched)
        _ledger(rd, tag, {"event": "fidelity_check", **{k: v for k, v in out.items() if k != "frames"}})
    write_json(p, out)
    return out

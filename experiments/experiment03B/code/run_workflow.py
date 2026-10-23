#!/usr/bin/env python3
"""experiment03B workflow runner (PROTOCOL.md §9, RECORDS.md).

Stages, in dependency order:
  preflight -> errata -> manifest -> retrieval -> baseline -> fixed -> adaptive -> evaluation -> analysis ->
  validation -> report
Every stage is idempotent: requests already answered (logged as fresh/cached/prior_cache) are never recomputed, so
--resume continues after an interruption at request granularity. Model outputs are cached by experiment02's request
keys; caches named in config prior_caches (experiment02 run_002, or an earlier 3B run) are read-only layers.

    python run_workflow.py --config CONFIG --seeds 0 --dry-run
    python run_workflow.py --config CONFIG --seeds 0 --resume
    python run_workflow.py --config CONFIG --seeds 0 --ranking-mode adaptive --resume
    python run_workflow.py --config CONFIG --seeds 0 1 2 --resume --stages analysis validation report

Failure semantics: a limit (GPU hours, model calls, new transformed inputs) that is reached stops the run with a failed
stage; errors are retried inside the request up to retries.error_retries; any failed stage skips its dependents and
the process exits 1. One exclusive lock per run directory: invocations (including --videos/--shard subsets) run one at
a time. Seeds change bootstrap draws only; model requests are deterministic and shared across seeds.
"""
import argparse
import difflib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", CODE.parents[2])).resolve()
sys.path.insert(0, str(CODE))
import config3b as c3  # noqa: E402
import io3b  # noqa: E402
import xb_core as xc  # noqa: E402
from io3b import read_jsonl, append_jsonl, write_json  # noqa: E402

CODE_FILES = ["experiments/experiment03B/code/" + f for f in (
    "run_workflow.py", "xb_core.py", "config3b.py", "io3b.py", "engine3b.py", "acquire3b.py", "analyze3b.py",
    "appendix_table.py", "manifest3b.py", "validate3b.py")] + [
    "experiments/experiment02/code/engine.py", "experiments/experiment02/code/attr_lib.py",
    "setup/scripts/vera_pipeline.py", "setup/scripts/fetch_zip_member.py",
    "experiments/experiment02/errata/precision_crossings.py"]
SUCCESS = ("fresh", "cached", "prior_cache")
GPU_STAGES = ("baseline", "fixed", "adaptive", "evaluation")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes():
    return {f: sha256_file(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()}


def readout(rec, field):
    """S_hi, or the protocol v1.1 native margin S_native = logit_1 - logit_0 (exact BF16 logits; the stored S carries
    ~1e-7 of log-softmax rounding). None for a missing record or value."""
    if rec is None:
        return None
    if field == "S_native":
        l0, l1 = rec.get("logit_0"), rec.get("logit_1")
        return None if l0 is None or l1 is None else l1 - l0
    return rec.get(field)


def rid_of(uid, target, family, mask, repeat=0):
    mask = list(xc.norm_mask(mask))
    return f"{uid}|{target}|{family if mask else 'none'}|{','.join(map(str, mask))}|r{repeat}"


class StageFailed(Exception):
    pass


class Budget(StageFailed):
    pass


# ---------------------------------------------------------------- request store

class Store:
    """Request logs + forward cache (append-only; the latest successful line per request id wins)."""

    def __init__(self, cfg, rd, shard_tag, engine, prior_gpu_s, inv):
        self.cfg, self.rd, self.tag, self.eng, self.inv = cfg, Path(rd), shard_tag, engine, inv
        self.cache_path = self.rd / "cache" / f"forward_cache{shard_tag}.jsonl"
        self.cache = {}
        for p in sorted((self.rd / "cache").glob("forward_cache*.jsonl")):
            for r in read_jsonl(p):
                self.cache[r["key"]] = r
        self.prior, self.prior_seq = {}, set()
        for rel in cfg["prior_caches"]:
            for r in read_jsonl(ROOT / rel):
                self.prior[r["key"]] = r
                if r.get("transformed"):
                    self.prior_seq.add(r["sequence_key"])
        self.new_seq = {r["sequence_key"] for r in self.cache.values()
                        if r.get("transformed") and r["sequence_key"] not in self.prior_seq}
        self.model_calls = len(self.cache)
        self.logs = {}            # stage -> {rid: latest line}
        self.latest = {}          # rid -> latest line over all stage logs
        self.errors = {}          # rid -> error attempts over all stage logs and invocations
        self.attempts = {}        # rid -> lines written for it over all stage logs
        for p in sorted((self.rd / "requests").glob("*.jsonl")):
            stage = p.name.split(".")[0]
            for r in read_jsonl(p):
                self._index(stage, r)
        self.lim = cfg["limits"]
        self.prior_gpu_s = prior_gpu_s
        self.t_model = None
        self.counts = {"fresh": 0, "cached": 0, "prior_cache": 0, "error": 0, "skipped_budget": 0}

    def _index(self, stage, r):
        rid = r["request_id"]
        self.logs.setdefault(stage, {})[rid] = r
        self.latest[rid] = r
        self.attempts[rid] = self.attempts.get(rid, 0) + 1
        if r["cache_status"] == "error":
            self.errors[rid] = self.errors.get(rid, 0) + 1

    def gpu_s_this_invocation(self):
        return (time.time() - self.t_model) if self.t_model else 0.0

    def gpu_s(self):
        return self.prior_gpu_s + self.gpu_s_this_invocation()

    def _heartbeat(self):
        write_json(self.rd / "gpu_time" / f"{self.inv}.json", {"invocation": self.inv, "updated_at": now(),
                                                               "gpu_s": round(self.gpu_s_this_invocation(), 2)})

    def lookup(self, stage, rid):
        r = self.logs.get(stage, {}).get(rid)
        return r if r is not None and r["cache_status"] in SUCCESS else None

    def exhausted(self, rid):
        return self.errors.get(rid, 0) >= 1 + self.cfg["retries"]["error_retries"]

    def request(self, stage, inp, phase, purpose, target, family, mask, repeat=0):
        mask = list(xc.norm_mask(mask))
        fam = family if mask else "none"
        rid = rid_of(inp["window_uid"], target, fam, mask, repeat)
        prev = self.lookup(stage, rid)
        if prev is not None:
            return prev
        key, seq = self.eng.keymaker.keys(inp, target, fam, tuple(mask), repeat)
        transformed = bool(mask)
        meta = {"request_id": rid, "window_uid": inp["window_uid"], "video_id": inp["video_id"], "phase": phase,
                "purpose": purpose, "target": target, "family": fam, "mask": mask, "repeat": repeat}
        while True:
            if self.exhausted(rid):  # retries used up: stays an error (missing), never a label
                return self.latest[rid]
            if key in self.cache:
                return self._write(stage, meta, "cached", key, seq, self.cache[key])
            if key in self.prior:
                return self._write(stage, meta, "prior_cache", key, seq, self.prior[key])
            reason = self._over_budget(transformed, seq)
            if reason:
                self._write(stage, meta, "skipped_budget", key, seq, {"error": f"skipped: {reason}"})
                raise Budget(reason)
            if self.t_model is None:
                self.t_model = time.time()  # the clock includes model loading
            try:
                out = self.eng.run(target, inp, fam, mask)
            except Exception as e:  # recorded as an error state, never as a label
                res = {"error": repr(e), "traceback": traceback.format_exc()[-2000:]}
                self._write(stage, meta, "error", key, seq, res)
                self._heartbeat()
                if "CUDA" in repr(e) or "cuda" in type(e).__name__.lower():
                    raise StageFailed(f"device error, stopping: {e!r}")
                continue  # retry the identical request (bounded by retries.error_retries)
            res = {"key": key, "sequence_key": seq, "transformed": transformed, "computed_at": now(), **out}
            append_jsonl(self.cache_path, res)
            self.cache[key] = res
            self.model_calls += 1
            if transformed and seq not in self.prior_seq:
                self.new_seq.add(seq)
            self._heartbeat()
            return self._write(stage, meta, "fresh", key, seq, res)

    def _over_budget(self, transformed, seq):
        lim = self.lim
        if lim["max_gpu_hours"] is not None and self.gpu_s() >= lim["max_gpu_hours"] * 3600:
            return f"GPU-hour limit {lim['max_gpu_hours']} reached ({self.gpu_s() / 3600:.3f} h used)"
        if lim["max_model_calls"] is not None and self.model_calls >= lim["max_model_calls"]:
            return f"model-call limit {lim['max_model_calls']} reached"
        if (transformed and lim["max_new_transformed_inputs"] is not None and seq not in self.prior_seq
                and seq not in self.new_seq and len(self.new_seq) >= lim["max_new_transformed_inputs"]):
            return f"new-transformed-input limit {lim['max_new_transformed_inputs']} reached"
        return None

    def _write(self, stage, meta, status, key, seq, res):
        self.counts[status] += 1
        rec = {**meta, "cache_status": status, "result_key": key, "sequence_key": seq,
               "attempt": self.attempts.get(meta["request_id"], 0) + 1,
               **{k: v for k, v in res.items() if k not in ("key", "sequence_key", "transformed")}}
        append_jsonl(self.rd / "requests" / f"{stage}{self.tag}.jsonl", rec)
        self._index(stage, rec)
        return rec

    def label(self, stage, inp, phase, purpose, family, mask, target="original_generation"):
        """Generation label with the parse-failure repeat rule (repeat 0, then 1..N). Returns (label, records)."""
        recs = []
        for rep in range(0, 1 + self.cfg["retries"]["parse_failure_repeats"]):
            r = self.request(stage, inp, phase, purpose if rep == 0 else purpose + "_parse_repeat", target, family,
                             mask, rep)
            recs.append(r)
            if r["cache_status"] not in SUCCESS:
                return None, recs
            if r.get("label") is not None:
                return r["label"], recs
        return None, recs

    def terminal(self, rid):
        r = self.latest.get(rid)
        return r is not None and (r["cache_status"] in SUCCESS or self.exhausted(rid))


# ---------------------------------------------------------------- inputs

def build_inputs(cfg, args):
    pop = cfg["population"]
    if pop["kind"] == "cases":
        inputs = [dict(c) for c in pop["cases"]]
    else:
        mp = ROOT / pop["manifest"]
        if sha256_file(mp) != pop["manifest_sha256"]:
            raise StageFailed(f"manifest {mp} does not match population.manifest_sha256 (manifests are immutable)")
        man = json.loads(mp.read_text(encoding="utf-8"))
        vids = {v["video_id"]: v for v in man["videos"]}
        inputs = []
        for u in man["unique_inputs"]:
            v = vids[u["video_id"]]
            inputs.append({**u, "manifest_entry": v["manifest_entry"], "manifest_length": v["length"],
                           "frames_dir": f"{cfg['retrieval']['frames_root']}/{u['video_id']}",
                           "frame_sha256": None, "video_sha256": None, "metadata_flags": [], "status": "selected"})
    if args.videos:
        unknown = set(args.videos) - {i["video_id"] for i in inputs}
        if unknown:
            raise StageFailed(f"--videos not in the population: {sorted(unknown)}")
        inputs = [i for i in inputs if i["video_id"] in set(args.videos)]
    if args.shard:
        try:
            k, n = map(int, args.shard.split("/"))
        except ValueError:
            raise StageFailed("--shard must be k/n")
        if not (n >= 1 and 0 <= k < n):
            raise StageFailed(f"--shard {args.shard}: need 0 <= k < n")
        inputs = [i for i in inputs if int(hashlib.sha256(i["video_id"].encode()).hexdigest(), 16) % n == k]
    if not inputs:
        raise StageFailed("no inputs selected (check --videos/--shard)")
    return sorted(inputs, key=lambda i: (i["video_id"], i["anchor"]))


def planned_uids(cfg):
    pop = cfg["population"]
    if pop["kind"] == "cases":
        return sorted(c["window_uid"] for c in pop["cases"])
    return sorted(u["window_uid"] for u in json.loads((ROOT / pop["manifest"]).read_text())["unique_inputs"])


# ---------------------------------------------------------------- stages

class Runner:
    def __init__(self, cfg, args, rd, inv):
        self.cfg, self.args, self.rd, self.inv = cfg, args, Path(rd), inv
        self.tag = "" if not args.shard else ".shard" + args.shard.replace("/", "of")
        self.inputs = []
        self.store = None
        self.engine = None
        self.stages = list(c3.STAGES)

    def ledger(self, stage, event, detail=None):
        if self.args.dry_run:
            return
        append_jsonl(self.rd / "ledger.jsonl", {
            "stage": stage, "event": event, "at": now(), "invocation": self.inv, "shard": self.args.shard,
            "videos": self.args.videos, "ranking_mode": self.args.ranking_mode, "seeds": self.args.seeds,
            "detail": detail})

    def prior_gpu_s(self):
        """GPU seconds of every earlier invocation, from heartbeat files (written after each model call, so a hard
        kill loses at most the call in flight)."""
        total = 0.0
        for p in (self.rd / "gpu_time").glob("*.json") if (self.rd / "gpu_time").exists() else []:
            if p.stem != self.inv:
                total += json.loads(p.read_text()).get("gpu_s", 0.0)
        return total

    def make_engine(self):
        import engine3b
        if self.cfg["engine"] == "mock":
            spec = json.loads((ROOT / self.cfg["mock_spec"]).read_text(encoding="utf-8"))
            return engine3b.MockEngine(self.cfg, self.rd, self.inputs, spec)
        return engine3b.RealEngine(self.cfg, self.rd, self.inputs)

    def ready_inputs(self):
        return [i for i in self.inputs if i.get("status") == "ok"]

    def gate_retrieved(self):
        """Before any model stage: every selected input must have been processed by retrieval in this run."""
        p = self.rd / "inputs.json"
        saved = {i["window_uid"]: i for i in json.loads(p.read_text())["inputs"]} if p.exists() else {}
        missing = [i["window_uid"] for i in self.inputs if i["window_uid"] not in saved
                   or saved[i["window_uid"]].get("status") in (None, "selected")]
        if missing:
            raise StageFailed(f"retrieval has not processed {len(missing)} selected inputs (e.g. {missing[:3]})")
        for i in self.inputs:
            i.update(saved[i["window_uid"]])

    # -- stages
    def st_preflight(self):
        import acquire3b
        rep = acquire3b.preflight(self.cfg, self.inputs, dry=self.args.dry_run,
                                  need_gpu=any(s in GPU_STAGES for s in self.stages))
        if not self.args.dry_run:
            write_json(self.rd / f"preflight{self.tag}.json", rep)
        if rep["failures"]:
            raise StageFailed(f"preflight failures: {rep['failures']}")
        return {"checks": {k: v for k, v in rep["checks"].items() if k != "gpus"}, "warnings": rep.get("warnings")}

    def st_errata(self):
        """E1: re-derive experiment02's precision erratum counts from its effect table (CPU, read-only)."""
        p = ROOT / "experiments/experiment02/errata/precision_crossings.py"
        with tempfile.TemporaryDirectory() as td:
            out = (Path(td) if self.args.dry_run else self.rd) / "errata_E1.json"
            r = subprocess.run([sys.executable, "-B", str(p), "--effects",
                                str(ROOT / "experiments/experiment02/runs/run_002/results/temporal_effects.csv"),
                                "--out", str(out)], capture_output=True, text=True)
            if r.returncode != 0:
                raise StageFailed(f"erratum script failed: {r.stderr[-500:]}")
            tot = json.loads(out.read_text())["totals"]
        want = {"sub_step": 109, "sub_step_but_bf16_moved": 36, "step_or_more_but_bf16_unchanged": 2, "opposite_sign": 4}
        bad = {k: (tot.get(k), v) for k, v in want.items() if tot.get(k) != v}
        if bad:
            raise StageFailed(f"E1 counts differ from the recorded erratum: {bad}")
        return {"E1": tot}

    def st_manifest(self):
        if self.args.dry_run:
            return {"inputs": len(self.inputs)}
        if self.cfg["population"]["kind"] == "manifest":
            shutil.copyfile(ROOT / self.cfg["population"]["manifest"], self.rd / "frozen" / "manifest.json")
        else:
            cases = self.cfg["population"]["cases"]
            lengths = {}
            for c in cases:
                if lengths.setdefault(c["video_id"], c["manifest_length"]) != c["manifest_length"]:
                    raise StageFailed(f"cases disagree on the length of {c['video_id']}")
            write_json(self.rd / "frozen" / "manifest.json", {
                "version": "cases", "cases": cases,
                "videos": [{"video_id": v, "length": n, "development": v in xc.DEVELOPMENT_VIDEOS}
                           for v, n in sorted(lengths.items())],
                "unique_inputs": [{"window_uid": c["window_uid"], "window_record_ids": c["window_record_ids"],
                                   "strata": c["strata"]} for c in cases]})
        return {"inputs": len(self.inputs), "videos": len({i["video_id"] for i in self.inputs})}

    def st_retrieval(self):
        import acquire3b
        res = acquire3b.retrieve(self.cfg, self.rd, self.inputs, dry=self.args.dry_run, tag=self.tag)
        if not self.args.dry_run:
            p = self.rd / "inputs.json"
            existing = {i["window_uid"]: i for i in json.loads(p.read_text())["inputs"]} if p.exists() else {}
            for i in self.inputs:
                existing[i["window_uid"]] = {k: i.get(k) for k in (
                    "window_uid", "video_id", "category", "anchor", "frame_indices", "frame_sha256", "video_sha256",
                    "strata", "window_record_ids", "development", "metadata_flags", "annotated_slots", "clipped",
                    "status", "frames_dir", "manifest_entry", "manifest_length")}
            write_json(p, {"inputs": sorted(existing.values(), key=lambda i: i["window_uid"])})
        bad = [i["window_uid"] for i in self.inputs if i.get("status") in (None, "selected")]
        if bad:
            raise StageFailed(f"retrieval left {len(bad)} inputs unprocessed")
        return res

    def st_baseline(self):
        self.write_prompts()
        for inp in self.ready_inputs():
            self.store.request("baseline", inp, "baseline", "baseline", "label_only_score", "none", ())
            self.store.label("baseline", inp, "baseline", "baseline", "none", ())
            self.store.label("baseline", inp, "baseline", "nosound", "none", (), target="nosound_generation")
        if self.cfg.get("integration_checks", {}).get("fresh_recompute"):
            self.fresh_recompute()
        if self.cfg["engine"] == "real" and self.ready_inputs():
            import engine3b
            engine3b.save_example_inputs(self.cfg, self.ready_inputs()[0], self.rd / "results" / "example_inputs")
        return self.store.counts.copy()

    def write_prompts(self):
        """The exact question strings for the three targets, the no-sound diff, and a comparison with experiment02."""
        import engine3b
        pd = self.rd / "results" / "prompts"
        pd.mkdir(parents=True, exist_ok=True)
        if (pd / "prompts_check.json").exists():
            return
        out = {}
        for v in ("original", "nosound", "label_only"):
            text = engine3b.effective_question_text(self.cfg, v)
            (pd / f"{v}_question.txt").write_text(text, encoding="utf-8")
            e02 = ROOT / "experiments/experiment02/runs/run_002/results/prompts" / f"{v}_question.txt"
            out[v] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "equals_experiment02_run_002": e02.exists() and e02.read_text(encoding="utf-8") == text}
        qs = self.cfg["prompt"]["guiding_questions"]
        diff = "".join(difflib.unified_diff([q + "\n" for q in qs], [q + "\n" for q in qs if not q.startswith("5.")],
                                            "original_questions", "nosound_questions"))
        (pd / "nosound_questions.diff").write_text(diff, encoding="utf-8")
        write_json(pd / "prompts_check.json", out)

    def fresh_recompute(self):
        """Integration check: recompute CACHED requests with the model and compare field by field (exact). A failure
        persists: later invocations re-raise it instead of passing on the file's existence."""
        p = self.rd / "integration_fresh_recompute.json"
        if p.exists():
            prev = json.loads(p.read_text())
            if not prev.get("all_equal") or not prev.get("all_from_cache"):
                raise StageFailed("fresh-recompute check failed earlier (see integration_fresh_recompute.json); "
                                  "start a new run")
            return
        by_uid = {i["window_uid"]: i for i in self.inputs}
        fields = {"label_only_score": ("S", "S_hi", "logit_0", "logit_1", "hi_logit_0", "hi_logit_1", "slot_tensor_sha256",
                                       "model_input_sha256", "question_sha256"),
                  "original_generation": ("raw_response", "label", "parse_status", "slot_tensor_sha256",
                                          "model_input_sha256", "question_sha256", "n_new_tokens")}
        out = []
        for spec in self.cfg["integration_checks"]["fresh_recompute"]:
            inp = by_uid[spec["window_uid"]]
            stage = "baseline" if not spec["mask"] else "fixed"
            cached = self.store.request(stage, inp, stage, "single" if spec["mask"] else "baseline", spec["target"],
                                        spec["family"], tuple(spec["mask"]))
            if self.store.t_model is None:
                self.store.t_model = time.time()
            fresh = self.engine.run(spec["target"], inp, spec["family"], list(spec["mask"]))
            self.store.model_calls += 1
            self.store._heartbeat()
            cmp_ = {f: {"cached": cached.get(f), "fresh": fresh.get(f), "equal": cached.get(f) == fresh.get(f)}
                    for f in fields[spec["target"]]}
            out.append({**spec, "cached_status": cached["cache_status"],
                        "from_cache": cached["cache_status"] in ("cached", "prior_cache"),
                        "all_equal": all(c["equal"] for c in cmp_.values()), "fields": cmp_,
                        "runtime_s": fresh.get("runtime_s")})
        rep = {"at": now(), "checks": out, "all_equal": all(o["all_equal"] for o in out),
               "all_from_cache": all(o["from_cache"] for o in out)}
        write_json(p, rep)
        if not (rep["all_equal"] and rep["all_from_cache"]):
            raise StageFailed("fresh recomputation differs from the cached record, or the compared record was not cached")

    def L0(self, inp):
        lab, _ = self.store.label("baseline", inp, "baseline", "baseline", "none", ())
        return lab

    def st_fixed(self):
        for inp in self.ready_inputs():
            if self.L0(inp) is None:
                continue  # attribution unavailable: Phase 0 only
            for fam in ("blur", "grey"):
                for s in xc.SLOTS:
                    self.store.request("fixed", inp, "fixed", "single", "label_only_score", fam, (s,))
            for fam in ("blur", "grey"):
                for p in xc.all_pairs():
                    self.store.request("fixed", inp, "fixed", "pair", "label_only_score", fam, p)
            for p in xc.all_pairs():
                self.store.label("fixed", inp, "fixed", "pair", "grey", p)
        return self.store.counts.copy()

    def score(self, inp, family, mask, stage="fixed", phase="fixed", purpose="single"):
        if not mask:
            r = self.store.request("baseline", inp, "baseline", "baseline", "label_only_score", "none", ())
        else:
            r = self.store.request(stage, inp, phase, purpose, "label_only_score", family, mask)
        return r if r["cache_status"] in SUCCESS else None

    def fixed_rankings(self, inp, q):
        out = {}
        base = self.score(inp, "none", ())
        for fam, field, name in (("blur", "S_hi", "blur"), ("grey", "S_hi", "grey"), ("blur", "S_native", "blur_native")):
            singles = {}
            for s in xc.SLOTS:
                singles[s] = readout(self.score(inp, fam, (s,)), field)
            out[name] = xc.ranking(q, readout(base, field), singles)
        return out

    def _adaptive_score_record(self, inp, mask):
        if not mask:
            return self.score(inp, "none", ())
        if len(mask) <= 2:  # singles and pairs are the fixed stage's shared forwards
            return self.score(inp, "blur", mask, stage="fixed", phase="fixed", purpose="single" if len(mask) == 1 else "pair")
        return self.score(inp, "blur", mask, stage="adaptive", phase="adaptive", purpose="adaptive_candidate")

    def adaptive(self, inp, q):
        recs = {}

        def score_of(mask):
            r = self._adaptive_score_record(inp, mask)
            recs[tuple(mask)] = r
            return None if r is None else r.get("S_hi")
        traj, complete = xc.adaptive_trajectory(q, score_of, self.cfg["ranking"]["adaptive_max_budget"])
        for st in traj["states"]:  # per-state call accounting from the request log (same whichever stage computed it)
            sr = recs.get(tuple(st["masked"]))
            st["state_sequence_key"] = None if sr is None else sr.get("sequence_key")
            calls = {"new_adaptive": 0, "shared_with_fixed": 0, "cached": 0, "missing": 0}
            for c in st["candidates"]:
                r = recs.get(tuple(c["mask"]))
                c["sequence_key"] = None if r is None else r.get("sequence_key")
                c["cache_status"] = None if r is None else r.get("cache_status")
                if r is None:
                    calls["missing"] += 1
                elif r["cache_status"] in ("cached", "prior_cache") and r.get("phase") == "adaptive":
                    calls["cached"] += 1
                elif r.get("phase") == "adaptive":
                    calls["new_adaptive"] += 1
                else:
                    calls["shared_with_fixed"] += 1
            st["model_calls"] = calls
        traj.update({"family": "blur", "readout": "S_hi", "max_budget": self.cfg["ranking"]["adaptive_max_budget"],
                     "complete": complete})
        return traj

    def st_adaptive(self):
        for inp in self.ready_inputs():
            L0 = self.L0(inp)
            if L0 is not None:
                self.adaptive(inp, xc.q_of(L0))
        return self.store.counts.copy()

    def st_evaluation(self):
        rows = []
        run_adaptive = self.args.ranking_mode in ("adaptive", "both") and "adaptive" in self.cfg["ranking"]["modes"]
        for inp in self.ready_inputs():
            L0 = self.L0(inp)
            if L0 is None:
                continue
            q = xc.q_of(L0)
            fr = self.fixed_rankings(inp, q)
            rec = {"window_uid": inp["window_uid"], "L0": L0, "q": q, "ranking_mode": self.args.ranking_mode,
                   "baseline": {"S_hi": readout(self.score(inp, "none", ()), "S_hi"),
                                "S": readout(self.score(inp, "none", ()), "S_native")},
                   "fixed": fr, "prefixes": {}}
            if fr["blur"]["status"] == "complete":
                rec["prefixes"]["fixed"] = xc.prefixes(fr["blur"]["order"])
                rec["T_w"] = rec["prefixes"]["fixed"]["2"]
                for k, m in rec["prefixes"]["fixed"].items():
                    self.store.request("evaluation", inp, "evaluation", f"fixed_prefix_k{k}", "label_only_score", "grey", m)
                    self.store.label("evaluation", inp, "evaluation", f"fixed_prefix_k{k}", "grey", m)
            if run_adaptive:
                traj = self.adaptive(inp, q)
                rec["adaptive"] = traj
                if traj["order"]:
                    rec["prefixes"]["adaptive"] = xc.prefixes(traj["order"], budgets=range(1, len(traj["order"]) + 1))
                    for k, m in rec["prefixes"]["adaptive"].items():
                        self.store.request("evaluation", inp, "evaluation", f"adaptive_prefix_k{k}", "label_only_score", "grey", m)
                        self.store.label("evaluation", inp, "evaluation", f"adaptive_prefix_k{k}", "grey", m)
            rows.append(rec)
        if not self.args.dry_run:  # merge by window_uid: a subset invocation never truncates other inputs' records
            p = self.rd / "derived" / "rankings.jsonl"
            merged = {d["window_uid"]: d for d in read_jsonl(p)}
            for r in rows:
                merged[r["window_uid"]] = r
            io3b.write_text_atomic(p, "".join(json.dumps(merged[u], ensure_ascii=True) + "\n" for u in sorted(merged)))
        return {**self.store.counts, "rankings_written": len(rows)}

    def completeness(self):
        """Which planned inputs/requests have no terminal record. Used to gate analysis."""
        missing_inputs = sorted(set(planned_uids(self.cfg)) - {
            i["window_uid"] for i in json.loads((self.rd / "inputs.json").read_text())["inputs"]})
        st = Store(self.cfg, self.rd, self.tag, _NoEngine(), 0.0, self.inv)
        open_reqs = []
        rk = {d["window_uid"]: d for d in read_jsonl(self.rd / "derived" / "rankings.jsonl")}
        adaptive = "adaptive" in self.cfg["ranking"]["modes"]
        for inp in json.loads((self.rd / "inputs.json").read_text())["inputs"]:
            if inp["status"] != "ok":
                continue
            u = inp["window_uid"]
            need = [rid_of(u, "label_only_score", "none", ()), rid_of(u, "original_generation", "none", ()),
                    rid_of(u, "nosound_generation", "none", ())]
            base = st.latest.get(need[1])
            L0_known = base is not None and base.get("label") is not None
            if L0_known or any(st.latest.get(rid_of(u, "original_generation", "none", (), r), {}).get("label") is not None
                               for r in (1, 2)):
                need += [rid_of(u, "label_only_score", f, (s,)) for f in ("blur", "grey") for s in xc.SLOTS]
                need += [rid_of(u, "label_only_score", f, p) for f in ("blur", "grey") for p in xc.all_pairs()]
                need += [rid_of(u, "original_generation", "grey", p) for p in xc.all_pairs()]
                d = rk.get(u)
                if d is None:
                    open_reqs.append(f"{u}: no derived ranking record")
                    continue
                for mode in ("fixed",) + (("adaptive",) if adaptive else ()):
                    pref = d.get("prefixes", {}).get(mode)
                    if not pref:
                        if mode == "adaptive" and not d.get("adaptive"):
                            open_reqs.append(f"{u}: adaptive trajectory not computed")
                        continue
                    for m in pref.values():
                        need += [rid_of(u, "label_only_score", "grey", m), rid_of(u, "original_generation", "grey", m)]
            open_reqs += [r for r in need if not st.terminal(r)]
        return missing_inputs, open_reqs

    def _sub(self, script, extra):
        cmd = [sys.executable, "-B", str(CODE / script), "--run-dir", str(self.rd), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "VERA_PROJECT_ROOT": str(ROOT)})
        (self.rd / "logs").mkdir(exist_ok=True)
        (self.rd / "logs" / f"{script}.log").write_text(f"$ {cmd}\n{r.stdout}\n{r.stderr}", encoding="utf-8")
        if r.returncode != 0:
            raise StageFailed(f"{script} exited {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
        return {"cmd": cmd, "script_sha256": sha256_file(CODE / script), "stdout_tail": r.stdout[-400:]}

    def st_analysis(self):
        led = read_jsonl(self.rd / "ledger.jsonl")
        ev = [e for e in led if e["stage"] == "evaluation" and e["event"] == "done"]
        if "adaptive" in self.cfg["ranking"]["modes"] and (not ev or ev[-1]["ranking_mode"] == "fixed"):
            raise StageFailed("the config includes adaptive mode but evaluation last ran with --ranking-mode fixed; "
                              "run --ranking-mode adaptive --resume first")
        missing_inputs, open_reqs = self.completeness()
        detail = {"planned_inputs_missing": len(missing_inputs), "open_requests": len(open_reqs)}
        if (missing_inputs or open_reqs) and not self.args.allow_partial:
            raise StageFailed(f"incomplete run: {len(missing_inputs)} planned inputs never retrieved "
                              f"(e.g. {missing_inputs[:3]}), {len(open_reqs)} planned requests without a terminal "
                              f"record (e.g. {open_reqs[:3]}); finish the run or pass --allow-partial")
        extra = ["--seeds", *map(str, self.args.seeds)] + (["--partial"] if (missing_inputs or open_reqs) else [])
        return {**self._sub("analyze3b.py", extra), **detail}

    def st_validation(self):
        return self._sub("validate3b.py", ["--out", str(self.rd / "validation" / "validation.json"),
                                           "--exp02-run-dir", str(ROOT / "experiments/experiment02/runs/run_002")])

    def st_report(self):
        return self._sub("appendix_table.py", ["--seed", str(self.args.seeds[0])])


class _NoEngine:
    loaded = False
    keymaker = None


def dry_run_plan(cfg, inputs, ranking_mode):
    ub = xc.upper_bound_counts()
    n = len(inputs)
    adaptive = ranking_mode in ("adaptive", "both") and "adaptive" in cfg["ranking"]["modes"]
    per = dict(ub)
    if not adaptive:
        per = {"label_only_score": ub["label_only_score"] - 13, "generation": ub["generation"] - 2,
               "transformed_inputs": ub["transformed_inputs"] - 13}
    t = json.loads((ROOT / "experiments/experiment02/runs/run_002/results/costs.json").read_text())["seconds"]
    gen_s, fwd_s = t["mean_generation_s"], t["mean_score_s"]
    tot = {k: v * n for k, v in per.items()}
    plan = {"inputs": n, "videos": len({i["video_id"] for i in inputs}), "ranking_mode": ranking_mode,
            "per_input_upper_bound": per, "total_upper_bound": tot,
            "core_gpu_hours_upper_bound": round((tot["label_only_score"] * fwd_s + tot["generation"] * gen_s) / 3600, 2),
            "timings_s": {"generation": gen_s, "forward": fwd_s, "source": "experiment02 run_002 results/costs.json"}}
    known = [i for i in inputs if i.get("frame_sha256") and all(i["frame_sha256"])]
    if known and cfg["engine"] == "real":
        import engine3b
        km = engine3b.KeyMaker(cfg, lambda v: engine3b.effective_question_text(cfg, v))
        prior, prior_seq = set(), set()
        for rel in cfg["prior_caches"]:
            for r in read_jsonl(ROOT / rel):
                prior.add(r["key"])
                if r.get("transformed"):
                    prior_seq.add(r["sequence_key"])
        hits = new_req = 0
        new_seq = set()
        for inp in known:
            for _, _, target, fam, mask in xc.static_plan():
                k, s = km.keys(inp, target, fam, mask)
                if k in prior:
                    hits += 1
                else:
                    new_req += 1
                    if mask and s not in prior_seq:
                        new_seq.add(s)
        dyn = 15 if adaptive else 2  # fixed grey prefixes k=3,4 (+ adaptive: 11 blur candidates, 2 grey prefixes)
        plan["cache_projection"] = {
            "inputs_with_known_frames": len(known), "static_requests_in_prior_cache": hits,
            "static_requests_new": new_req, "static_new_transformed_inputs": len(new_seq),
            "dynamic_new_transformed_inputs_upper_bound": dyn * len(known),
            "new_transformed_inputs_upper_bound": len(new_seq) + dyn * len(known),
            "limit": cfg["limits"]["max_new_transformed_inputs"]}
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="analysis/bootstrap seeds (default: config seeds.execute)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ranking-mode", choices=("fixed", "adaptive", "both"), default="both",
                    help="fixed: skip the adaptive stage and adaptive prefixes; adaptive/both: run fixed and adaptive")
    ap.add_argument("--stages", nargs="+", choices=c3.STAGES, default=None)
    ap.add_argument("--run-id", default=None, help="defaults to config run_id")
    ap.add_argument("--videos", nargs="+", default=None, help="restrict this invocation to these video ids")
    ap.add_argument("--shard", default=None, help="k/n: restrict this invocation to videos with sha256(video_id) %% n == k "
                                                  "(invocations on one run directory run one at a time)")
    ap.add_argument("--allow-partial", action="store_true", help="analyse an incomplete run (results marked partial)")
    ap.add_argument("--dry-run-out", default=None)
    args = ap.parse_args()

    cfg = c3.load(args.config)
    args.seeds = args.seeds if args.seeds is not None else cfg["seeds"]["execute"]
    if any(s < 0 for s in args.seeds):
        sys.exit("seeds must be non-negative")
    run_id = args.run_id or cfg["run_id"]
    rd = ROOT / cfg["paths"]["runs_root"] / run_id
    stages = [s for s in c3.STAGES if (args.stages is None or s in args.stages)
              and s not in c3.MODE_STAGES[args.ranking_mode]]
    if args.ranking_mode in ("adaptive", "both") and "adaptive" not in cfg["ranking"]["modes"]:
        sys.exit("--ranking-mode adaptive needs 'adaptive' in config ranking.modes")
    inv = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    if args.dry_run:
        runner = Runner(cfg, args, rd, inv)
        runner.stages = stages
        try:
            runner.inputs = build_inputs(cfg, args)
        except StageFailed as e:
            print(json.dumps({"dry_run": True, "error": str(e)}))
            return 1
        rep = {"dry_run": True, "config": str(args.config), "config_sha256": c3.canonical_sha(cfg), "run_dir": str(rd),
               "run_dir_exists": rd.exists(), "stages": stages, "seeds": args.seeds,
               "plan": dry_run_plan(cfg, runner.inputs, args.ranking_mode), "checks": {}}
        for st in ("preflight", "errata"):
            if st in stages:
                try:
                    rep["checks"][st] = {"ok": True, "detail": getattr(runner, "st_" + st)()}
                except Exception as e:  # report, do not raise: a dry run lists what would fail
                    rep["checks"][st] = {"ok": False, "detail": repr(e)}
        txt = json.dumps(rep, indent=1, default=str)
        if args.dry_run_out:
            Path(args.dry_run_out).write_text(txt + "\n")
        print(txt)
        return 0

    if rd.exists() and not args.resume:
        sys.exit(f"{rd} exists; pass --resume to continue it, or use a new --run-id")
    (rd / "frozen").mkdir(parents=True, exist_ok=True)
    lock = open(rd / "run.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another invocation holds this run's lock (invocations on one run directory run one at a time)")
    frozen_cfg, frozen_code = rd / "frozen" / "config.json", rd / "frozen" / "code_sha256.json"
    if frozen_cfg.exists():
        old = json.loads(frozen_cfg.read_text())
        if c3.canonical_sha({k: v for k, v in old.items() if k != "seeds"}) != c3.canonical_sha(
                {k: v for k, v in cfg.items() if k != "seeds"}):
            sys.exit("config differs from this run's frozen config; start a new run (config changes need a new run id)")
        old_code, now_code = json.loads(frozen_code.read_text()), code_hashes()
        changed = sorted(f for f in set(old_code) | set(now_code) if old_code.get(f) != now_code.get(f))
        if changed:
            sys.exit(f"code changed since this run was frozen: {changed}; start a new run id (list this run's "
                     f"cache/forward_cache*.jsonl in prior_caches to reuse its model outputs)")
    else:
        write_json(frozen_cfg, cfg)
        write_json(frozen_code, code_hashes())

    runner = Runner(cfg, args, rd, inv)
    runner.stages = stages
    runner.ledger("invocation", "invocation_start", {"stages": stages, "argv": sys.argv})
    done_ok = {e["stage"] for e in read_jsonl(rd / "ledger.jsonl") if e["event"] == "done"}
    failed = None
    gpu_before = runner.prior_gpu_s()
    try:
        runner.inputs = build_inputs(cfg, args)
        for st in stages:
            deps = [d for d in c3.DEPENDS[st] if d not in c3.MODE_STAGES[args.ranking_mode]]
            missing = [d for d in deps if d not in done_ok]
            if failed or missing:
                runner.ledger(st, "skipped", {"reason": f"dependency failed: {failed}" if failed else f"not done: {missing}"})
                failed = failed or f"{st} (unmet {missing})"
                continue
            runner.ledger(st, "start")
            try:
                if st in GPU_STAGES and runner.store is None:
                    runner.gate_retrieved()
                    runner.engine = runner.make_engine()
                    runner.store = Store(cfg, rd, runner.tag, runner.engine, gpu_before, inv)
                detail = getattr(runner, "st_" + st)()
                runner.ledger(st, "done", detail)
                done_ok.add(st)
            except Exception as e:
                runner.ledger(st, "failed", {"error": repr(e), "budget": isinstance(e, Budget),
                                             "traceback": traceback.format_exc()[-1500:]})
                failed = st
                print(f"stage {st} failed: {e!r}", file=sys.stderr)
    except Exception as e:
        failed = failed or f"setup ({e!r})"
        print(f"failed before stages: {e!r}", file=sys.stderr)
    finally:
        gs = runner.store.gpu_s_this_invocation() if runner.store else 0.0
        if runner.store and runner.store.t_model:
            runner.store._heartbeat()
        runner.ledger("invocation", "invocation_end", {
            "gpu_s_this_invocation": round(gs, 1), "gpu_s_all_invocations": round(gpu_before + gs, 1),
            "counts": runner.store.counts if runner.store else None,
            "new_transformed_inputs_total": len(runner.store.new_seq) if runner.store else None,
            "engine": runner.engine.summary() if runner.engine is not None else None, "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

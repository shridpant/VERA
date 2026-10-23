"""experiment03B configuration: loading, schema validation and the stage DAG.

The JSON schema document is ../config.schema.json (for readers and editors). The environment has no `jsonschema`
package, so this module enforces the same rules by hand, plus the cross-field rules a JSON schema cannot express.
Every violation is collected and reported; nothing is silently defaulted.
"""
import copy
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "exp03b-config-1"
STAGES = ("preflight", "errata", "manifest", "retrieval", "baseline", "fixed", "adaptive", "evaluation",
          "analysis", "validation", "report")
DEPENDS = {  # stage -> stages that must be done first (in the same run)
    "preflight": [], "errata": ["preflight"], "manifest": ["errata"], "retrieval": ["manifest"],
    "baseline": ["retrieval"], "fixed": ["baseline"], "adaptive": ["fixed"], "evaluation": ["fixed"],
    "analysis": ["evaluation"], "validation": ["analysis"], "report": ["validation"],
}
MODE_STAGES = {"fixed": {"adaptive"}, "adaptive": set(), "both": set()}  # stages skipped per --ranking-mode
SUPPORTED = {
    "scoring.contract": {"label_only_first_token_v1"},
    "scoring.ranking_readout": {"S_hi"},
    "ranking.construction_family": {"blur"},
    "ranking.evaluation_family": {"grey"},
    "ranking.aggregation": {"singleton"},
    "ranking.method": {"temporal_occlusion"},
    "population.kind": {"manifest", "cases"},
    "engine": {"real", "mock"},
}


class ConfigError(ValueError):
    pass


def _get(d, dotted):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return KeyError
        cur = cur[k]
    return cur


def check_dag(depends=DEPENDS, stages=STAGES):
    """Every dependency is a known stage, no cycles, and STAGES is a valid topological order."""
    errs = []
    pos = {s: i for i, s in enumerate(stages)}
    for s, deps in depends.items():
        if s not in pos:
            errs.append(f"unknown stage {s}")
        for d in deps:
            if d not in pos:
                errs.append(f"{s} depends on unknown stage {d}")
            elif pos[d] >= pos.get(s, -1):
                errs.append(f"{s} depends on {d}, which is not earlier in the stage order")
    seen, stack = set(), set()

    def visit(s):
        if s in stack:
            errs.append(f"cycle through {s}")
            return
        if s in seen:
            return
        stack.add(s)
        for d in depends.get(s, []):
            visit(d)
        stack.discard(s)
        seen.add(s)
    for s in depends:
        visit(s)
    return errs


REQUIRED = {  # dotted path -> type(s)
    "schema_version": str, "experiment_id": str, "protocol_version": str, "run_id": str, "engine": str,
    "seeds.planned": list, "seeds.execute": list,
    "source.repo": str, "source.base_commit": str, "source.checkout": str, "source.expected_head": str,
    "model.repo": str, "model.revision": str, "model.local_dir": str, "model.dtype": str, "model.use_flash_attn": bool,
    "generation.num_beams": int, "generation.max_new_tokens": int, "generation.do_sample": bool,
    "prompt.learner_instruct_file": str, "prompt.guiding_questions": list,
    "scoring.contract": str, "scoring.ranking_readout": str,
    "replacements.preprocessed_resolution": int, "replacements.blur_radius_fraction": (int, float),
    "replacements.grey_rgb": list,
    "ranking.method": str, "ranking.aggregation": str, "ranking.construction_family": str,
    "ranking.evaluation_family": str, "ranking.modes": list, "ranking.budgets": list, "ranking.adaptive_max_budget": int,
    "population.kind": str, "population.development_videos": list,
    "analysis.bootstrap_B": int, "analysis.alpha": (int, float), "analysis.stability_seed_offsets": list,
    "retries.error_retries": int, "retries.parse_failure_repeats": int,
    "limits.max_new_transformed_inputs": (int, type(None)), "limits.max_gpu_hours": (int, float, type(None)),
    "limits.max_scratch_mib": (int, float, type(None)), "limits.max_model_calls": (int, type(None)),
    "retrieval.mode": str, "retrieval.frames_root": str, "retrieval.scratch_dir": str,
    "prior_caches": list, "paths.runs_root": str, "gpu.min_free_mib": int,
}


def validate(cfg):
    errs = []
    for path, typ in REQUIRED.items():
        v = _get(cfg, path)
        if v is KeyError:
            errs.append(f"missing {path}")
        elif not isinstance(v, typ) or (typ is int and isinstance(v, bool)):
            errs.append(f"{path} has type {type(v).__name__}, expected {typ}")
    if errs:
        raise ConfigError("; ".join(errs))
    if cfg["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {cfg['schema_version']!r} != {SCHEMA_VERSION!r}")
    for path, allowed in SUPPORTED.items():
        v = _get(cfg, path)
        if v not in allowed:
            errs.append(f"{path} = {v!r} is not supported (allowed: {sorted(allowed)})")
    modes = cfg["ranking"]["modes"]
    if not modes or any(m not in ("fixed", "adaptive") for m in modes) or "fixed" not in modes:
        errs.append(f"ranking.modes must contain 'fixed' and only 'fixed'/'adaptive', got {modes}")
    b = cfg["ranking"]["budgets"]
    if b != [1, 2, 3, 4]:
        errs.append(f"ranking.budgets must be [1, 2, 3, 4] under protocol v1.0, got {b}")
    if cfg["ranking"]["adaptive_max_budget"] not in (1, 2, 3, 4):
        errs.append("ranking.adaptive_max_budget must be 1..4")
    if "adaptive" in modes and cfg["ranking"]["adaptive_max_budget"] < 2:
        errs.append("an adaptive mode with max budget < 2 never recomputes conditional effects")
    if cfg["ranking"]["construction_family"] == cfg["ranking"]["evaluation_family"]:
        errs.append("construction and evaluation families must differ (cross-family evaluation)")
    s = cfg["seeds"]
    if not all(isinstance(x, int) and x >= 0 for x in s["planned"] + s["execute"]):
        errs.append("seeds must be non-negative integers")
    if cfg["analysis"]["bootstrap_B"] < 1000:
        errs.append("analysis.bootstrap_B must be >= 1000")
    if not 0 < cfg["analysis"]["alpha"] < 1:
        errs.append("analysis.alpha must be in (0, 1)")
    if len(cfg["prompt"]["guiding_questions"]) != 5:
        errs.append("prompt.guiding_questions must hold the 5 released questions")
    if cfg["replacements"]["grey_rgb"] != [127, 127, 127] or cfg["replacements"]["preprocessed_resolution"] != 448 \
            or cfg["replacements"]["blur_radius_fraction"] != 0.04:
        errs.append("replacements differ from protocol v1.0 (blur 0.04 x 448, grey RGB 127 at 448)")
    for k in ("error_retries", "parse_failure_repeats"):
        if not 0 <= cfg["retries"][k] <= 5:
            errs.append(f"retries.{k} must be in 0..5")
    lim = cfg["limits"]
    for k, v in lim.items():
        if v is not None and v <= 0:
            errs.append(f"limits.{k} must be positive or null")
    kind = cfg["population"]["kind"]
    if kind == "manifest" and not isinstance(cfg["population"].get("manifest"), str):
        errs.append("population.kind 'manifest' needs population.manifest (path)")
    if kind == "cases":
        cases = cfg["population"].get("cases")
        if not isinstance(cases, list) or not cases:
            errs.append("population.kind 'cases' needs a non-empty population.cases list")
        else:
            for c in cases:
                for f in ("window_uid", "video_id", "category", "anchor", "frame_indices", "frames_dir", "video_sha256",
                          "strata", "development"):
                    if f not in c:
                        errs.append(f"case {c.get('window_uid')} lacks {f}")
                if len(c.get("frame_indices", [])) != 8:
                    errs.append(f"case {c.get('window_uid')} needs 8 frame indices")
    if cfg["retrieval"]["mode"] not in ("existing_only", "fetch", "none"):
        errs.append("retrieval.mode must be 'existing_only', 'fetch' or 'none'")
    if cfg["retrieval"]["mode"] == "none" and cfg["engine"] != "mock":
        errs.append("retrieval.mode 'none' (no frames) is only valid with the mock engine")
    if cfg["engine"] == "mock" and not isinstance(cfg.get("mock_spec"), str):
        errs.append("engine 'mock' needs mock_spec (path to the analytic scorer spec)")
    if cfg["retrieval"]["mode"] == "fetch" and not isinstance(cfg["retrieval"].get("archives"), dict):
        errs.append("retrieval.mode 'fetch' needs retrieval.archives {zip name: official URL}")
    if cfg["analysis"]["stability_seed_offsets"] != [1, 2, 3, 4]:
        errs.append("analysis.stability_seed_offsets must be [1, 2, 3, 4] (PROTOCOL §7.1: seeds seed+1 ... seed+4)")
    if cfg["retrieval"]["mode"] == "fetch" and not isinstance(cfg["retrieval"].get("fidelity_check"), dict):
        errs.append("retrieval.mode 'fetch' needs retrieval.fidelity_check (sparse extraction must reproduce VERA's "
                    "full extraction before any frames are extracted)")
    if not isinstance(cfg["retrieval"].get("prefer_local", True), bool):
        errs.append("retrieval.prefer_local must be a boolean")
    if cfg["analysis"].get("secondary_scope", "confirmation") not in ("confirmation", "all"):
        errs.append("analysis.secondary_scope must be 'confirmation' or 'all'")
    errs += check_dag()
    if errs:
        raise ConfigError("; ".join(errs))
    return cfg


def load(path, overrides=None):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return validate(cfg)


def canonical_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exp02_compat(cfg):
    """A config dict in experiment02's shape, so experiment02's engine/attr_lib compute identical prompts, specs and
    cache keys. 3B's `grey` family is experiment02's `constant`."""
    return {
        "seed": 0,  # greedy decoding; the torch seed has no effect on outputs, kept identical to experiment02
        "source": dict(cfg["source"]), "model": dict(cfg["model"]), "generation": dict(cfg["generation"]),
        "prompt": dict(cfg["prompt"]), "frames": {"cache_root": cfg["retrieval"]["frames_root"]},
        "stage_b": {"preprocessed_resolution": cfg["replacements"]["preprocessed_resolution"],
                    "blur_radius_fraction": cfg["replacements"]["blur_radius_fraction"],
                    "constant_rgb": list(cfg["replacements"]["grey_rgb"])},
    }


FAMILY_TO_EXP02 = {"none": "none", "blur": "blur", "grey": "constant"}

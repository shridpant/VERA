#!/usr/bin/env python3
"""experiment03B CPU test suite. No model, no download. Run with the project interpreter:

    setup/environment/env/bin/python -B experiments/experiment03B/code/tests/run_tests.py [--only NAME ...] [--out FILE]

Each test maps to one of the brief's required behaviours (VERA_Fixed_Adaptive_Workflow_Handoff.md, "Mandatory
validation evidence" 1-11). Prints PASS/FAIL per test and writes a JSON summary; exit 1 on any failure.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
CODE = TESTS_DIR.parent
ROOT = CODE.parents[2]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(TESTS_DIR))
import config3b as c3  # noqa: E402
import synthetic as syn  # noqa: E402
import xb_core as xc  # noqa: E402

PY = sys.executable
RESULTS = []


def test(maps_to):
    def deco(fn):
        fn.maps_to = maps_to
        RESULTS.append(fn)
        return fn
    return deco


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_wf(cfg_path, *args, env=None):
    cmd = [PY, "-B", str(CODE / "run_workflow.py"), "--config", str(cfg_path), *args]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return r


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]


# ---------------------------------------------------------------- 1, 2, 7: core arithmetic

@test("1 known additive scorer")
def t_additive():
    w = [1, 2, 3, 4, 5, 6, 7, 8]
    f = lambda m: -20 + sum(wi for i, wi in enumerate(w) if i not in m)  # noqa: E731
    r = xc.ranking(1, f(()), {s: f((s,)) for s in xc.SLOTS})
    check([r["effects"][str(s)] for s in xc.SLOTS] == [float(x) for x in w] or [r["effects"][str(s)] for s in xc.SLOTS] == w,
          f"effects {r['effects']}")
    check(r["order"] == [7, 6, 5, 4, 3, 2, 1, 0], f"order {r['order']}")
    for i, j in xc.all_pairs():
        J = xc.effect(1, f(()), f((i, j))) - xc.effect(1, f(()), f((i,))) - xc.effect(1, f(()), f((j,)))
        check(J == 0, f"additive J({i},{j}) = {J}")
    rn = xc.ranking(-1, f(()), {s: f((s,)) for s in xc.SLOTS})  # normal target: support for class 0
    check(rn["order"] == [0, 1, 2, 3, 4, 5, 6, 7] and rn["effects"]["7"] == -8, f"normal-target order {rn['order']}")


@test("2 interacting scorer: fixed vs adaptive differ, analytically")
def t_interaction():
    def F(m):
        a, b, c = (0 if s in m else 1 for s in (0, 1, 2))
        return 10 * a + 5 * b + c + 5 * a * c
    r = xc.ranking(1, F(()), {s: F((s,)) for s in xc.SLOTS})
    check({s: r["effects"][str(s)] for s in (0, 1, 2)} == {0: 15, 1: 5, 2: 6}, f"singletons {r['effects']}")
    check(r["order"][:3] == [0, 2, 1], f"fixed order {r['order']}")
    traj, ok = xc.adaptive_trajectory(1, F, 4)
    check(ok and traj["order"][:3] == [0, 1, 2], f"adaptive order {traj['order']}")
    st2 = {c["slot"]: c["effect"] for c in traj["states"][1]["candidates"]}
    check(st2[1] == 5 and st2[2] == 1, f"conditional effects after masking 0: {st2}")
    check(traj["states"][1]["state_score"] == F((0,)), "state score after step 1")
    check(xc.prefixes(r["order"])["2"] == [0, 2] and xc.prefixes(traj["order"])["2"] == [0, 1], "prefixes")


@test("4 adaptive audit: never reselects, masked set grows by one, q locked")
def t_adaptive_audit():
    calls = []

    def F(m):
        calls.append(m)
        return 3.0 - len(m) + (0.5 if 4 in m else 0)
    traj, ok = xc.adaptive_trajectory(-1, F, 4)  # q = -1: support for the normal class
    check(ok and len(set(traj["order"])) == 4, f"order {traj['order']}")
    for k, st in enumerate(traj["states"]):
        check(st["masked"] == sorted(traj["order"][:k]), f"state {k} masked {st['masked']}")
        check(all(c["slot"] not in st["masked"] for c in st["candidates"]), "candidate already masked")
        check(len(st["candidates"]) == 8 - k, "every remaining slot is a candidate")
        check(all(c["effect"] == -1 * (st["state_score"] - c["score"]) for c in st["candidates"]), "q locked to -1")
    # q = -1: effect = F(R u {j}) - F(R) = -1 for every slot except 4 (-0.5), so 4 is selected first and the remaining
    # exact ties go to the lower slot
    check(traj["order"] == [4, 0, 1, 2], f"selection by largest effect, ties to the lower slot: {traj['order']}")


@test("7 ties, missing values and NaN")
def t_ties_missing():
    r = xc.ranking(1, 3.0, {s: 1.0 for s in xc.SLOTS})
    check(r["all_tied"] and r["tie_spans_cut"]["2"] and r["order"] == list(range(8)), "all tied")
    r = xc.ranking(1, 3.0, {**{s: 1.0 for s in xc.SLOTS}, 5: None})
    check(r["status"] == "incomplete" and r["missing"] == [5], "missing single")
    r = xc.ranking(1, 3.0, {**{s: 1.0 for s in xc.SLOTS}, 5: float("nan")})
    check(r["status"] == "incomplete", "NaN invalidates the ranking")
    labels = {p: 1 for p in xc.all_pairs()}
    labels[(0, 1)] = None
    d, ft, fr, n = xc.d_w((0, 1), labels)
    check(d is None and ft is None and n == 27, "missing selected label -> d_w missing")
    lo, hi = xc.d_w_bounds((0, 1), labels)
    check(lo == 0 - 1 / 28 and hi == 1 - 0, f"bounds {lo} {hi}")


@test("8 numerical interpretation: sub-step FP32 change crosses a BF16 boundary; E1 counts")
def t_bf16():
    import struct

    def bf16(x):  # round-to-nearest-even to bfloat16
        b = struct.unpack(">I", struct.pack(">f", x))[0]
        b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
        return struct.unpack(">f", struct.pack(">I", b))[0]
    hi1a, hi1b, hi0 = 36.12, 36.13, 33.30  # S_hi changes by 0.01 (far below a 0.25 step)
    Sa, Sb = bf16(hi1a) - bf16(hi0), bf16(hi1b) - bf16(hi0)
    check(abs((hi1b - hi0) - (hi1a - hi0)) < 0.25 and Sb - Sa == 0.25, f"no boundary crossing: {Sa} {Sb}")
    p = subprocess.run([PY, "-B", str(ROOT / "experiments/experiment02/errata/precision_crossings.py"), "--effects",
                        str(ROOT / "experiments/experiment02/runs/run_002/results/temporal_effects.csv"), "--out",
                        str(Path(tempfile.mkdtemp()) / "e1.json")], capture_output=True, text=True)
    check(p.returncode == 0, p.stderr)
    tot = json.loads(p.stdout.split("\n")[0])
    check(tot["sub_step"] == 109 and tot["sub_step_but_bf16_moved"] == 36 and tot["opposite_sign"] == 4, f"E1 {tot}")


@test("aggregation: hand-worked video-cluster fixture and reproducible bootstrap")
def t_aggregation():
    vals = {"a@1": 0.5, "a@2": 0.0, "b@1": 1.0, "c@1": None, "c@2": -0.5}
    vid = {k: k.split("@")[0] for k in vals}
    vm = xc.video_means(vals, vid)
    check(vm == {"a": 0.25, "b": 1.0, "c": -0.5}, f"video means {vm}")
    p1, ci1 = xc.cluster_bootstrap(vm, 7, 2000)
    p2, ci2 = xc.cluster_bootstrap(vm, 7, 2000)
    check(abs(p1 - 0.25) < 1e-12 and ci1 == ci2, "point or reproducibility")
    check(ci1[0] >= -0.5 - 1e-12 and ci1[1] <= 1.0 + 1e-12, "interval within the support of video means")
    check(xc.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0 and xc.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0, "tau_b")
    tb = xc.kendall_tau_b([1, 1, 2, 3], [1, 2, 2, 3])
    check(abs(tb - 4 / math.sqrt(5 * 5)) < 1e-12, f"tau_b with ties {tb}")


@test("5 time adjacency uses original slot positions")
def t_adjacency():
    adj = [p for p in xc.all_pairs() if p[1] - p[0] == 1]
    check(len(adj) == 7 and (2, 4) not in adj, "adjacency by original slots")


@test("9 cache keys: identity-based, mask/family/repeat-sensitive, prompt-independent sequence keys")
def t_keys():
    import engine3b
    cfg = c3.load(CODE.parent / "config.integration.json")
    km = engine3b.KeyMaker(cfg, lambda v: engine3b.effective_question_text(cfg, v))
    inp = cfg["population"]["cases"][0]
    k1, s1 = km.keys(inp, "label_only_score", "blur", (3,))
    k2, s2 = km.keys(inp, "original_generation", "blur", (3,))
    k3, s3 = km.keys(inp, "label_only_score", "grey", (3,))
    k4, _ = km.keys(inp, "label_only_score", "blur", (3,), repeat=1)
    k5, s5 = km.keys(inp, "label_only_score", "blur", (3, 4))
    check(s1 == s2 and k1 != k2, "sequence key prompt-independent; request key target-specific")
    check(len({k1, k3, k4, k5}) == 4 and len({s1, s3, s5}) == 3, "keys distinguish family, mask and repeat")
    prior = {r["key"] for r in read_jsonl(ROOT / "experiments/experiment02/runs/run_002/results/cache_records.jsonl")}
    check(k1 in prior, "3B key for R-accident blur slot 3 equals experiment02's (cache sharing)")
    kg, _ = km.keys(inp, "label_only_score", "grey", (3, 4))
    check(kg in prior, "3B grey = experiment02 constant family key")


@test("10 config/DAG validation fails clearly")
def t_config():
    check(c3.check_dag() == [], "shipped DAG valid")
    bad = dict(c3.DEPENDS)
    bad["preflight"] = ["report"]
    check(any("cycle" in e or "not earlier" in e for e in c3.check_dag(bad)), "cycle detected")
    base = json.loads((CODE.parent / "config.integration.json").read_text())
    for path, val, frag in (("scoring.ranking_readout", "native_bf16", "not supported"),
                            ("ranking.budgets", [2, 4], "budgets"),
                            ("ranking.evaluation_family", "blur", "not supported"),
                            ("limits.max_gpu_hours", -1, "positive"),
                            ("retries.parse_failure_repeats", 9, "0..5")):
        cfg = json.loads(json.dumps(base))
        cur = cfg
        for k in path.split(".")[:-1]:
            cur = cur[k]
        cur[path.split(".")[-1]] = val
        try:
            c3.validate(cfg)
            raise AssertionError(f"{path}={val} accepted")
        except c3.ConfigError as e:
            check(frag in str(e), f"{path}: {e}")


class SynthRun:
    def __init__(self, name, **kw):
        self.tmp = Path(tempfile.mkdtemp(prefix=f"exp03b_{name}_"))
        self.cfg = syn.build(self.tmp / "fx", self.tmp / "runs", run_id=name, **kw)
        self.rd = self.tmp / "runs" / name


@test("10 dry-run has no side effects")
def t_dry_run():
    s = SynthRun("dry")
    r = run_wf(s.cfg, "--seeds", "0", "--dry-run")
    check(r.returncode == 0, r.stderr[-800:])
    rep = json.loads(r.stdout)
    check(not s.rd.exists(), "dry-run created the run directory")
    check(rep["plan"]["per_input_upper_bound"] == {"label_only_score": 88, "generation": 35, "transformed_inputs": 87},
          rep["plan"])
    check(rep["checks"]["errata"]["ok"], rep["checks"])


@test("3/6/10/11 end-to-end synthetic run: fixed masks, cross-family isolation, resume, independent validation")
def t_end_to_end():
    s = SynthRun("e2e")
    r = run_wf(s.cfg, "--seeds", "0", "--ranking-mode", "fixed", "--stages", "preflight", "errata", "manifest",
               "retrieval", "baseline", "fixed", env={"MOCK_CRASH_AFTER": "200"})
    check(r.returncode != 0, "the crash fixture should interrupt the run")
    led = read_jsonl(s.rd / "ledger.jsonl")
    check(not any(e["stage"] == "fixed" and e["event"] == "done" for e in led), "no false 'done' after a crash")
    n_before = sum(1 for p in (s.rd / "requests").glob("*.jsonl") for r_ in read_jsonl(p) if r_["cache_status"] == "fresh")
    r = run_wf(s.cfg, "--seeds", "0", "--ranking-mode", "fixed", "--resume", "--stages", "preflight", "errata", "manifest",
               "retrieval", "baseline", "fixed", "evaluation")
    check(r.returncode == 0, r.stderr[-1500:])
    recs = [r_ for p in (s.rd / "requests").glob("*.jsonl") for r_ in read_jsonl(p)]
    ids = [r_["request_id"] for r_ in recs if r_["cache_status"] == "fresh"]
    check(len(ids) == len(set(ids)), "a request was computed twice across the resume")
    check(not any(r_["phase"] == "adaptive" for r_ in recs), "fixed mode issued adaptive conditional requests")
    rk = {d["window_uid"]: d for d in read_jsonl(s.rd / "derived" / "rankings.jsonl")}
    A = rk["SynthArson_001@160"]
    check(A["fixed"]["blur"]["order"] == syn.EXPECTED["fixed_order_A"], A["fixed"]["blur"]["order"])
    check(A["prefixes"]["fixed"]["3"] == [0, 1, 2] and A["T_w"] == [0, 2], A["prefixes"])
    ev = [r_ for r_ in recs if r_["window_uid"] == "SynthArson_001@160" and r_["phase"] == "evaluation"]
    check({tuple(r_["mask"]) for r_ in ev} == {(0,), (0, 2), (0, 1, 2), (0, 1, 2, 3)}, "fixed prefix masks from the frozen order")
    # adaptive on top, then analysis/validation/report
    r = run_wf(s.cfg, "--seeds", "0", "1", "--ranking-mode", "adaptive", "--resume")
    check(r.returncode == 0, r.stderr[-2500:] + r.stdout[-500:])
    rk = {d["window_uid"]: d for d in read_jsonl(s.rd / "derived" / "rankings.jsonl")}
    check(rk["SynthArson_001@160"]["adaptive"]["order"] == syn.EXPECTED["adaptive_order_A"], rk["SynthArson_001@160"]["adaptive"])
    check(rk["SynthArson_001@160"]["fixed"]["blur"]["order"] == syn.EXPECTED["fixed_order_A"], "fixed order changed by adaptive")
    pr = json.loads((s.rd / "results" / "primary.json").read_text())
    check(abs(pr["estimate_pp"] - syn.EXPECTED["primary_pp"]) < 1e-9, f"primary {pr['estimate_pp']}")
    for uid, d in syn.EXPECTED["d_w"].items():
        if uid in pr["d_w"]:
            check((pr["d_w"][uid] is None and d is None) or abs(pr["d_w"][uid] - d) < 1e-12, f"d_w {uid}: {pr['d_w'][uid]} vs {d}")
    check("RoadAccidents127_x264@2224" not in pr["d_w"], "development input in the confirmation set")
    for v, m in syn.EXPECTED["video_means"].items():
        check(abs(pr["video_means_pp"][v] - 100 * m) < 1e-9, f"video mean {v}")
    check((s.rd / "results" / "seeds" / "primary_seed1.json").exists(), "additional seed output")
    lo, hi = syn.EXPECTED["bounds_SynthBurglary_002@600"]
    met = {r_["window_uid"]: r_ for r_ in __import__("csv").DictReader(open(s.rd / "results" / "inputs_metrics.csv"))}
    check(abs(float(met["SynthBurglary_002@592"]["d_w_lo"]) - lo) < 1e-12 and abs(float(met["SynthBurglary_002@592"]["d_w_hi"]) - hi) < 1e-12,
          "worst-case bounds")
    check(met["SynthRobbery_005@160"]["attribution"] == "unavailable_L0_missing", "missing baseline label kept visible")
    val = json.loads((s.rd / "validation" / "validation.json").read_text())
    check(val["passed"], [c for c in val["checks"] if c.get("passed") is False][:3])
    check((s.rd / "results" / "APPENDIX_RESULTS.md").read_text().startswith("> **SYNTHETIC RUN"), "synthetic banner")
    # cross-family isolation: different grey outputs, identical blur rankings and adaptive choices
    s2 = SynthRun("e2e_grey", grey_overrides={"SynthArson_001@160": {"w": [0, 0, 0, 0, 9, 9, 0, 0], "offset": -1}})
    r = run_wf(s2.cfg, "--seeds", "0")
    check(r.returncode == 0, r.stderr[-1500:])
    rk2 = {d["window_uid"]: d for d in read_jsonl(s2.rd / "derived" / "rankings.jsonl")}
    a1, a2 = rk["SynthArson_001@160"], rk2["SynthArson_001@160"]
    check(a1["fixed"]["blur"] == a2["fixed"]["blur"] and a1["adaptive"]["order"] == a2["adaptive"]["order"]
          and a1["T_w"] == a2["T_w"], "grey outputs changed a blur ranking")
    check(a1["fixed"]["grey"]["order"] != a2["fixed"]["grey"]["order"], "fixture did not change grey")
    return {"run_dir": str(s.rd)}


@test("10 immutability: config or code change refuses to resume; manifest hash enforced")
def t_immutable():
    s = SynthRun("immut")
    r = run_wf(s.cfg, "--seeds", "0", "--stages", "preflight", "errata", "manifest")
    check(r.returncode == 0, r.stderr[-800:])
    cfg = json.loads(s.cfg.read_text())
    cfg["analysis"]["bootstrap_B"] = 3000
    s.cfg.write_text(json.dumps(cfg))
    r = run_wf(s.cfg, "--seeds", "0", "--resume", "--stages", "manifest")
    check(r.returncode != 0 and "frozen config" in (r.stderr + r.stdout), "changed config accepted on resume")
    full = json.loads((CODE.parent / "config.json").read_text())
    full["population"]["manifest_sha256"] = "0" * 64
    full["paths"]["runs_root"] = str(s.tmp / "runs2")
    p = s.tmp / "full_bad.json"
    p.write_text(json.dumps(full))
    r = run_wf(p, "--seeds", "0", "--dry-run")
    check(r.returncode != 0 and "immutable" in (r.stderr + r.stdout), "manifest hash not enforced")


@test("3 masks are rebuilt from original pixels (real pixel path, CPU)")
def t_pixels():
    fdir = ROOT / "shared/frames/ucf_crime/RoadAccidents127_x264"
    if not fdir.exists():
        return {"skipped": "frame cache not present on this machine"}
    sys.path.insert(0, str(ROOT / "experiments/experiment02/code"))
    sys.path.insert(0, str(ROOT / "setup/source/VERA"))
    import attr_lib as al
    import generate_initial_score as gis
    cfg = c3.exp02_compat(c3.load(CODE.parent / "config.integration.json"))
    ds = gis.Video_Instruct_Dataset_Inference(vis_root=str(fdir.parent) + "/", ann_root=str(ROOT / "experiments/experiment02/runs/run_002/results/eval_manifest_cases.json"))
    fi = [2090, 2127, 2165, 2202, 2240, 2277, 2315, 2352]
    blur = al.family_spec(cfg, "blur")
    _, h0 = al.slot_tensors(gis, ds.transform, fdir, fi, (), {"family": "none"})
    _, h3 = al.slot_tensors(gis, ds.transform, fdir, fi, (3,), blur)
    _, h34 = al.slot_tensors(gis, ds.transform, fdir, fi, (3, 4), blur)
    _, h4 = al.slot_tensors(gis, ds.transform, fdir, fi, (4,), blur)
    check(h34[3] == h3[3] and h34[4] == h4[4], "a slot's replaced tensor depends on other masked slots (accumulation)")
    check(all(h34[i] == h0[i] for i in range(8) if i not in (3, 4)), "unmasked slots changed")


@test("10 a reached limit stops the run (failed stage, no analysis), never a silent truncation")
def t_budget_stop():
    s = SynthRun("budget", limits={"max_new_transformed_inputs": 10, "max_gpu_hours": None, "max_scratch_mib": None,
                                   "max_model_calls": None})
    r = run_wf(s.cfg, "--seeds", "0")
    check(r.returncode == 1, f"exit {r.returncode}")
    led = read_jsonl(s.rd / "ledger.jsonl")
    fails = [e for e in led if e["event"] == "failed"]
    check(fails and fails[0]["stage"] == "fixed" and fails[0]["detail"]["budget"], fails[:1])
    check(not any(e["stage"] == "analysis" and e["event"] == "done" for e in led), "analysis ran after a budget stop")
    skipped = [r_ for p in (s.rd / "requests").glob("*.jsonl") for r_ in read_jsonl(p) if r_["cache_status"] == "skipped_budget"]
    check(len(skipped) == 1, f"{len(skipped)} skipped lines (the stop should be immediate)")


@test("6/7 transient errors are retried inside the request and logged")
def t_retry():
    s = SynthRun("retry")
    r = run_wf(s.cfg, "--seeds", "0", "--stages", "preflight", "errata", "manifest", "retrieval", "baseline", "fixed",
               env={"MOCK_FAIL_ONCE": "SynthArson_001@160|original_generation|grey|0,2"})
    check(r.returncode == 0, r.stderr[-800:])
    recs = [r_ for p in (s.rd / "requests").glob("*.jsonl") for r_ in read_jsonl(p)
            if r_["request_id"] == "SynthArson_001@160|original_generation|grey|0,2|r0"]
    check([x["cache_status"] for x in recs] == ["error", "fresh"] and recs[-1]["attempt"] == 2, recs)


@test("10 torn JSONL tails are repaired and kept for audit")
def t_torn():
    import io3b
    d = Path(tempfile.mkdtemp())
    p = d / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')
    check(io3b.read_jsonl(p) == [{"a": 1}, {"a": 2}], "reader must ignore the torn tail")
    io3b.append_jsonl(p, {"a": 4})
    check(io3b.read_jsonl(p) == [{"a": 1}, {"a": 2}, {"a": 4}], io3b.read_jsonl(p))
    check((d / "x.jsonl.torn.0").read_text() == '{"a": 3', "fragment kept")


@test("10 analysis refuses an incomplete run unless --allow-partial")
def t_partial_gate():
    s = SynthRun("partial")
    r = run_wf(s.cfg, "--seeds", "0", "--videos", "SynthArson_001")
    check(r.returncode == 1 and any(e["stage"] == "analysis" and e["event"] == "failed" and "incomplete run" in
                                    e["detail"]["error"] for e in read_jsonl(s.rd / "ledger.jsonl")), r.stderr[-600:])
    r = run_wf(s.cfg, "--seeds", "0", "--resume", "--stages", "analysis", "--allow-partial")
    check(r.returncode == 0, r.stderr[-600:])
    pr = json.loads((s.rd / "results" / "primary.json").read_text())
    check(pr["run_flags"]["partial"] is True, "partial flag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = []
    for fn in RESULTS:
        if a.only and fn.__name__ not in a.only:
            continue
        try:
            detail = fn()
            out.append({"test": fn.__name__, "maps_to": fn.maps_to, "passed": True, "detail": detail})
            print(f"PASS {fn.__name__}  [{fn.maps_to}]" + (f"  {detail}" if detail else ""))
        except Exception as e:
            out.append({"test": fn.__name__, "maps_to": fn.maps_to, "passed": False, "error": repr(e),
                        "traceback": traceback.format_exc()[-1500:]})
            print(f"FAIL {fn.__name__}  [{fn.maps_to}]  {e!r}")
    n_fail = sum(not o["passed"] for o in out)
    print(f"{len(out) - n_fail}/{len(out)} passed")
    if a.out:
        Path(a.out).write_text(json.dumps({"passed": n_fail == 0, "tests": out}, indent=1) + "\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

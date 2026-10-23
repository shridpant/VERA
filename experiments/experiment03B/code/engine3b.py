"""experiment03B model access: the real experiment02 engine (InternVL2-8B, BF16) and a deterministic mock.

Real requests go through experiment02's Engine unchanged (experiments/experiment02/code/engine.py, attr_lib.py), so
prompts, preprocessing, replacement specs and cache keys are identical to experiment02's; 3B's `grey` family is
experiment02's `constant`. The mock implements the same interface with an analytic scorer, for CPU tests and the
synthetic end-to-end run (never for results).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", CODE.parents[2])).resolve()
EXP02_CODE = ROOT / "experiments" / "experiment02" / "code"
sys.path.insert(0, str(CODE))
import config3b as c3  # noqa: E402

TARGET_VARIANT = {"original_generation": "original", "nosound_generation": "nosound",
                  "label_only_score": "label_only"}


def _al():
    if str(EXP02_CODE) not in sys.path:
        sys.path.insert(0, str(EXP02_CODE))
    import attr_lib
    return attr_lib


def exp02_case(inp):
    """experiment02's case shape for one 3B input (the fields its engine and key functions read)."""
    return {"case_id": inp["window_uid"], "video_id": inp["video_id"], "anchor": inp["anchor"],
            "frames_dir": inp["frames_dir"], "frame_indices": list(inp["frame_indices"]),
            "frame_sha256": list(inp["frame_sha256"]), "video_sha256": inp["video_sha256"]}


class KeyMaker:
    """Cache keys exactly as experiment02 computes them (attr_lib.request_key / sequence_key)."""

    def __init__(self, cfg, effective_question):
        self.al = _al()
        self.cfg = cfg
        self.cc = c3.exp02_compat(cfg)
        self.prompt_sha = {v: self.al.sha256_bytes(effective_question(v).encode()) for v in TARGET_VARIANT.values()}

    def spec(self, family, mask):
        return self.al.family_spec(self.cc, c3.FAMILY_TO_EXP02[family] if mask else "none")

    def keys(self, inp, target, family, mask, repeat=0):
        case = exp02_case(inp)
        spec = self.spec(family, mask)
        key = self.al.request_key(self.cc, target, self.prompt_sha[TARGET_VARIANT[target]], case, mask, spec)
        if repeat:
            key = self.al.sha256_bytes(f"{key}:repeat{repeat}".encode())
        return key, self.al.sequence_key(case, mask, spec)


def effective_question_text(cfg, variant):
    """The exact question string VERA builds for 8 frames (experiment02 Engine.effective_question, no model)."""
    al = _al()
    src = ROOT / cfg["source"]["checkout"]
    instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
    qs_all = "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"])
    ins, qs = {"original": (instruct, qs_all), "nosound": (instruct, al.nosound_questions(cfg["prompt"]["guiding_questions"])),
               "label_only": (al.label_only_instruct(instruct), qs_all)}[variant]
    prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(al.N_SLOTS))
    q = ins.replace("$Data", prefix)
    flag = ("Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in "
            "'Yes, there is an anomaly' or 'No, there is no anomaly'.")
    i = q.find(flag)
    return q[:i] + qs + q[i:]


def save_example_inputs(cfg, inp, out_dir, slot=3):
    """Visual examples of the declared replacements: one slot of one input as the 448x448 image VERA's pixel path
    produces (original, blur, grey). Images derived from dataset frames stay local (.gitignore)."""
    out_dir = Path(out_dir)
    if (out_dir / "examples.json").exists():
        return
    from PIL import Image
    al = _al()
    src = ROOT / cfg["source"]["checkout"]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import generate_initial_score as gis
    out_dir.mkdir(parents=True, exist_ok=True)
    fi = inp["frame_indices"][slot]
    frame = Image.open(ROOT / inp["frames_dir"] / f"{fi:06d}.jpg").convert("RGB")
    img = gis.dynamic_preprocess([frame], grid_size=1)
    cc = c3.exp02_compat(cfg)
    rec = {"window_uid": inp["window_uid"], "slot": slot, "frame_index": fi, "files": {}}
    for fam, im in (("original", img), ("blur", al.apply_family(img, al.family_spec(cc, "blur"))),
                    ("grey", al.apply_family(img, al.family_spec(cc, "constant")))):
        p = out_dir / f"{inp['window_uid'].replace('@', '_')}_slot{slot}_{fam}.png"
        im.save(p)
        rec["files"][fam] = {"file": p.name, "sha256": al.sha256_file(p)}
    rec["specs"] = {"blur": al.family_spec(cc, "blur"), "grey": al.family_spec(cc, "constant")}
    (out_dir / "examples.json").write_text(json.dumps(rec, indent=1) + "\n")


class RealEngine:
    """Lazy wrapper: the model loads on the first request that is not served from a cache."""

    def __init__(self, cfg, run_dir, inputs):
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.inputs = inputs
        self._eng = None
        self.keymaker = KeyMaker(cfg, lambda v: effective_question_text(cfg, v))

    def _engine(self):
        if self._eng is None:
            if str(EXP02_CODE) not in sys.path:
                sys.path.insert(0, str(EXP02_CODE))
            import engine as e02
            em = self.run_dir / "eval_manifest_inputs.json"
            seen = {}
            for inp in self.inputs:
                seen.setdefault(inp["manifest_entry"], inp["manifest_length"])
            em.write_text(json.dumps([{"video": k, "length": v} for k, v in seen.items()], indent=2) + "\n")
            self._eng = e02.Engine(c3.exp02_compat(self.cfg), em)
            got = self._eng.effective_question("original")
            if got != effective_question_text(self.cfg, "original"):
                raise RuntimeError("engine question differs from the key-maker question; refusing to run")
        return self._eng

    @property
    def loaded(self):
        return self._eng is not None

    def run(self, target, inp, family, mask):
        eng = self._engine()
        case = exp02_case(inp)
        spec = self.keymaker.spec(family, mask)
        if target == "label_only_score":
            return eng.score(case, tuple(mask), spec)
        return eng.generate(TARGET_VARIANT[target], case, tuple(mask), spec)

    def summary(self):
        return self._eng.summary() if self._eng is not None else {"model_loaded": False}


class MockEngine:
    """Analytic stand-in. Per input, `spec[window_uid]` gives:
       blur: {"w": [8 weights], "inter": [[i, j, coef], ...], "offset": float}   S_hi = offset + sum w_i v_i + sum coef v_i v_j
       grey: same shape (defaults to blur's)
       label_threshold: float (original label = 1 iff S_hi(grey or none input) > threshold; blur inputs use blur)
       missing_labels: [[family, mask], ...]   -> unparsable generation text for these inputs (all repeats)
       nosound_label: 0|1|null, baseline_label: 0|1|null (overrides the threshold rule for the unmasked input)
    v_i = 1 if slot i is visible. Native S = S_hi rounded to the 0.25 grid (mimics the coarse BF16 readout)."""

    def __init__(self, cfg, run_dir, inputs, spec):
        self.cfg, self.inputs, self.spec = cfg, inputs, spec
        self.calls = {"generation_calls": 0, "score_calls": 0}
        self.keymaker = _MockKeys()
        self.loaded = True
        self._failed = set()

    def _f(self, uid, family, mask):
        sp = self.spec[uid]
        fam = sp.get(family) or sp["blur"] if family != "none" else sp["blur"]
        v = [0 if i in mask else 1 for i in range(8)]
        val = fam.get("offset", 0.0) + sum(w * vi for w, vi in zip(fam["w"], v))
        for i, j, c in fam.get("inter", []):
            val += c * v[i] * v[j]
        return float(val)

    def _hashes(self, inp, family, mask):
        return [hashlib.sha256(f"{inp['frame_sha256'][i]}|{family if i in mask else 'none'}".encode()).hexdigest()
                for i in range(8)]

    def run(self, target, inp, family, mask):
        crash = os.environ.get("MOCK_CRASH_AFTER")  # test fixture: hard crash (no clean-up) after N model calls
        if crash and sum(self.calls.values()) >= int(crash):
            os._exit(3)
        fail = os.environ.get("MOCK_FAIL_ONCE")  # test fixture: one transient exception per matching request
        tag = f"{inp['window_uid']}|{target}|{family}|{','.join(map(str, mask))}"
        if fail and fail in tag and tag not in self._failed:
            self._failed.add(tag)
            raise RuntimeError(f"mock transient failure for {tag}")
        uid = inp["window_uid"]
        sp = self.spec[uid]
        hashes = self._hashes(inp, family, mask)
        base = self._hashes(inp, "none", ())
        ident = all((hashes[i] != base[i]) == (i in mask) for i in range(8))
        if target == "label_only_score":
            self.calls["score_calls"] += 1
            s_hi = self._f(uid, family, mask)
            s = round(s_hi * 4) / 4
            # like the real engine, the stored S (log-softmax difference) differs from logit_1 - logit_0 by ~1e-7
            return {"S": s + 1.0e-7, "S_hi": s_hi, "logit_0": 36.0, "logit_1": 36.0 + s, "hi_logit_0": 36.0,
                    "hi_logit_1": 36.0 + s_hi, "S_resolution": 0.25, "S_hi_rounds_to_model_logits": True,
                    "candidate_mass": 0.9, "finite": True, "slot_tensor_sha256": hashes, "slot_identity_ok": ident,
                    "scoring_contract": "label_only_first_token_v1", "runtime_s": 0.0}
        self.calls["generation_calls"] += 1
        if [family, list(mask)] in sp.get("missing_labels", []):
            raw = "I cannot decide."
        elif target == "nosound_generation" and ("nosound_label" in sp or "baseline_label" in sp):
            lab = sp.get("nosound_label", sp.get("baseline_label"))
            raw = "unparsable" if lab is None else f"Answers: mock.\nOutput: {lab}"
        elif not mask and "baseline_label" in sp:
            lab = sp["baseline_label"]
            raw = "unparsable" if lab is None else f"Answers: mock.\nOutput: {lab}"
        else:
            lab = int(self._f(uid, family, mask) > sp.get("label_threshold", 0.0))
            raw = f"Answers: mock.\nOutput: {lab}"
        label, status = _parse(raw)
        return {"raw_response": raw, "label": label, "parse_status": status, "slot_tensor_sha256": hashes,
                "slot_identity_ok": ident, "runtime_s": 0.0}

    def summary(self):
        return {"mock": True, "counts": dict(self.calls)}


def _parse(raw):
    sys.path.insert(0, str(ROOT / "setup" / "scripts"))
    import vera_pipeline as vp
    return vp.strict_parse_label(raw)


class _MockKeys:
    def keys(self, inp, target, family, mask, repeat=0):
        seq = hashlib.sha256(json.dumps([inp["window_uid"], inp["frame_sha256"], family if mask else "none",
                                         sorted(mask)]).encode()).hexdigest()
        key = hashlib.sha256(json.dumps(["mock", target, seq, repeat]).encode()).hexdigest()
        return key, seq

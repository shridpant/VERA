"""experiment02 (temporal_attribution_feasibility): GPU-free building blocks.

Targets are kept apart everywhere:
  original     VERA's original explanation-first prompt; outcome = strict-parsed binary label (+ raw text)
  nosound      the original prompt with ONLY guiding question 5 (sound) removed (Stage A3 ablation)
  label_only   an explicitly modified label-only prompt; used for label-only generation and for the
               auxiliary numeric score S = log P(1) - log P(0) at the start of the assistant reply
Nothing here attributes VERA's final refined timeline.

Perturbations act on the 448x448 RGB image each slot has right after VERA's dynamic_preprocess() and
before ToTensor/Normalize, so the eight slots, their order, timestamps and all prompt text stay fixed.
"""
import hashlib
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VERA_PROJECT_ROOT", CODE.parents[2])).resolve()
sys.path.insert(0, str(ROOT / "setup" / "scripts"))
import vera_pipeline as vp  # noqa: E402

N_SLOTS = 8
SCORING_CONTRACT_VERSION = "label_only_first_token_v1"
PREPROCESSING_ID = ("vera generate_initial_score.py: Image.open().convert('RGB') -> dynamic_preprocess([frame], "
                    "grid_size=1) [PIL resize to 448x448] -> <perturbation here> -> build_transform(448): "
                    "Resize((448,448), BICUBIC) [no-op], ToTensor, Normalize(ImageNet) -> stack, bf16")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest()


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    # split on "\n" only: str.splitlines() would also split on U+2028/U+0085 inside a record
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def jsonl_line(obj):
    return json.dumps(obj, ensure_ascii=True) + "\n"


CODE_FILES = ("attr_lib.py", "engine.py", "run_attr.py")


def code_sha256():
    """Hashes of the code that computes results (analyzer/validator/tests are excluded: they never call the model)."""
    return {f: sha256_file(CODE / f) for f in CODE_FILES}


def bf16_spacing(x):
    """Gap between adjacent bfloat16 values at magnitude |x| (8 significant bits)."""
    import math
    ax = abs(x)
    if ax == 0 or not math.isfinite(ax):
        return 0.0
    return 2.0 ** (math.floor(math.log2(ax)) - 7)


def write_json(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


# ---------------------------------------------------------------- masks

def adjacent_groups(size):
    return [tuple(range(i, i + size)) for i in range(N_SLOTS - size + 1)]


def primary_masks():
    """22 per replacement family: 8 singles, 7 adjacent pairs, 6 adjacent triples, all eight."""
    out = [("single", g) for g in adjacent_groups(1)]
    out += [("adjacent_pair", g) for g in adjacent_groups(2)]
    out += [("adjacent_triple", g) for g in adjacent_groups(3)]
    out += [("all_eight", tuple(range(N_SLOTS)))]
    return out


def is_nonadjacent(g):
    return all(b - a >= 2 for a, b in zip(g, g[1:]))


def nonconsecutive_groups(size, n, seed):
    """n distinct size-k slot sets in which no two slots are temporally adjacent (seeded, sorted pool)."""
    pool = [g for g in itertools.combinations(range(N_SLOTS), size) if is_nonadjacent(g)]
    return sorted(random.Random(seed).sample(pool, n))


def control_masks(cfg):
    s = cfg["stage_b"]["nonconsecutive_seed"]
    n = cfg["stage_b"]["nonconsecutive_per_size"]
    return ([("nonconsecutive_pair", g) for g in nonconsecutive_groups(2, n, s)]
            + [("nonconsecutive_triple", g) for g in nonconsecutive_groups(3, n, s + 1)])


# ---------------------------------------------------------------- prompts

LABEL_ONLY_EDITS = [  # (original text, replacement) applied to VERA_learner_instruct.txt, each must occur once
    ("Please answer the prompt questions.", "Please consider the prompt questions."),
    ("Answer the following questions based on what you see from the video frames and provide an explanation in one sentence. ",
     "Consider the following questions based on what you see from the video frames. "),
    ("Answers to Prompt Questions: [Provide your analysis by answering the questions listed in Prompt Questions.]\nOutput:\n",
     ""),
]


def label_only_instruct(original):
    s = original
    for old, new in LABEL_ONLY_EDITS:
        if s.count(old) != 1:
            raise ValueError(f"label-only edit anchor not found exactly once: {old[:50]!r}")
        s = s.replace(old, new)
    return s


def nosound_questions(questions):
    """The released questions with only #5 (sound) removed; numbering and wording of 1-4 unchanged."""
    kept = [q for q in questions if not q.startswith("5.")]
    if len(kept) != 4 or "sound" in "".join(kept).lower():
        raise ValueError("expected to remove exactly the sound question")
    return "".join(q + "\n" for q in kept)


_LABEL_ONLY_STRIP = " \t\r\n`[]()\"'*._"


def parse_label_only(text):
    """Label-only replies must be a single 0 or 1 after stripping whitespace/fences/brackets/quotes/period."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    t = text.replace("```", " ").strip(_LABEL_ONLY_STRIP)
    if re.fullmatch(r"(?i)output\s*:\s*[01]", t):
        t = t[-1]
    if t in ("0", "1"):
        return int(t), "ok"
    return None, "not_a_single_label"


SOUND_SENT = re.compile(r"[^.\n]*\b(sound|sounds|noise|noises|noisy|audio|hear|heard|loud|silent|siren|honk)\b[^.\n]*[.\n]?",
                        re.IGNORECASE)


def sound_sentences(text):
    return [m.group(0).strip() for m in SOUND_SENT.finditer(text or "")]


CLAIM_KEYWORDS = {  # declared before any perturbation output was read; descriptive text-claim indicators only
    "train": r"\btrain", "truck": r"\btruck", "collision": r"collid|collision|crash|hit by|struck",
    "damage": r"damage|debris", "person": r"\bperson|\bpeople|pedestrian|\bman\b|\bwoman",
    "crossing": r"cross", "no_anomaly_stated": r"\bno anomaly\b|is not an anomaly|isn't an anomaly",
}


def claim_flags(text):
    t = (text or "").lower()
    return {k: bool(re.search(v, t)) for k, v in CLAIM_KEYWORDS.items()}


# ---------------------------------------------------------------- transformations and cache keys

def family_spec(cfg, family):
    if family == "none":
        return {"family": "none"}
    if family == "blur":
        res = cfg["stage_b"]["preprocessed_resolution"]
        return {"family": "blur", "op": "PIL.ImageFilter.GaussianBlur", "applied_at": "448x448 after dynamic_preprocess",
                "resolution": [res, res], "radius_fraction": cfg["stage_b"]["blur_radius_fraction"],
                "radius_px": cfg["stage_b"]["blur_radius_fraction"] * res,
                "radius_semantics": "Pillow 10.2 GaussianBlur radius = Gaussian standard deviation (sigma), "
                                    "approximated by 3 extended box blurs; effective support about +/-3 sigma"}
    if family == "constant":
        res = cfg["stage_b"]["preprocessed_resolution"]
        return {"family": "constant", "rgb": list(cfg["stage_b"]["constant_rgb"]), "resolution": [res, res],
                "applied_at": "448x448 after dynamic_preprocess"}
    raise ValueError(family)


def apply_family(img, spec):
    from PIL import Image, ImageFilter
    if list(img.size) != list(spec["resolution"]):
        raise ValueError(f"perturbation expects {spec['resolution']}, got {img.size}")
    if spec["family"] == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=spec["radius_px"]))
    if spec["family"] == "constant":
        return Image.new("RGB", img.size, tuple(spec["rgb"]))
    raise ValueError(spec)


def slot_tensors(gis, transform, frames_dir, frame_indices, mask, spec):
    """The upstream per-frame pixel path (generate_initial_score.py main loop), with `spec` applied to the slots in
    `mask` between dynamic_preprocess and the ToTensor/Normalize transform. Returns the float32 CPU tensor
    (1, 8, 3, 448, 448) and one SHA-256 per slot of its normalized float32 tensor."""
    import torch
    from PIL import Image
    pixel_values_list, slot_hashes = [], []
    for slot, fi in enumerate(frame_indices):
        frame = Image.open(Path(frames_dir) / f"{fi:06d}.jpg").convert("RGB")
        stitched_image = gis.dynamic_preprocess([frame], grid_size=1)
        if slot in mask:
            stitched_image = apply_family(stitched_image, spec)
        pixel_values = [transform(stitched_image)]
        pixel_values = torch.stack(pixel_values)
        slot_hashes.append(sha256_bytes(pixel_values.numpy().tobytes()))
        pixel_values_list.append(pixel_values)
    return torch.cat(pixel_values_list).unsqueeze(0), slot_hashes


def sequence_key(case, mask, spec):
    """Identity of one 8-slot input sequence (independent of prompt/target)."""
    return sha256_bytes(json.dumps({
        "video_sha256": case["video_sha256"], "frame_ids": case["frame_indices"],
        "frame_sha256": case["frame_sha256"], "mask": sorted(mask), "family": spec if mask else {"family": "none"},
        "preprocessing": PREPROCESSING_ID}, sort_keys=True).encode())


def request_key(cfg, target, prompt_sha, case, mask, spec):
    """Cache key: model revision + prompt/scoring contract + source identity + frames + transform + preprocessing."""
    return sha256_bytes(json.dumps({
        "model_revision": cfg["model"]["revision"], "source_head": cfg["source"]["expected_head"],
        "target": target, "prompt_sha256": prompt_sha,
        "scoring_contract": SCORING_CONTRACT_VERSION if target == "label_only_score" else None,
        "generation": cfg["generation"] if target != "label_only_score" else None,
        "sequence": sequence_key(case, mask, spec)}, sort_keys=True).encode())


# ---------------------------------------------------------------- effects

def signed_binary_diff(orig_label, pert_label):
    """orig - perturbed on the 0/1 anomaly label: +1 = perturbation removed the anomaly label."""
    if orig_label is None or pert_label is None:
        return None
    return int(orig_label) - int(pert_label)


def interaction(group_effect, single_effects):
    """Non-additivity on the chosen scale: effect(G) - sum(effect(single g) for g in G). None if anything missing."""
    if group_effect is None or any(v is None for v in single_effects):
        return None
    return group_effect - sum(single_effects)


def rank_slots(values, direction=1):
    """Order slots by direction*value descending; ties broken by slot index ascending.
    Returns (order, tie_groups) where tie_groups lists slot sets sharing an identical value."""
    items = [(s, v) for s, v in values.items() if v is not None]
    order = [s for s, _ in sorted(items, key=lambda t: (-direction * t[1], t[0]))]
    by_val = {}
    for s, v in items:
        by_val.setdefault(v, []).append(s)
    ties = [sorted(g) for g in by_val.values() if len(g) > 1]
    return order, ties


def singleton_scores(effects):
    """effects: {mask_tuple: Delta}. Frame score = Delta of its single-frame intervention."""
    return {s: effects.get((s,)) for s in range(N_SLOTS)}


def group_scores_pooled(effects, sizes=(2, 3)):
    """Heuristic named in the brief: mean of Delta(G)/|G| over all tested adjacent groups (of the given sizes)
    containing the frame. Edge frames are covered by fewer groups (see coverage())."""
    out = {}
    for s in range(N_SLOTS):
        vals = [effects[g] / len(g) for g in effects if len(g) in sizes and s in g and _is_adjacent(g)
                and effects[g] is not None]
        out[s] = sum(vals) / len(vals) if vals else None
    return out


def group_scores_size_balanced(effects, sizes=(2, 3)):
    """Same ingredients, but averaged within each group size first, then across sizes, so a constant per-size
    effect cannot create a boundary artifact through unequal pair/triple coverage."""
    out = {}
    for s in range(N_SLOTS):
        per_size = []
        for k in sizes:
            vals = [effects[g] / k for g in effects if len(g) == k and s in g and _is_adjacent(g)
                    and effects[g] is not None]
            if vals:
                per_size.append(sum(vals) / len(vals))
        out[s] = sum(per_size) / len(per_size) if per_size else None
    return out


def coverage(sizes=(2, 3)):
    return {s: {k: sum(1 for g in adjacent_groups(k) if s in g) for k in sizes} for s in range(N_SLOTS)}


def _is_adjacent(g):
    return all(b - a == 1 for a, b in zip(g, g[1:]))


def ranking_status(values, budgets=(), direction=1):
    """complete | incomplete (missing slots) | all_tied; plus whether a tie straddles each budget cut."""
    present = {s: v for s, v in values.items() if v is not None}
    if len(present) < N_SLOTS:
        return {"status": "incomplete", "missing_slots": sorted(set(range(N_SLOTS)) - set(present))}
    distinct = len(set(present.values()))
    out = {"status": "all_tied" if distinct <= 1 else "complete", "n_distinct_values": distinct}
    order, _ = rank_slots(present, direction)
    for b in budgets:
        out[f"tie_spans_cut_at_{b}"] = present[order[b - 1]] == present[order[b]] if 0 < b < N_SLOTS else False
    return out


def random_ranking(seed):
    return random.Random(seed).sample(range(N_SLOTS), N_SLOTS)


def evaluation_masks(order, budget):
    """removal: replace the top-`budget` slots; retention: keep them and replace the rest."""
    if sorted(order) != list(range(N_SLOTS)) or not 0 < budget < N_SLOTS:
        raise ValueError(f"evaluation needs a full 8-slot ranking and 0 < budget < 8, got {order} / {budget}")
    top = tuple(sorted(order[:budget]))
    rest = tuple(s for s in range(N_SLOTS) if s not in top)
    return {"removal": top, "retention": rest}

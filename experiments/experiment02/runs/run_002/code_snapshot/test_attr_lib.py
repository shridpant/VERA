#!/usr/bin/env python3
"""CPU acceptance tests for experiment02. Exit 1 on any failure.
Usage (on oz): setup/environment/env/bin/python experiments/experiment02/code/test_attr_lib.py
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import yaml  # noqa: E402

import attr_lib as al  # noqa: E402

failures = []


def test(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


cfg = yaml.safe_load((CODE.parent / "config.yaml").read_text(encoding="utf-8"))

# masks
pm = al.primary_masks()
sizes = [len(m) for _, m in pm]
test("22 primary masks: 8 singles, 7 adjacent pairs, 6 adjacent triples, 1 all-eight",
     len(pm) == 22 and sizes.count(1) == 8 and sizes.count(2) == 7 and sizes.count(3) == 6 and sizes.count(8) == 1)
test("primary masks unique", len({m for _, m in pm}) == 22)
cm = al.control_masks(cfg)
test("6 nonconsecutive controls (3 pairs, 3 triples), no two slots adjacent, distinct, reproducible",
     len(cm) == 6 and all(al.is_nonadjacent(m) for _, m in cm) and len({m for _, m in cm}) == 6
     and cm == al.control_masks(cfg) and [len(m) for _, m in cm] == [2, 2, 2, 3, 3, 3], str([m for _, m in cm]))
test("nonconsecutive controls never coincide with adjacent groups", not ({m for _, m in cm} & {m for _, m in pm}))

# prompts
src = al.ROOT / cfg["source"]["checkout"]
instruct = (src / cfg["prompt"]["learner_instruct_file"]).read_text(encoding="utf-8")
lo = al.label_only_instruct(instruct)
test("label-only instruct keeps $Data and the question insertion flag",
     "[$Data]" in lo and "Based on the analysis above, please conclude your answer" in lo)
test("label-only instruct no longer asks for answers/explanations or an Output field",
     "Answers to Prompt Questions" not in lo and "explanation" not in lo and "Output:" not in lo)
test("label-only instruct keeps the label request", "[ONLY the integer class label; make necessary assumptions if needed]" in lo)
ns = al.nosound_questions(cfg["prompt"]["guiding_questions"])
test("no-sound questions drop only question 5", ns.count("\n") == 4 and "sound" not in ns.lower()
     and ns == "".join(q + "\n" for q in cfg["prompt"]["guiding_questions"][:4]))

# label-only parser
for text, want in [("1", 1), ("0", 0), (" 1\n", 1), ("```\n0\n```", 0), ("[1]", 1), ("1.", 1), ("Output: 1", 1),
                   ("Yes, there is an anomaly", None), ("0 or 1", None), ("", None), ("10", None)]:
    test(f"label-only parse {text!r}", al.parse_label_only(text)[0] == want, str(al.parse_label_only(text)))

# constant synthetic group-effect fixture (boundary artifacts)
const = {g: 1.0 for k in (1, 2, 3) for g in al.adjacent_groups(k)}
pooled = al.group_scores_pooled(const)
bal = al.group_scores_size_balanced(const)
test("FIXTURE: pooled mean of Delta/|G| is NOT flat under a constant group effect (boundary/coverage artifact)",
     len({round(v, 12) for v in pooled.values()}) > 1, json.dumps({k: round(v, 4) for k, v in pooled.items()}))
test("FIXTURE: size-balanced score is flat under a constant group effect", len({round(v, 12) for v in bal.values()}) == 1,
     json.dumps({k: round(v, 4) for k, v in bal.items()}))
cov = al.coverage()
test("coverage: edge slots 0/7 in 1 pair + 1 triple, slot 1 in 2+2, middle slots in 2+3",
     cov[0] == {2: 1, 3: 1} and cov[7] == {2: 1, 3: 1} and cov[1] == {2: 2, 3: 2} and cov[3] == {2: 2, 3: 3})
order, ties = al.rank_slots(bal, 1)
test("all-tied ranking falls back to slot order and reports one tie group", order == list(range(8)) and ties == [list(range(8))])

# ranking and effects
order, ties = al.rank_slots({0: 0.1, 1: 0.5, 2: 0.5, 3: -0.2, 4: 0.0, 5: 0.3, 6: None, 7: 0.2}, direction=1)
test("rank_slots: descending, ties by slot index, None dropped", order == [1, 2, 5, 7, 0, 4, 3] and ties == [[1, 2]], str(order))
order, _ = al.rank_slots({0: 0.1, 1: 0.5, 2: -0.4, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}, direction=-1)
test("rank_slots direction -1 ranks most negative first", order[0] == 2 and order[-1] == 1, str(order))
test("signed binary diff keeps sign and missing", al.signed_binary_diff(1, 0) == 1 and al.signed_binary_diff(0, 1) == -1
     and al.signed_binary_diff(1, None) is None)
test("interaction = group - sum(singles); None if a single is missing",
     abs(al.interaction(0.9, [0.2, 0.3]) - 0.4) < 1e-12 and al.interaction(0.9, [0.2, None]) is None)
em = al.evaluation_masks([5, 2, 7, 0, 1, 3, 4, 6], 2)
test("evaluation masks: removal = top-2, retention = complement", em == {"removal": (2, 5), "retention": (0, 1, 3, 4, 6, 7)})
test("random rankings are seeded permutations", al.random_ranking(0) == al.random_ranking(0)
     and sorted(al.random_ranking(1)) == list(range(8)) and al.random_ranking(0) != al.random_ranking(1))

# review fixes: ranking status, resolution, guards, jsonl, claim regex
test("ranking_status: all tied", al.ranking_status({s: 0.0 for s in range(8)}, (2, 4))["status"] == "all_tied")
test("ranking_status: incomplete", al.ranking_status({**{s: 0.1 * s for s in range(7)}, 7: None})["status"] == "incomplete")
st = al.ranking_status({0: 0.5, 1: 0.5, 2: 0.4, 3: 0.4, 4: 0.3, 5: 0.2, 6: 0.1, 7: 0.0}, (2, 4))
test("ranking_status: tie straddling a cut detected", st["status"] == "complete" and st["tie_spans_cut_at_2"] is False
     and st["tie_spans_cut_at_4"] is False, str(st))
st = al.ranking_status({0: 0.5, 1: 0.4, 2: 0.4, 3: 0.3, 4: 0.3, 5: 0.2, 6: 0.1, 7: 0.0}, (2, 4), direction=1)
test("ranking_status: tie across top-2 and top-4 cuts", st["tie_spans_cut_at_2"] is True and st["tie_spans_cut_at_4"] is True, str(st))
test("bf16 spacing: 0.0625 in [8,16), 0.125 in [16,32), 2^-7 at 1", al.bf16_spacing(9.0) == 0.0625
     and al.bf16_spacing(-20.0) == 0.125 and al.bf16_spacing(1.0) == 2 ** -7)
try:
    al.evaluation_masks([0, 1, 2], 2)
    test("evaluation_masks refuses a partial ranking", False)
except ValueError:
    test("evaluation_masks refuses a partial ranking", True)
import tempfile  # noqa: E402
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
    f.write(al.jsonl_line({"t": "a b\u0085c"}))
    f.write(al.jsonl_line({"t": "d"}))
test("jsonl round-trips text containing U+2028/U+0085", al.read_jsonl(f.name) == [{"t": "a b\u0085c"}, {"t": "d"}])
e01 = al.read_jsonl(al.ROOT / "experiments/experiment01/runs/run_001/results/raw_responses.jsonl")
fp = [r["anchor_frame"] for r in e01 if r["label"] == 1 and al.claim_flags(r["raw_response"])["no_anomaly_stated"]]
fn = [r["anchor_frame"] for r in e01 if r["label"] == 0 and not al.claim_flags(r["raw_response"])["no_anomaly_stated"]]
test("claim flag no_anomaly_stated never fires on experiment01 label-1 responses", not fp, f"fires on {fp}")
print(f"INFO no_anomaly_stated absent from {len(fn)}/{sum(r['label'] == 0 for r in e01)} experiment01 label-0 responses")

# cache keys
case = {"video_sha256": "v", "frame_indices": list(range(8)), "frame_sha256": ["h"] * 8}
blur, const_ = al.family_spec(cfg, "blur"), al.family_spec(cfg, "constant")
base = al.request_key(cfg, "original_generation", "p", case, (3,), blur)
variants = [al.request_key(cfg, "label_only_score", "p", case, (3,), blur),
            al.request_key(cfg, "original_generation", "q", case, (3,), blur),
            al.request_key(cfg, "original_generation", "p", case, (4,), blur),
            al.request_key(cfg, "original_generation", "p", case, (3,), const_),
            al.request_key(cfg, "original_generation", "p", {**case, "video_sha256": "w"}, (3,), blur),
            al.request_key(cfg, "original_generation", "p", {**case, "frame_indices": list(range(1, 9))}, (3,), blur),
            al.request_key({**cfg, "model": {**cfg["model"], "revision": "x"}}, "original_generation", "p", case, (3,), blur)]
test("cache key changes with target, prompt, mask, family, video, frames and model revision", base not in variants and len(set(variants)) == len(variants))
test("cache key is order-insensitive in the mask", al.request_key(cfg, "original_generation", "p", case, (3, 5), blur)
     == al.request_key(cfg, "original_generation", "p", case, (5, 3), blur))
test("blur radius = 0.04 * 448 = 17.92 px", abs(blur["radius_px"] - 17.92) < 1e-9)

# pixel path identity against the upstream main-loop lines (CPU)
try:
    import torch
    from PIL import Image
    sys.path.insert(0, str(src))
    import generate_initial_score as gis
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{"video": "frames/RoadAccidents/RoadAccidents127_x264.mp4", "length": 2580}], f)
    ds = gis.Video_Instruct_Dataset_Inference(vis_root=str(al.ROOT / cfg["frames"]["cache_root"]) + "/", ann_root=f.name)
    video_path, _, _ = ds[0]
    idx = [2090, 2127, 2165, 2202, 2240, 2277, 2315, 2352]
    # verbatim upstream lines (generate_initial_score.py main loop), CPU
    pixel_values_list, num_patches_list = [], []
    for frame_name in ['{:06d}'.format(i) + '.jpg' for i in idx]:
        frame_path = video_path + '/' + frame_name
        frame = Image.open(frame_path).convert('RGB')
        stitched_image = gis.dynamic_preprocess([frame], grid_size=1)
        pixel_values = [ds.transform(stitched_image)]
        pixel_values = torch.stack(pixel_values)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
    upstream = torch.cat(pixel_values_list).unsqueeze(0)
    ours, h0 = al.slot_tensors(gis, ds.transform, video_path, idx, (), {"family": "none"})
    test("unperturbed slot tensors are bit-identical to the upstream main loop", torch.equal(ours, upstream)
         and tuple(ours.shape) == (1, 8, 3, 448, 448) and num_patches_list == [1] * 8)
    for fam in ("blur", "constant"):
        pert, h1 = al.slot_tensors(gis, ds.transform, video_path, idx, (3, 4), al.family_spec(cfg, fam))
        same = [i for i in range(8) if h1[i] == h0[i]]
        diff = (pert - ours).abs().amax(dim=(0, 2, 3, 4))
        test(f"{fam}: only slots 3 and 4 change; others bit-identical", same == [0, 1, 2, 5, 6, 7]
             and all(float(diff[i]) == 0 for i in same) and all(float(diff[i]) > 0 for i in (3, 4)),
             f"max|diff| per slot {[round(float(x), 3) for x in diff]}")
    const_img = al.apply_family(Image.new("RGB", (448, 448)), al.family_spec(cfg, "constant"))
    test("constant replacement is exactly RGB(127,127,127) at 448x448", const_img.size == (448, 448)
         and const_img.getextrema() == ((127, 127), (127, 127), (127, 127)))
except Exception as e:  # noqa: BLE001
    test("pixel path tests ran", False, repr(e))

print("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)

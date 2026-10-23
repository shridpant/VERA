#!/usr/bin/env python3
"""Stage D: bounded spatial-attribution feasibility for the AUXILIARY label-only score S (one case, R-accident).

Isolated: imports the unchanged engine, builds its own differentiable forward (gradients enabled only here, weights
frozen, no parameter gradients), and writes only to <run-dir>/spatial/. Every map explains S under the label-only
prompt, not VERA's explanation-first decision and not VERA's final timeline.

Maps (all saved as native arrays):
  gradcam_vit_32x32      Grad-CAM adapted to ViT tokens: A = InternViT last-layer patch tokens (8x1024 tokens x 1024 ch),
                         w_c = mean_tokens dS/dA_c, map = ReLU(sum_c w_c A_c)
  gradcampp_vit_32x32    Grad-CAM++ closed-form weights (Chattopadhay et al. 2018, as implemented in pytorch-grad-cam):
                         alpha = g^2 / (2 g^2 + sum_tokens(A) g^3), w_c = sum_tokens alpha * ReLU(g); map = ReLU(sum_c w_c A_c)
  gradxinput_tokens_16x16  (dS/dE) . E summed over 4096 dims, E = projected visual tokens fed to the LLM (signed)
  pixel_grad_abs_448     |dS/dx| summed over RGB of the normalized 448x448 inputs
Usage: python experiments/experiment02/code/spatial_probe.py --run-dir experiments/experiment02/runs/run_002
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import attr_lib as al  # noqa: E402
import engine as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--case", default="R-accident")
    ap.add_argument("--occlusion-fraction", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    rd = al.ROOT / a.run_dir
    cfg = yaml.safe_load((rd / "resolved_config.yaml").read_text(encoding="utf-8"))
    man = json.loads((rd / "run_manifest.json").read_text(encoding="utf-8"))
    diag = json.loads((rd / "results" / "scorer_diagnostics.json").read_text(encoding="utf-8"))
    if diag["status"] == "blocked" or not any(s["status"] == "ok" for s in man["stages"].get("B", [])):
        sys.exit("Stage D precondition failed: needs a non-blocked numeric scorer and a completed Stage B")
    cases = {c["case_id"]: c for c in json.loads((rd / "cases.json").read_text(encoding="utf-8"))["cases"]}
    case = cases[a.case]
    out = rd / "spatial"
    out.mkdir(exist_ok=True)
    summ = {"case_id": a.case, "target": "auxiliary label-only score S = log P('1') - log P('0') (contract "
            f"{al.SCORING_CONTRACT_VERSION}); NOT VERA's original explanation-first decision",
            "spatial_probe_sha256": al.sha256_file(Path(__file__)), "engine_sha256": al.sha256_file(CODE / "engine.py")}

    eng = E.Engine(cfg, rd / "results" / "eval_manifest_cases.json")
    torch = eng.torch
    m = eng.model

    # 1. no-grad reference through the Stage A scorer path
    pv0, _ = eng.pixel_values(case, (), {"family": "none"})
    pv, ids, am, q = eng.capture_label_only_inputs(pv0)
    eng.setup_candidates(q, ids)
    lp, _, _ = eng.score_from_inputs(pv, ids, am)
    c0 = eng.candidates["candidates"]["0"]["token_ids"][0]
    c1 = eng.candidates["candidates"]["1"]["token_ids"][0]
    summ["S_no_grad_path"] = lp["1"] - lp["0"]
    summ["S_hi_no_grad_path"] = lp["hi_logit_1"] - lp["hi_logit_0"]

    # 2. token <-> pixel mapping, measured on InternVL's own pixel_shuffle (ps_version v2)
    ids_grid = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1).repeat(1, 1, 1, 4).cuda()
    ps = m.pixel_shuffle(ids_grid, scale_factor=m.downsample_ratio)[0]  # (16, 16, 16): 4 source ids per token
    mapping_ok = True
    for r in range(16):
        for cc in range(16):
            src = sorted(int(v) for v in ps[r, cc, ::4].tolist())
            want = sorted([(2 * r) * 32 + 2 * cc, (2 * r) * 32 + 2 * cc + 1, (2 * r + 1) * 32 + 2 * cc, (2 * r + 1) * 32 + 2 * cc + 1])
            mapping_ok &= src == want
    summ["token_mapping"] = {
        "vit_tokens": "InternViT-300M last layer output, CLS dropped: 32x32 row-major grid; token (i, j) covers pixels "
                      "rows 14i..14i+13, cols 14j..14j+13 of the 448x448 input",
        "visual_tokens": "after pixel_shuffle(0.5, ps_version v2) + mlp1: 16x16 tokens; token (r, c) merges ViT tokens "
                         "(2r..2r+1, 2c..2c+1) = pixels rows 28r..28r+27, cols 28c..28c+27",
        "measured_on_model_pixel_shuffle": mapping_ok,
        "llm_placement": "all 8x256 visual tokens are written, in slot order, into the single <IMG_CONTEXT> block at Frame1"}

    # 3. isolated differentiable forward (no parameter gradients; weights already requires_grad=False)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.enable_grad():
        pvg = pv.detach().clone().requires_grad_(True)
        vit_full = m.vision_model(pixel_values=pvg, output_hidden_states=False, return_dict=True).last_hidden_state
        vit_full.retain_grad()
        x = vit_full[:, 1:, :]
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, -1)
        x = m.pixel_shuffle(x, scale_factor=m.downsample_ratio)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        vit_embeds = m.mlp1(x)
        vit_embeds.retain_grad()
        emb = m.language_model.get_input_embeddings()(ids)
        B, N, C = emb.shape
        emb = emb.reshape(B * N, C).clone()
        sel = ids.reshape(B * N) == m.img_context_token_id
        emb[sel] = vit_embeds.reshape(-1, C).to(emb.dtype)
        outp = m.language_model(inputs_embeds=emb.reshape(B, N, C), attention_mask=am, use_cache=False, return_dict=True)
        logits = outp.logits[0, -1].float()
        logp = torch.log_softmax(logits, -1)
        S = logp[c1] - logp[c0]
        S.backward()
    torch.cuda.synchronize()
    eng.counts["gradient_calls"] += 1
    summ["S_grad_path"] = float(S)
    summ["forward_agreement_abs_diff"] = abs(float(S) - summ["S_no_grad_path"])
    summ["gradient_call_seconds"] = round(time.time() - t0, 2)
    summ["peak_gpu_mem_allocated_gib_gradient_call"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    summ["parameter_grads_stored"] = sum(p.grad is not None for p in m.parameters())

    G = vit_full.grad[:, 1:, :].float()
    A = vit_full[:, 1:, :].detach().float()
    wts = G.mean(dim=1)
    gradcam = torch.relu((A * wts[:, None, :]).sum(-1)).reshape(8, 32, 32)
    g2, g3 = G ** 2, G ** 3
    sumA = A.sum(dim=1, keepdim=True)
    alpha = g2 / (2 * g2 + sumA * g3 + 1e-7)
    alpha = torch.where(G != 0, alpha, torch.zeros_like(alpha))
    wpp = (torch.relu(G) * alpha).sum(dim=1)
    gradcampp = torch.relu((A * wpp[:, None, :]).sum(-1)).reshape(8, 32, 32)
    gxi = (vit_embeds.grad.float() * vit_embeds.detach().float()).sum(-1).reshape(8, 16, 16)
    pg = pvg.grad.float()
    pix_abs = pg.abs().sum(1)
    arrays = {"gradcam_vit_32x32": gradcam, "gradcampp_vit_32x32": gradcampp, "gradxinput_tokens_16x16": gxi,
              "pixel_grad_abs_448": pix_abs}
    summ["finite"] = {k: bool(torch.isfinite(v).all()) for k, v in arrays.items()}
    summ["finite"]["vit_token_grad"] = bool(torch.isfinite(G).all())
    summ["nonzero"] = {k: float(v.abs().sum()) > 0 for k, v in arrays.items()}
    for k, v in arrays.items():
        np.save(out / f"{k}.npy", v.detach().cpu().numpy().astype(np.float32 if v.numel() < 10 ** 6 else np.float16))
    # per-frame totals on one common scale (not per-frame normalized)
    summ["per_frame_share"] = {k: [round(float(x), 4) for x in (v.clamp(min=0).flatten(1).sum(1) / v.clamp(min=0).sum()).tolist()]
                               for k, v in arrays.items() if k != "gradxinput_tokens_16x16"}
    summ["per_frame_gradxinput_signed_sum"] = [round(float(x), 5) for x in gxi.flatten(1).sum(1).tolist()]
    summ["grad_norm_per_frame_pixels"] = [round(float(x), 6) for x in pg.flatten(1).norm(dim=1).tolist()]
    del vit_full, vit_embeds, outp, logits, emb, pvg
    torch.cuda.empty_cache()

    # 4. small occlusion diagnostic: grey out the top Grad-CAM cells (28x28 px) vs random cells of equal count
    cell = gradcam.reshape(8, 16, 2, 16, 2).sum((2, 4)).flatten().cpu().numpy()  # 8*256 cells
    k = int(round(a.occlusion_fraction * cell.size))
    top = [int(i) for i in np.argsort(-cell, kind="stable")[:k]]
    mean = torch.tensor(E.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(E.IMAGENET_STD).view(3, 1, 1)
    grey = (torch.full((3, 1, 1), 127 / 255.0) - mean) / std
    base_cpu, _ = al.slot_tensors(eng.gis, eng.transform, al.ROOT / case["frames_dir"], case["frame_indices"], (), {"family": "none"})

    def occluded_score(cells):
        t = base_cpu.clone()
        for ci in cells:
            f, rc = divmod(ci, 256)
            r, cc = divmod(rc, 16)
            t[0, f, :, 28 * r:28 * r + 28, 28 * cc:28 * cc + 28] = grey
        p, i2, m2, _ = eng.capture_label_only_inputs(t.to(torch.bfloat16).cuda())
        l2, _, _ = eng.score_from_inputs(p, i2, m2)
        eng.counts["score_calls"] += 1
        return {"S": l2["1"] - l2["0"], "S_hi": l2["hi_logit_1"] - l2["hi_logit_0"],
                "cells_per_frame": [sum(1 for ci in cells if ci // 256 == f) for f in range(8)],
                "input_sha256": al.sha256_bytes(t.numpy().tobytes())}
    occ = {"cells_total": int(cell.size), "k": k, "cell_px": 28, "fill": "RGB(127,127,127) before normalization",
           "baseline": {"S": summ["S_no_grad_path"], "S_hi": summ["S_hi_no_grad_path"]},
           "gradcam_top": occluded_score(top)}
    for sd in [int(s) for s in a.seeds.split(",")]:
        occ[f"random_seed{sd}"] = occluded_score(sorted(random.Random(sd).sample(range(cell.size), k)))
    occ["note"] = ("Diagnostic of the auxiliary score only; 4 extra transformed sequences; greyed 28x28 cells are an "
                   "out-of-distribution intervention, so a larger drop for top cells is consistent with, not proof of, "
                   "the map tracking evidence")
    summ["occlusion_check"] = occ
    summ["engine_counts"] = dict(eng.counts)
    al.write_json(out / "spatial_summary.json", summ)
    make_preview(case, gradcam.cpu().numpy(), out / "preview_gradcam_global_norm.png", summ)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("token_mapping",)}, indent=1)[:4000])


def make_preview(case, cam, path, summ):
    """Small labelled diagnostic: 8 input frames with the Grad-CAM map, ONE normalization shared by all frames."""
    from PIL import Image, ImageDraw
    vmax = float(cam.max()) or 1.0
    W = 224
    sheet = Image.new("RGB", (4 * W, 2 * (W + 30) + 34), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), "Grad-CAM (ViT last layer) for the AUXILIARY label-only score S, not VERA's explanation. "
                   f"One color scale for all 8 frames (max={vmax:.3g}). Diagnostic only.", fill="black")
    d.text((6, 18), f"case {summ['case_id']}; S={summ['S_no_grad_path']:.2f}; red = higher map value", fill="black")
    for s, fi in enumerate(case["frame_indices"]):
        im = Image.open(al.ROOT / case["frames_dir"] / f"{fi:06d}.jpg").convert("RGB").resize((W, W))
        heat = Image.fromarray((np.clip(cam[s] / vmax, 0, 1) * 255).astype(np.uint8)).resize((W, W), Image.NEAREST)
        red = Image.merge("RGB", (heat, Image.new("L", (W, W), 0), Image.new("L", (W, W), 0)))
        blend = Image.blend(im, red, 0.45)
        x, y = (s % 4) * W, 34 + (s // 4) * (W + 30)
        sheet.paste(blend, (x, y + 28))
        share = summ["per_frame_share"]["gradcam_vit_32x32"][s]
        d.text((x + 4, y + 2), f"slot {s} | frame {fi} | {fi / case['fps']:.2f}s", fill="black")
        d.text((x + 4, y + 14), f"share of total map mass {share:.2f}", fill="black")
    sheet.save(path)


if __name__ == "__main__":
    main()

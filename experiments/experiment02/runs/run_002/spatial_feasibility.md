# Stage D: spatial attribution feasibility (R-accident only)

**Outcome: technically implementable within the thesis budget, but not yet trustworthy.**

- **Feasible:** gradients of the auxiliary score reach input pixels and spatially indexed visual tokens. Four map types
  are computed, all finite, in about 1 s and 36 GiB.
- **Two warning signs:**
  - The maps put 23–35 % of their mass on the black letterbox bars, which carry no content and are 25 % of the input
    area. The CAM maps are roughly area-proportional there; the gradient maps over-weight the bars.
  - At frame level the maps disagree with the temporal perturbation results. All of them rank slot 0 highly, although
    replacing it removes no anomaly support.
- **Recommendation:** do not show these maps to participants. They are suitable only for further *technical*
  validation (see `NEXT_STEP.md`).

Every map here explains the **auxiliary label-only score S** (`scoring_contract.md`). None explains VERA's
explanation-first decision or VERA's final timeline.

## What was run

- **Script:** `experiments/experiment02/code/spatial_probe.py` (sha256 in `spatial/spatial_summary.json`). It is
  isolated: it imports the unchanged engine, builds its own differentiable forward with `torch.enable_grad()` only inside
  this script, keeps weights frozen (`requires_grad=False`, so 0 parameter gradients are stored), and writes only to `spatial/`.
- **Job:** `exp02_r2_D_20261022T202933_21188`: 16 s wall time, exit 0, on one RTX 6000 Ada.
- **Engineering time:** roughly 6–10 minutes of the 45-minute cap (agent's estimate). The work started at 20:28 EDT,
  the probe job ran 20:29:34–20:29:50, and the border check and write-up followed.
- **Precondition:** checked in code. The numeric scorer was not blocked (`provisional_post_hoc_gate`) and Stage B was complete.

## Attention backend and forward agreement

The backend is unchanged: FlashAttention 2 in both the InternViT encoder and InternLM2, the same as the baseline. Both
support backward passes, so no backend switch was needed.

| Check | Value |
|---|---|
| S, no-grad scorer path (Stage A route) | 2.7499999 |
| S, differentiable path | 2.75 |
| abs. difference | 1.2e-7 |
| gradient call time | 0.99 s |
| peak allocated GPU memory (one backward) | **36.4 GiB** (device peak 38,982 MiB). This fits only on the 46 GB RTX 6000 Ada cards, not on the 15 GB A16s. |
| parameter gradients stored | 0 |
| all maps and the ViT token gradient finite and non-zero | yes |

## Target tensors and spatial mapping

| Map (file) | Tensor | Spatial index |
|---|---|---|
| `gradcam_vit_32x32.npy` (8×32×32) | InternViT-300M last-layer output `vision_model(...).last_hidden_state`, CLS dropped (8 frames × 1024 patch tokens × 1024 channels) | patch token (i, j) ↔ pixels rows 14i…14i+13, cols 14j…14j+13 of the 448×448 input |
| `gradcampp_vit_32x32.npy` (8×32×32) | same tensor | same |
| `gradxinput_tokens_16x16.npy` (8×16×16) | projected visual tokens `mlp1(pixel_shuffle(...))` (8 × 256 × 4096), i.e. what the LLM receives | token (r, c) ↔ ViT tokens (2r…2r+1, 2c…2c+1) ↔ pixels rows 28r…28r+27, cols 28c…28c+27. This was measured on the model's own `pixel_shuffle` (ps_version v2) for all 256 tokens, not assumed. |
| `pixel_grad_abs_448.npy` (8×448×448, float16) | the normalized bf16 `pixel_values` | pixel |

All 8 × 256 visual tokens are written, in slot order, into the single `<IMG_CONTEXT>` block at `Frame1` (VERA's prompt layout).

Formulas, with `g = dS/dA` and `A` the patch-token activations:

- **Grad-CAM (ViT-token adaptation).** `w_c = mean_tokens g_c`; `map = ReLU(Σ_c w_c A_c)`. This is ordinary Grad-CAM
  with tokens standing in for a convolutional feature grid.
- **Grad-CAM++ weights.** These follow the closed form of Chattopadhay et al. (2018) as implemented in
  pytorch-grad-cam's `GradCAMPlusPlus`:
  - `α = g² / (2g² + Σ_tokens(A)·g³)` (α = 0 where g = 0);
  - `w_c = Σ_tokens α·ReLU(g)`;
  - `map = ReLU(Σ_c w_c A_c)`.

  **Adaptations and caveats.**
  - The closed form assumes the class score is passed through an exponential and that the network is piecewise linear,
    so that higher derivatives vanish.
  - InternViT (GELU, LayerNorm), the mlp1 projector (GELU) and InternLM2 (SiLU/SwiGLU, RMSNorm) all use smooth
    activations, softmax attention and normalisation, so that assumption does not hold. Only the formula is reused, on
    transformer tokens.
  - The target is the log-odds S, not a softmax class logit.
  - This is "the Grad-CAM++ weighting formula applied to ViT tokens", not a validated Grad-CAM++ for this architecture.
- **Gradient × input** on the projected visual tokens: `Σ_d g_d·E_d` per token (signed).
- **Absolute pixel gradient** `|dS/dx|`, summed over RGB.

## Diagnostics: what the maps do and do not show

### 1. Occlusion check

Grey out (RGB 127, before normalisation) 102 of 2048 28×28 cells, 5 % of all cells across the 8 frames.

| Cells greyed | S | S_hi |
|---|---|---|
| none (baseline) | 2.75 | 2.85 |
| top-102 by Grad-CAM | **−0.75** | **−0.66** (scorer class flips) |
| random 102, seed 0 | 1.75 | 1.96 |
| random 102, seed 1 | 1.75 | 1.96 |
| random 102, seed 2 | 3.00 | 3.01 |

Greying the cells this Grad-CAM ranks highest moves the auxiliary score far more than greying random cells. That is
*consistent with* the map tracking what the scorer uses, but it does not prove faithfulness:

- greyed cells are out of distribution;
- the top cells cluster in slots 0, 3, 4 and 5 (32, 14, 21 and 33 cells), while random cells are spread evenly;
- it is a single case.

### 2. Content-free regions

The source frames are letterboxed, so 25.0 % of each 448×448 input is near-black bars (`spatial/border_mass_check.json`).
Mass on those tokens:

| Map | Mass on the bars |
|---|---|
| Grad-CAM | 25.3 % |
| Grad-CAM++ | 22.9 % |
| |gradient × input| | 35.3 % |
| |pixel gradient| | 33.5 % |

- **CAM maps:** roughly area-proportional on the bars (Grad-CAM 25.3 %, Grad-CAM++ 22.9 %, against 25.0 % of the area).
- **Gradient maps:** they over-weight the bars (35.3 % and 33.5 %).

Either way, a large share of each map sits on regions with no content. ViT token artifacts are one untested possible
explanation; no token-norm or outlier-token analysis was run. This is a reason not to read these maps as object
evidence.

### 3. Frame-level disagreement with temporal perturbation

Per-frame share of map mass (one common scale, not per-frame normalised), slots 0–7:

| Map | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 | Slot 7 |
|---|---|---|---|---|---|---|---|---|
| Grad-CAM | 0.25 | 0.05 | 0.02 | 0.20 | 0.12 | 0.28 | 0.07 | 0.01 |
| Grad-CAM++ | 0.33 | 0.00 | 0.01 | 0.56 | 0.04 | 0.03 | 0.00 | 0.03 |

Stage B's single-frame replacement effects (S_hi Delta, blur / constant) are largest for slot 4 (+1.59 / +1.74),
then slot 3 (+0.74 / +0.80), and near zero for slot 0 (−0.13 / −0.14).

- **Slot 0 (69.7 s, before the collision):** every map ranks it first or second. Replacing it removes no anomaly
  support: S_hi rises slightly (Delta −0.13 / −0.14, within bf16 resolution) and VERA's label stays 1.
- **Slot 4, the frame with the largest replacement effect:** the maps differ.
  - Grad-CAM++ gives it little weight (0.04).
  - Grad-CAM gives it about the uniform share (0.12, rank 4 of 8).
  - The two gradient maps rank it near the top: |gradient × input| 1st (0.195) and |pixel gradient| 2nd (0.194).
  - So the disagreement that holds for every map is slot 0. For slot 4 it depends on the map.

Local gradients and frame replacement measure different things. Brightness or mass of these maps must not be read as frame importance.

### 4. Visual impression of `spatial/preview_gradcam_global_norm.png`

The agent viewed the preview, which uses one colour scale across all 8 frames:

- the map is diffuse and low-contrast;
- scattered cells sit on the road surface, the letterbox bars and parts of the trailer/collision region in slot 3;
- it is not concentrated on the truck–train contact point.

## Answer for the thesis plan

- **Can heatmaps be implemented within budget?** Yes.
  - One backward pass takes about 1 s and 36 GiB on an RTX 6000 Ada.
  - The engineering took about 15 minutes.
  - Maps for all 8 frames of a window cost one backward pass.
- **Are they suitable to show people?** **Not now.**
  - The maps explain the auxiliary scorer, not VERA's text decision.
  - They put content-free regions on a par with content.
  - They disagree with perturbation evidence at frame level.
  - They have no sanity-check validation, such as model or label randomisation.
- **Least costly next test.** On 5–10 windows:
  - crop or mask the letterbox before attribution;
  - compare Grad-CAM with occlusion maps;
  - run a randomisation sanity check (re-initialise the mlp1 projector).

  Ask for a new instruction before doing this; it was not run here.

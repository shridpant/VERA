# Right for the right reasons: grounding feedback in VERA's verbalized learning

Pre-registration for the grounding study. Written 2026-10-26 before any grounding measurement
was taken. The user asked for this design and its run; everything not stated by the user is my
call and marked (mine). Deviations go to `studies/firearm_grounding/DEVIATIONS.md`.

## Contents

- [Questions](#questions)
- [Setting](#setting)
- [Measures](#measures)
- [Phase 0: measurement checks](#phase-0-measurement-checks)
- [Phase 1: validity and gate](#phase-1-validity-and-gate)
- [Phase 2: training arms](#phase-2-training-arms)
- [Outcomes and analysis](#outcomes-and-analysis)
- [Compute and execution](#compute-and-execution)
- [Threats](#threats)

## Questions

1. **RQ1 (measurement).** When the frozen InternVL2-8B decides a VISILAB clip's category, does it
   look at the annotated firearm (attention), and does it rely on it (occlusion)? Do the two
   agree?
2. **RQ2 (training).** Does VERA-style question learning make the model rely more on the firearm
   if the optimizer is also told where the model looked? Accuracy must not drop by more than
   5 points (user: "prefer questions that pull attention onto the gun while keeping accuracy";
   the 5-point margin is mine).

## Setting

- Training 1 unchanged:
  - split `training1/data` (240/62/96, recording groups never span splits);
  - codebook and learner prompt;
  - 8 frames per video at `np.arange(0, n, n/8)`;
  - frozen InternVL2-8B 6fb9ad69, CUDA BF16, greedy, 1,024 tokens;
  - batch 2, 10 epochs, validation every 100 batches within an epoch.
- Boxes: Training 2's box files (audited origin 1, 0–1000 grid of the 448×448 input).
  - A sampled frame is **boxed** when its audit status is `positive`.
  - A video is **grounding-evaluable** when at least one of its 8 sampled frames is boxed.
  - No_Gun videos are never grounding-evaluable.
- Training uses training-split boxes only; validation boxes are used for selection; test boxes
  only for the final evaluation.

## Measures

Every measure is computed from the model's own greedy answer for a video, under the questions
being evaluated.

**Decision position.**
- The answer is cut at its last `Category:` field, keeping the whitespace the model wrote after
  the colon.
- The three category names are scored as continuations: summed token log-probabilities after
  the shared prefix, normalised over the three, with an FP32 head readout of the final hidden
  states.
- `p_true` = the normalised probability of the true category.
- The decision position is the last token before the category name. Answers without a
  `Category:` field get no score (counted as missing, never as 0).

**M1: attention lift (correlational).**
- One scoring forward over prompt + answer prefix, with eager attention in every layer, keeping
  only the decision position's attention row.
- Frame k's image tokens are the k-th run of 256 tokens in the single `<img>` span. Token
  (r, c) on the 16×16 grid covers rows [62.5r, 62.5(r+1)) and columns [62.5c, 62.5(c+1)) of the
  0–1000 grid; `ps_version` v2, checked in Phase 0.
- Box coverage per token = the fraction of the token's cell inside the box.
- For each boxed frame: a = attention over its 256 tokens (mean over all 32 layers and 32 heads,
  renormalised within the frame); inside = Σ a·coverage; chance = Σ coverage / 256.
- **Video lift = Σ inside / Σ chance over boxed frames.** 1.0 = chance.
- Primary aggregation: all layers and heads, fixed now.
  - Sensitivity: layers 0–10, 11–21 and 22–31.
  - Sensitivity: excluding each frame's single highest-attention token (sink check).

**M2: occlusion reliance (causal).**
- Two masked inputs per video, each regenerating the answer greedily from the same prompt:
  - **box-masked:** every boxed frame has its box filled with RGB (127, 127, 127), the runner's
    grey;
  - **control-masked:** an equal-sized box at a control position in the same frame. The
    position is the horizontal mirror (x → 1000 − x); if that overlaps the true box, the
    vertical mirror; if that also overlaps, the seeded non-overlapping position closest to the
    centre.
- **Reliance = log p_true(control-masked) − log p_true(box-masked).** Positive means masking the
  gun hurts the true answer more than masking an equal patch elsewhere.
- Also reported: label flips away from the true category under each mask.

**M3: behaviour.** Category, action-group and code accuracy (Training 1 parser, invalid = wrong).

**M4: explanation mentions.** Share of firearm-video answers whose explanation names a weapon
(`gun|pistol|handgun|rifle|firearm|weapon|machine gun`, case-insensitive), and share of No_Gun
answers that name one (false mentions).

**Right for the wrong reasons:** a grounding-evaluable video whose category is correct with
reliance ≤ 0.

## Phase 0: measurement checks

Before any grounding number is looked at:

- (a) Unit tests:
  - box → token coverage on synthetic geometry;
  - frame token runs located from input ids;
  - control placement rule;
  - candidate scoring on a fake model.
- (b) **Eager equals flash** on real inputs, for 4 videos:
  - the eager scoring pass's next-token argmax at every answer position equals the greedy
    answer;
  - its candidate log-probabilities match a FlashAttention scoring pass within 0.05 nats.
- (c) **Token-grid orientation.** Grey out one 16×16 cell region of a frame and confirm that the
  image tokens whose vision embeddings change most are at that cell on the assumed grid (top-1
  within one cell in ≥ 9 of 10 trials).

## Phase 1: validity and gate

Frozen model, seed questions, all 398 videos (train+val+test; no learning happens). Then the
Training 1 selected questions on the test split.

- **V1 spatial specificity.** Video lift with the true boxes vs with **shuffled boxes**: the
  boxes of another grounding-evaluable video of the same camera, frame by frame, permuted with
  seed 20261026. Paired difference with a recording-group bootstrap.
- **V2 agreement.** Spearman ρ between video lift and reliance over grounding-evaluable videos,
  with a bootstrap CI.
- **V3 behaviour link (descriptive).** Lift and reliance for correct vs incorrect categories.
- **Gate:** Phase 2 runs only if V1's difference CI lies above 0 and V2's ρ CI lies above 0.
  Otherwise attention is not a valid training signal here: stop and report RQ1.

## Phase 2: training arms

Three arms, each with three training-order seeds: 20261026 (Training 1's own run is T1 seed 1),
20261027, 20261028. Arms are paired by seed (identical video order).

| Arm | Optimizer sees | Selection among validated steps |
|---|---|---|
| **T1** | Training 1's prompt: predictions and targets | highest val category accuracy, then group accuracy, then earliest |
| **G** | Training 1's prompt plus a grounding block | among steps within 5 points of the arm's best val category accuracy, the highest val video lift (pooled over val grounding-evaluable videos), then category accuracy, then earliest |
| **P** (placebo) | the same block, values drawn at random | same rule as G, on true val lifts |

- **Grounding block (G).** Inserted after the targets, one line per video in the batch:
  - firearm videos: "Video i (target …): attention on the annotated firearm, 1.0 = chance:
    Frame3 0.4, Frame5 2.1, …; overall low/typical/high";
  - No_Gun videos: "Video i: no firearm in the frames".
  - low/typical/high = below the 25th, between, or above the 75th percentile of Phase 1's
    training-split lifts.
  - Then one fixed sentence: "Where there is a firearm, prefer questions that lead the model to
    look at the firearm itself, without lowering the accuracy of the category and action."
- **Placebo block (P).** Identical text, but each frame's lift is an independent seeded draw from
  Phase 1's training-split frame lifts. The overall level is computed from those draws.
- Because training feedback uses attention (M1) while the primary outcome uses occlusion (M2),
  questions that move attention without changing reliance cannot pass.
- No test box is read before the final evaluation.

## Outcomes and analysis

Test split, grounding-evaluable videos for M1/M2/right-for-wrong-reasons, all 96 videos for M3.

- **Primary H1:** mean reliance (M2) of G exceeds P. The effect is averaged over the three seeds
  as a paired, per-video mean difference. 95 % CI from 2,000 bootstrap draws resampling test
  recording groups within category (seed 20261026). Supported if the CI lies above 0.
- **Co-primary H2 (non-inferiority):** category accuracy of G ≥ T1 − 5 pp (lower 95 % bound of
  the paired difference above −5 pp).
- **Secondary:**
  - G vs T1 and P vs T1 on M1 and M2;
  - right-for-the-wrong-reasons rate;
  - M4;
  - action-group accuracy;
  - results by GT size tertile;
  - per-seed estimates (seed spread is shown, not hidden by averaging);
  - layer and sink sensitivity of M1.
- **Reference:** seed questions (untrained) on all measures.
- Selection uses validation only. No test number influences any choice.

## Compute and execution

- One idle RTX 6000 Ada, GPU-2931eba2 (oz index 6). GPUs 4 and 5 run the A/G/AG study and
  Training 1.
- Estimate:
  - Phase 0–1: about 2 h;
  - 8 training runs (2 × T1 at about 3.5 h; 3 × G and 3 × P at about 4.5 h with the attention
    pass);
  - test evaluation of 10 question sets at about 10 s per video;
  - about 37 GPU-h in total.
- Each run is resumable and logs every step.
- An independent review precedes Phase 2.
- Code: branch `experiment/firearm-grounding`, stacked on Training 1 (which branches from
  `feature/configurable-runner`). PR into `feature/configurable-runner` after the study succeeds.

## Threats

- **Small boxes:** median 0.64 % of the frame, about 1–4 tokens. Fractional coverage, and results
  reported by size tertile.
- **Attention sinks:** handled by renormalising within frame and the sink-exclusion sensitivity.
- **Attention ≠ reliance:** the reason for M2 and the V2 gate.
- **Loose, interpolated boxes:** unboxed frames never count.
- **Eight frames in one image block:** the model may mix frames. Per-frame token runs are exact,
  but the model's use of them is an empirical question.
- **Grey masks create edges:** controlled by the equal-sized control mask.
- **Confounds** (four actors, V1-heavy, one camera per room): the plausible source of
  right-for-the-wrong-reasons behaviour. Reported, not removed.
- **Small, path-dependent question learning:** three seeds and per-seed reporting.

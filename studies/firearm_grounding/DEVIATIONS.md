# Deviations from the pre-registration

Each entry: what changed, why, the evidence, and when (before or after which result).

## D1: category probabilities are read with FlashAttention; attention with eager (2026-10-26, after Phase 0, before Phase 1)

**Pre-registered:**
- One eager-attention scoring pass gives both the decision attention (M1) and the category
  probabilities (M2).
- Phase 0 checks (b2) every greedy token reproduced by the eager replay.
- Phase 0 checks (b3) eager and FlashAttention category log-probabilities within 0.05 nats.

**Phase 0** (`grounding1/phase0/phase0.json`, 4 validation videos):

| Check | Result |
|---|---|
| b1 generate == batch_chat | 4/4 |
| b2 eager replay reproduces the greedy tokens | 207/209, failed |
| b3 eager vs FlashAttention log-probabilities | max difference 0.120 nats, failed |
| c grid orientation | 9/10 within one cell |

**Diagnostic** (`diagnostics/phase0_diag.py`, `grounding1/phase0/phase0_diagnostic.json`):
- The two b2 mismatches ("in" and ",", video Handgun__PAH1_C1_P1_V1_HB_3, positions 14 and 29)
  appear identically in a FlashAttention replay of the same sequence.
- Their margins are −0.19 and −0.42 there, too: near-ties that flip between incremental greedy
  decoding and a one-pass replay, not an effect of eager attention.
- The 0.12-nat b3 gap is on Machine_Gun at p ≈ 0.002. For the chosen class the gap is at most
  0.0007 nats.

**Change:**
- M1 attention still comes from the eager pass; attention weights are not available otherwise.
- Every category probability (the decision, p_true, and the occlusion readouts) is read from a
  FlashAttention replay: the same kernels that generate the answers.
- The eager p_true is kept in each record (`eager_p_true`) for comparison.

**Effect on the plan:** none on the arms, the outcomes or the gate. Phase 1 had not run.

## Independent review before Phase 2, and the Phase 1 restart (2026-10-26, Phase 1 stopped at 109 of 398 videos)

An independent review of the measurement and arm code (launched before Phase 1 began) found the
issues below. Phase 1 was stopped, its 109 records were kept as
`grounding1/phase1_v0_partial_measurements.jsonl` (not used), and Phase 1 was restarted from zero
with the fixed code.

**What had been seen before the restart (disclosed).** While testing the report build I computed
the gate on the first 72 Phase 1 records (all training-split Handgun videos): V1 and V2 passed, and
the lift–reliance correlation within correct answers (ρ −0.03) and within incorrect answers
(ρ 0.01) was near 0. D2–D7 come from the review, not from those numbers. The descriptive
within-correctness and within-size correlations in the report (marked "not pre-registered") were
added because of what I saw.

### D2: V1 pairs true and shuffled lifts on the same frames
- Before: the shuffled lift pooled only the frames that both the video and its donor had boxed, and
  the true lift pooled all of the video's boxed frames. On validation, 22 of 45 videos differed.
- Now: `lift_shared` is the true lift on the donor-shared frames, and V1 = mean(lift_shared −
  lift_shuffled). This is the pre-registered "frame by frame" pairing; `lift` (all boxed frames)
  stays the M1 measure everywhere else.

### D3: the placebo draws one training video's frame lifts per video
- Before: each frame's value was an independent draw from the pooled training frame lifts. The
  low/typical/high level uses quartiles of per-video means, whose frames are correlated, so the mean
  of independent draws is too narrow. In the reviewer's simulation P said "low" about 3× less often
  than G at a within-video correlation of 0.6. G − P could then reflect how often each arm says
  "low", not whether the values are true.
- Now: for each firearm video, P draws one Phase 1 training video (seeded per step) and uses that
  video's frame lifts: without replacement, or with replacement when it has fewer boxed frames.
  Within-video correlation and the level mix then match G.

### D4: a decision needs a category name right after "Category:", tokenised as generated
- Before: any token after the colon was taken as the decision position. `Category: **Handgun**`
  would have been scored at the token before `**`, over continuations the model did not produce.
- Now: the decision is excluded (recorded as `decision_excluded`) when:
  - `no_name`: anything but one of the three names, written exactly, follows the colon and
    whitespace (`**`, `[`, `Pistol`, `Machine Gun`, …);
  - `token_mismatch`: the written name with its leading whitespace does not tokenise to the
    generated tokens.
- Each record keeps the written name, the lead and the candidate mass (the total probability of the
  three names). On the 109 stopped Phase 1 answers every name followed as " Name", so this guards
  the learned questions of Phase 2.
- The real tokenizer was checked: a newline lead tokenises as `:\n` + `Hand`, `gun`, which the
  scorer reproduces.

### D5: the sink-excluded lift removes the token from chance too
- Before: the most-attended token was removed from the attention but its coverage stayed in the
  chance term (over 256). A gun-focused top token inside a two-token box gave 0.50 instead of 1.0.
- Now: the token's coverage is removed, and chance is taken over the remaining 255 tokens. This is a
  sensitivity only.

### D6: smaller alignments with the pre-registration
- M4's regex now matches plurals (`pistols`, `rifles`, `handguns`), as the pre-registered pattern
  does. Denials ("not carrying any firearms") still count as a mention, as pre-registered; this is
  noted in the report.
- G and P both write "no decision to measure" under the same rule (D4 on the re-tokenised answer).
  Before, only G could write it.
- A firearm video with no boxed sampled frame is told "the firearm is not annotated in these
  frames" instead of "no firearm in the frames", which contradicted its target.
- The gate refuses to run (exit 2) when Phase 1 does not cover every video.
- Label flips are counted as in the pre-registration: the unmasked answer is right and the masked
  one is not.

### D7: clarifications of the implementation (unchanged behaviour)
- The control fallback is a deterministic scan in half-cell steps (the closest disjoint position to
  the centre), not a seeded draw.
- A frame without an equal-sized disjoint control is left out of both masks.
- G's overall level is the mean of the frame lifts, compared with quartiles of the training videos'
  mean frame lifts.
- The block wording follows `arms.py` (`BLOCK_HEADER`, the per-video lines and `SENTENCE`), which
  differs slightly from the design text.
- Phase 1 reads test boxes, as pre-registered for the gate. No training or selection choice reads
  them. The design's "test boxes only for the final evaluation" refers to Phase 2.
- Each control's distance to the firearm boxes is recorded (`control_gaps`). In validation 16 % of
  controls lie within one token cell of the gun. A control that covers part of the hand or gun pulls
  reliance toward 0 in every arm. The report adds H1 on videos whose controls are all at least one
  cell away (descriptive).

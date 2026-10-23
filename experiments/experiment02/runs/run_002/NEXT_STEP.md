# Recommended next experiment (needs a new instruction; not started)

## Pre-registered interval-level perturbation study on 20–30 UCF-Crime test windows

**Why this one.** experiment02 suggests that perturbing adjacent frames together can reveal decision-relevant evidence
that one-at-a-time perturbation misses. It also showed that ranked attribution beats random on a true accident window,
and that a likely false alert is fragile rather than attributable. All of this rests on 3 windows from 2 videos, so
the cheapest step that tests whether it generalises is the same measurement on a small stratified set, with
everything decided before running.

**Design sketch** (reuses `experiments/experiment02/code/` unchanged; about 2.5 GPU-minutes per window per family on one RTX 6000 Ada):

- **Windows (pre-registered list).**
  - About 10 windows that overlap an annotated anomaly interval, across 4–5 anomaly classes.
  - About 10 benchmark-negative windows from anomalous videos, including VERA false alerts found in released or fresh
    Step 1 outputs.
  - About 10 windows from normal test videos.
  - Each window keeps VERA's original 8 frames.
- **Masks.** The same 22 per family. Use the same blur and constant families.
- **Controls matched on evidence, not only size.** In experiment02, every label-flipping adjacent group had a larger
  summed single-frame effect than any control, so adjacency and "more evidence removed" were confounded.
  - Add nonconsecutive controls matched on summed single-frame effect. For example, pair the top-ranked frames
    non-adjacently ({3,5} or {2,4,6} on R-accident), chosen by a pre-registered rule from the single-frame results.
  - Add seeded random controls for slot coverage.
- **Targets.**
  - (1) VERA's original binary label and text.
  - (2) S_hi, pre-registered with the tie rule "exact tie = undecided", reported with the coarse bf16 Delta beside it.
- **Pre-registered outcomes.**
  - (a) The fraction of windows where some adjacent group flips the label while no single frame does.
  - (b) The same fraction for size-matched nonconsecutive groups.
  - (c) For windows with ≥ 2 distinct singleton S_hi values above resolution: ranked-vs-random removal and retention at
    budgets 2 and 4, plus cross-family τ_b.
  - (d) Label fragility: the share of masks that flip the baseline label. Report it by stratum.
- **Keep as diagnostics.** Text claim changes and the sound-question ablation.
- **Out of scope for this step.** No heatmaps and no participants.

## What remains before using explanations with people

1. **Generality.** The steps above. One accident window cannot support any interface design.
2. **Choose the attribution target.**
   - Either attribute VERA's own decision with interval-level label flips, which is coarse but honest,
   - or validate the label-only scorer more broadly (agreement with VERA's label, tie behaviour, prompt sensitivity).
   - In both cases it must stay labelled as distinct from the explanation text and from VERA's final refined timeline.
3. **Sparse evidence display.** The model sees only 8 frames per 10-s window, and windows overlap.
   - A numbered 8-frame strip with timestamps can mark the frames or intervals whose replacement changed the decision.
   - No dense per-frame clip score should be interpolated.
   - Human evidence access (full video vs the 8 frames) must be stated per condition.
4. **Claim grounding.** A human audit of text claims is needed.
   - In the agent's inspection of the 40 input frames (not a human audit), "visible damage to the train" was not
     assessable at 320×240, and R-end's "no visible damage" appeared contradicted by debris.
   - Sound claims were made without audio.
   Removing or rewording guiding question 5 for image-only input should be considered and pre-registered. Here it
   removed the R-pedestrian false alert.
5. **Heatmap validity**, if heatmaps are to be shown at all:
   - mask or crop the letterbox bars;
   - run randomisation sanity checks;
   - compare against occlusion maps on 5–10 windows.

   Until then they are NO-GO for participants.
6. **Fragility reporting.** Alerts like R-pedestrian flip under almost any change. A study of appropriate reliance
   should decide whether and how to show such instability. That would be a new manipulation, not part of this run.

# Current status — October 26, 2026

Experiment03B full_v1 completed: 290 videos / 433 inputs. Original primary +15.166 pp [9.860, 20.600]. Original validation 16/20 remains preserved; the explicit post-data R1 integrity audit passes 20/20 without model reruns or changed results. See [R1 amendment](experiments/experiment03B/R1_AMENDMENT.md). The earlier not-launched/planned sections below are historical.

---

# VERA research: ongoing notes

Updated 2026-10-23. Each section marks what has been **executed** and what is **planned**. The registry is
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Objective

- **Appendix study.** A thesis appendix validates a frame-ranking method for local, VLM-based video anomaly
  judgments (VERA).
- **Main contribution.** The thesis's main technical contribution is developed separately.
- **Later.** An exploratory human study will ask how AI predictions, textual explanations and temporal importance
  affect people's classification and their choice of evidence frames.

## Motivation (Appendix D)

- **Earlier study.** 22 participants watched five RWF-2000 clips of 32 numbered frames each, judged physical
  aggression, and marked supporting frames.
- **What it showed.** People agreed that an event occurred, yet selected different evidence: median selection
  lengths of 22.5, 24 and 27 frames on the fight clips.
- **Implication.** Agreement with a human consensus does not show model faithfulness. The new work extends
  evidence-selection questions to UCF-Crime events and to exposure to AI explanations.

## VERA and the two targets

VERA (CVPR 2025) works in three steps:
1. A frozen InternVL2-8B answers learned guiding questions about **8 sampled frames** per 10-second window, giving
   an explanation and a 0/1 label.
2. The local labels are refined with ImageBind similarity.
3. They are smoothed, rounded and position-weighted into a timeline.

This work targets the local judgment, not the refined timeline. There are two targets:
- **Original VERA**: the explanation-first prompt plus the strict 0/1 parser. This is the primary behavioural
  outcome.
- **Auxiliary scorer** (`label_only_first_token_v1`): a label-only prompt giving the margin of the first-token
  log-odds.
  - `S_hi` is the FP32 readout of the frozen BF16 head and is used for ranking.
  - The native margin is `logit_1 − logit_0`, the exact BF16 logits.
  - The scorer's agreement with VERA is measured, never assumed.

## Methods: established versus experimental

- **Temporal occlusion sensitivity.** Replace declared frames and measure the change in output. This is an
  established principle (Zeiler & Fergus; Fong & Vedaldi). The systematic enumeration used here is **not** RISE,
  and it carries no novelty claim.
- **Fixed ranking** (primary). Importance is computed once on the unperturbed input. Prefixes are always built from
  the original pixels.
- **Adaptive ranking** (secondary). Conditional singleton effects are recomputed after each removal, up to k = 4,
  with `q` locked to the baseline class. It costs extra queries, so it is not an equal-cost competitor.
- **Replacements.** Blur (radius 17.92 at 448×448) and grey RGB 127 are study settings, and both shift the input
  distribution.

## Executed

| Date | What | Key outcome | Artifacts |
|---|---|---|---|
| 2026-10-22 | experiment01: one-video reproduction (RoadAccidents127) | 162/162 windows parsed. Frame AUC 0.995 on one video. run_002 reproduced run_001 exactly. | `experiments/experiment01/` |
| 2026-10-22 | experiment02: temporal attribution feasibility (3 primary windows) | **Accident:** 0/16 single-frame flips, 7/26 adjacent pair or triple flips. **Pedestrian:** a fragile alert (13/16 single-frame flips); removing the sound question also flips it. **Normal control:** no flips. Original and scorer agreed on 225/255 distinct inputs. | `experiments/experiment02/runs/run_002/` |
| 2026-10-23 | Erratum E1 | A sub-0.25 change in `S_hi` still moved the native BF16 score in 36 of 109 rows. There is no "invisible below 0.25" rule. | `experiments/experiment02/ERRATA.md` |
| 2026-10-23 | experiment03A protocol (30 windows) | Draft v0.4 after three review rounds. Not run. | `experiments/experiment03A/` |
| 2026-10-23 | experiment03B implementation and integration | Protocol v1.1. CPU suite 17/17, validator 37/37, independent validation 20/20. Integration: 84/100 new inputs, about 0.08 GPU-h; fresh recompute byte-identical to experiment02. Development only. | `experiments/experiment03B/IMPLEMENTATION_REPORT.md` |

## Planned: experiment03B full run (not launched)

| Parameter | Setting |
|---|---|
| Population | All 290 UCF-Crime test videos (13 categories plus Normal). 162 event-centred, 121 benchmark-negative (19 unavailable) and 150 normal-video windows, making 433 unique inputs. |
| Excluded from confirmation | The development videos RoadAccidents127 and Normal_Videos_018; they are kept in coverage and a sensitivity estimate |
| Primary endpoint | `d_w` = grey flip of the blur-`S_hi` top two minus the mean flip over all 28 pairs, averaged within video, then over videos. Video-cluster bootstrap, 10,000 resamples, seed 0. |
| Secondary | Grey numerical margin; fixed vs adaptive prefixes at k = 1..4; blur/grey and native/`S_hi` agreement; pair non-additivity; normal-video artefact flips; no-sound diagnostic; reciprocal check |
| Cost | About 12–15 core GPU-hours on one RTX 6000 Ada. 7.96 GB of sequential downloads to local scratch. About 0.45 GB in home. |
| Seeds | Planned [0, 1, 2]; executed [0]. Seeds change bootstrap draws only. |

The launch command is in `experiments/experiment03B/IMPLEMENTATION_REPORT.md` §5.

## Limitations

- **Distribution shift.** Blur and grey replacements shift the input; deletion tests can be distorted
  (Hooker et al. 2019; Wang & Wang 2024).
- **Labels and windows.** UCF intervals are benchmark labels, not human evidence, and event-centred windows are
  oracle-localised.
- **Small classes.** Some classes have only 2–3 videos.
- **Correlation.** Windows from one video are correlated, and masks and seeds are not independent samples.
- **Scorer.** The auxiliary scorer is not VERA's explanation-first decision.

## Decisions and next step

1. **Next step: decide whether to launch the experiment03B full run**, using the commands in the implementation
   report. Run retrieval first, then the model stages on one Ada.
2. **Report.** After the run, fill Table A1 from `results/` only (`appendix_table.py`). If the ranked pair does not
   beat the exact random-pair reference, report the null result.
3. **Human pilot.** Select its clips by a recorded rule, not because this appendix produced dramatic effects.

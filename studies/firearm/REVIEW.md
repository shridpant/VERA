# Independent pre-run review

A fresh AI reviewer read the handoff, the diff from e13acce to becc9b5 and the pilot v2 artifacts
(read-only, no network). It checked the handoff's section 10 list: scope, coordinate and frame
validity, leakage, denominators, identical inputs and parser robustness. This is an AI review,
not a human signature.

## Contents

- [Clean checks](#clean-checks)
- [Findings and resolutions](#findings-and-resolutions)

## Clean checks

The reviewer reproduced these and found no problem:

- the YAML → JSON conversion and the prompt-hash lock;
- AG minus its two insertions = A, byte for byte;
- no category, annotation or ID in any prompt;
- each call: no history, the same image object, the same generation settings,
  `num_patches_list=[8]`;
- one 2,048-token image run per condition;
- A identical to `run_vera.py` on 12/12 pilot windows;
- 6,375 windows = Σ ceil(n/16), with 8 distinct in-range frames each;
- the PAH7 boxes overlaid on the real 448×448 inputs sit on the handgun;
- truncation and BOS handling;
- the resume, input-hash and spec refusals.

## Findings and resolutions

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | major (validity) | Packing makes slot↔frame and per-frame coordinates unverified; pilot slots all `[]` | Kept: the handoff fixes the packing. Stated as an unverified assumption in the report. T7 counts answers whose boxes do not vary by frame or are whole-image only (mosaic signs). |
| 2 | major | A torn JSONL line makes the next append fuse with it, so the run can never be read again | Fixed in 5cc03da: `ledger.seal` moves a torn tail to `.torn.N` before appending. Tests reproduce the scenario. |
| 3 | major | Storage outages charge attempts to every unit, and the run exits 0 when all are exhausted | Fixed in 5cc03da: unreachable staged videos stop the run before any attempt (exit 2). Decode-only failures end as `preprocess_failed`. A finished run with failed units exits 3. `full_run.sh` stops on 0, 2 or 3. |
| 4 | major (AG only) | `FrameK:` lines in AG free text could duplicate real box lines | Fixed in 5cc03da: only the block after the last `Firearm boxes:` heading is parsed. Lines outside it are counted. |
| 5 | minor | Legacy label computed on truncated text; no parser-disagreement table | Legacy label kept as the original rule would act (it cannot detect truncation). New T2b reports agreement on complete, non-truncated answers only. |
| 6 | minor | No relaunch job yet | `jobs/full_run.sh` (16529db), updated for exit 3 in 5cc03da |
| 7 | minor | Model hash ignored library versions | Fixed in 5cc03da: versions of torch, transformers, flash-attn and others are part of the model hash and saved per session |
| 8 | minor | validate did not check the GPU | Fixed in 5cc03da: exactly one GPU by UUID, an RTX 6000 Ada, with no running process |
| 9 | minor | Rendered inputs did not collapse spaced image tokens | Fixed in 5cc03da (`<IMG_CONTEXT>×2048`) |
| 10 | minor (analysis draft) | T8 had a CI for recall only. "All" precision pooled No_Gun boxes. No-record slots mixed into G-vs-AG. "invalid" meant different things in T2 and T3. T2 ignored some ledger statuses. Refinement coverage was not declared. | T8: paired AG−G intervals for recall, precision (boxed frames) and the No_Gun box rate, on slots executed in both. Precision is defined on boxed frames, with an explicit `precision_incl_no_gun_frames`, and `no_record_slots` is reported. T2 uses `unparsed_labels` and every end status; T3 uses `not_valid` with a breakdown. Refinement coverage is declared below. |
| 11 | nit | The −(1+IoU) cost does not guarantee the most matches | Fixed in 5cc03da: weight min(#pred, #gt)+1. The reviewer's 5×5 case is a test. |

## Declared before the full run

Refinement needs at least 15 windows. 120 of 398 videos have fewer (Handgun 35,
Machine_Gun 44, No_Gun 41) and are excluded from T6/P4. A video with any invalid A or AG label
is excluded for that condition. So T6 covers at most 278 videos.

The pilot predicts near-zero G and AG localization under the fixed packing. T8 is reported as
measured.

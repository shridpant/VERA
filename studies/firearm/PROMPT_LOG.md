# Prompt log (G and AG)

Every prompt version and the pilot evidence that changed it. Pilots use six videos drawn with
seed 20261025 (one per category × camera) and their first two windows:
Handgun PAH7_C1_P1_V1_HB_2 and PAH9_C2_P3_V1_HB_2, Machine_Gun PAW8_C1_P2_V1_HB_2 and
PAW5_C2_P3_V1_HB_2, No_Gun N8_C1_P2_V1_HB_4 and N7_C2_P3_V1_HB_1. These videos stay in the
full run (full inclusion) and are flagged as pilot videos in the per-video table.

Revisions fixed **format** failures only. No localization score (IoU, recall or precision) was
computed on any pilot output before the prompts were locked. Pilot results still shaped the
prompt wording, so the study is reported as **exploratory**.

## Contents

- [Packing](#packing)
- [v1](#v1)
- [v2](#v2)

## Packing

The pinned InternVL2-8B `batch_chat` replaces only the first `<image>` with all eight frames
(`<img>` + 8 × 256 `<IMG_CONTEXT>` + `</img>`). `Frame2: <image>` to `Frame8: <image>` stay literal
text. A has always run this way, and G and AG keep it. So "FrameK" is a text label over one
image block, and whether the model can tie it to the K-th image is an empirical question. The
pilot's saved input ids confirm this: each condition has one `<img>…</img>` span holding a single
run of 2,048 `<IMG_CONTEXT>` tokens. The inputs are A 2,487 tokens, G v1 2,444 and AG v1 2,713. The
slot check (`slot_check.py`) tests it: it keeps one slot, greys the other seven and asks whether
boxes appear on that frame only.

## v1

Files at commit 57a45a8:

| Prompt | sha256 |
|---|---|
| A | f3d5e7c9894c852f761251cb9451786ee0a5464dc600a66c0587c375dd120924 |
| G | 674436df6ed3c6152d4d6256295efa60e200733073e9e5cd97acaf0527b28ff2 |
| AG | 5176785ced437738f9e8132f5191efd2aa06b04eeaea9a6d2703e039fe05b441 |

- G: the grounding paragraph, then the Input block, then a one-line format placeholder
  `[One line per frame, Frame1 to Frame8, as described in Task.]` under `Firearm boxes:`.
- AG: A plus the grounding paragraph inserted as question 6, before the concluding instruction.

Pilot run `firearm_pilot_v1` (Oz, GPU-6a490dc2, 12 windows × 3 conditions, all complete with EOS):

- G answered `Firearm boxes:\n[]` in 12/12 windows. The model copied the single bracketed
  placeholder line, so there were no FrameK lines (parser: all slots `missing`).
- AG gave VERA's two-field answer with no box lines in 12/12 windows. The original output
  format, which ends "Please ONLY reply according to this format", left no place for boxes.
- The slot check with the v1 G prompt gave `[]` for every variant, so it was uninformative.
- A: 12/12 parsed, about 50 generated tokens and 1.5 s per window.
- A preservation: `run_vera.py` on the same 12 windows gave byte-identical raw text, identical
  input RGB hashes and identical strict labels (`a_preservation.json`, all_identical = true).

## v2

The decision rests on the format failures above. Both outputs get the same explicit
eight-line block (`box_format.txt`):

```
Firearm boxes:
Frame1: [boxes for Frame1]
...
Frame8: [boxes for Frame8]
```

- G: this block is the whole output format.
- AG: the block is inserted into VERA's output format between `Answers to Prompt Questions`
  and the final `Output:` field. AG = A + two declared insertions; the final Output field and
  everything after it are byte-identical to A.

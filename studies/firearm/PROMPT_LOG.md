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
- [v3 probe](#v3-probe)
- [Lock](#lock)

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

| Prompt | sha256 |
|---|---|
| A | f3d5e7c9894c852f761251cb9451786ee0a5464dc600a66c0587c375dd120924 (unchanged) |
| G | 1fba432b24ab4569fff8a6e41a1de4a8160aa3cc784e9f95152182256dd63a4d |
| AG | 03dbf44a69c166a9ca7c6c3e68d84b78de472d8e5036c5bd139140114374b1c8 |

Pilot run `firearm_pilot_v2` (same windows; 12 × 3 calls, all complete with EOS, none truncated):

- G: 12/12 answers had the heading and all eight lines. Every slot was `[]` (96/96, 37–41 tokens).
- AG: 12/12 valid labels (all 1), the heading and all eight lines, every slot `[]` (96/96);
  62–145 tokens. The answers mention the firearm in text (e.g. "He is holding a firearm and
  appears to be aiming it") while every box line is empty.
- A: 12/12 valid (11 × 1, 1 × 0); A preservation again all identical.
- Mean seconds per call: A 1.62, G 1.22, AG 2.16.
- Slot check (G v2): keeping slot 0, 3 or 7 of two Handgun windows and greying the other seven
  gave `[]` on every line (6/6), as did the all-grey control (2/2). The model reported no firearm
  even with one real frame visible, so slot interpretability could not be shown either way.

## v3 probe

`prompt_probe.py` (`jobs/probe.sh`, result `firearm_pilot_v2/probe_v3.json`) asked G and AG on
five pilot windows (both Handgun and Machine_Gun videos and N7_C2, anchor 16) with three
prompt folders. This was a format check only; no localization score was computed.

| Folder | Change | G | AG |
|---|---|---|---|
| prompts (v2) | control | 5/5 all `[]` | 5/5 all `[]` |
| v3a | InternVL2-native `<ref>firearm</ref><box>[[x1, y1, x2, y2]]</box>` in the paragraph and lines | 5/5 all `[]` | 3/5 all `[]`; PAH9_C2: a full-frame `[[0, 0, 1000, 1000]]` on Frame1, then repeated non-English tokens to the 1,024-token cap (no label); PAW8_C1: the same unbracketed `380, 400, 420, 460` on all eight lines (malformed) |
| v3b | plain lines with a `[[x1, y1, x2, y2]]` template | 5/5 all `[]` | 3/5 all `[]`; PAH9_C2 and PAW8_C1: `[[0, 0, 1000, 1000]]` on all eight lines |

Neither change produced frame-specific coordinates. Both added malformed, truncated or
whole-image answers.

## Lock

**v2 is locked** (`conditions.prompt_sha256` in the study YAML; any other text is refused):

- It is the only version that the model follows cleanly in both G and AG: all lines parse,
  nothing is truncated, nothing is degenerate.
- More rewording would be tuning on box presence.
- The handoff fixes the eight-image packing, so a single-image condition is out of scope.

The full run therefore measures the frozen backbone under VERA's packing with a
format-compliant prompt. The pilot predicts that G and AG will rarely localize; that is
reported as a result, not repaired.

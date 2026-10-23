# Scoring contract: experiment02 run_001

Contract id: `label_only_first_token_v1` (`attr_lib.SCORING_CONTRACT_VERSION`). Code: `experiments/experiment02/code/engine.py`
(`Engine.score`, `capture_label_only_inputs`, `setup_candidates`, `lm_logits`, `score_from_inputs`, `alignment_check`)
and `attr_lib.py`. The exact sha256 of each code file is in `run_manifest.json` (written at analysis time) and in the Mac mirror.

## Three targets, kept apart

| # | Target | Prompt | Output used | Where recorded |
|---|---|---|---|---|
| 1 | **Original VERA local prediction** | VERA learner instruction + the 5 released UCF questions, verbatim. This is the question experiment01 used, byte-identical (validator V14). | Raw explanation text + strict-parsed 0/1 label (`vera_pipeline.strict_parse_label`; unparseable → missing) | `target = original_generation` |
| 2 | **Instrumented class scorer** (auxiliary) | **Modified** label-only prompt (below) | `S = log P(answer=1) − log P(answer=0)` at a fixed reply boundary; separately, label-only greedy generation | `target = label_only_score`, `label_only_generation` |
| 3 | Final VERA timeline | — | **Not attributed in this experiment** (it mixes other windows through ImageBind features and numerical refinement) | — |

Target 2 is a different inference condition. It explains how the label-only scorer responds, not how VERA's
explanation-first decision is made, and not VERA's final scores. S is a log-odds between two tokens under one prompt.
It is not a calibrated probability or uncertainty.

## Prompts (exact files in `results/prompts/`)

- `original_question.txt`: the question VERA's `forward()` assembles for 8 frames. Frames are referenced as
  `Frame1: <image>` … `Frame8: <image>`. InternVL's `batch_chat` replaces only the *first* `<image>` with
  `<img><IMG_CONTEXT>×2048</img>`, i.e. 8 × 256 image tokens in slot order; `Frame2…8: <image>` stay as literal text.
  This is upstream behaviour and is kept unchanged. The chat template adds InternVL's fixed Chinese system message.
- `label_only_instruct.diff`: the **only** edits to `VERA_learner_instruct.txt` for target 2:
  1. "Please answer the prompt questions." → "Please consider the prompt questions."
  2. "Answer the following questions … and provide an explanation in one sentence." → "Consider the following questions based on what you see from the video frames."
  3. The format block loses the lines `Answers to Prompt Questions: [...]` and `Output:`. Only `[ONLY the integer class label; make necessary assumptions if needed]` remains.

  All five guiding questions, the frame placeholders, the system message, image preprocessing and image order are unchanged.
- `nosound_questions.diff` (Stage A3 only): the original prompt with question 5 ("unusual sounds or noises")
  removed; questions 1–4 are byte-identical.

## How S is computed

1. **Query construction.** VERA's own `forward()` builds the label-only question, and InternVL's `batch_chat` builds
   the chat query and token ids. The engine intercepts the `generate()` call, so nothing is generated, and keeps the
   exact `input_ids`, `attention_mask` and bf16 `pixel_values`.
2. **Forward pass.** One forward pass mirrors `InternVLChatModel.generate`'s embedding injection:
   `extract_feature(pixel_values)` → 8 × 256 visual embeddings written into the `<IMG_CONTEXT>` positions →
   `language_model(inputs_embeds=…, attention_mask=…, use_cache=False)`. The weights are frozen
   (`requires_grad=False`) and `torch.no_grad()` is used.
3. **Boundary.** The next-token distribution at the last query position, i.e. the start of the assistant reply right
   after `<|im_start|>assistant\n`. No assistant prefix is supplied. In particular, no previously generated explanation
   is used, so no earlier decision is carried into a perturbed run.
4. **Candidates.**
   - The query is re-tokenized alone and with `"0"` / `"1"` appended. The query part must be prefix-stable; the
     appended tokens are the candidates.
   - If both candidates are single tokens, `S = log_softmax(logits)[id("1")] − log_softmax(logits)[id("0")]`.
   - Otherwise each candidate's full continuation log-likelihood is summed by teacher forcing.
   - Only label tokens are scored. Prompt and image tokens never enter S, and the end-of-turn token is not scored
     (termination policy).
5. **Recorded per score:** `logp_0`, `logp_1`, `p_0`, `p_1`, the candidate mass `p_0 + p_1`, the top-5 next tokens,
   the model-input hash, the per-slot pixel-tensor hashes and the runtime.

Measured tokenization and alignment results: see `results/scorer_diagnostics.json` and the section below (filled from that file after Stage A).

## Effects

- **Delta.** `Delta(G) = S(original window) − S(window with slot set G replaced)`.
  - Delta > 0: the replaced frames supported the scorer's **anomaly** preference.
  - Delta < 0: they supported **normality**.
  - For a window the scorer calls normal (S < 0), the columns `support_for_scorer_baseline_class = sign(S0) · Delta`
    separate "evidence for the predicted class" from "evidence for anomaly".
- **Binary.** `binary_diff = label(original) − label(perturbed)` on the original-prompt label. A missing (unparsed)
  label stays missing.
- **Interaction.** `Delta(G) − Σ_{g∈G} Delta({g})`: non-additivity on the S scale only. Saturation and thresholding can
  produce it; it is not evidence of causal synergy.
- **Frame heuristics.** These are labelled heuristics, not validated attribution rules.
  - `group_pooled` (the brief's rule): mean of `Delta(G)/|G|` over the adjacent pairs and triples containing the frame.
    It has a boundary artifact: under a constant group effect it scores the middle slots 0.400 and slots 0, 1, 6, 7 0.417.
  - `group_size_balanced`: the same terms averaged within each group size first, then across sizes. It is flat under
    the constant fixture, and is used for the Stage C group ranking (chosen before data).
  - Coverage is unequal: edge slots are in 1 pair and 1 triple, slot 1 and slot 6 in 2 and 2, slots 2–5 in 2 and 3.

## Outcome in run_001 (v1 rules)

Stage A under v1 classed the numeric scorer as **blocked**. Exactly one technical criterion failed: "sign(S) equals
the label-only generation label on all 5 cases".

- On R-pedestrian both candidate logits were exactly 36.0, so S = 0.
- The v1 tie rule maps S ≥ 0 to class 1; greedy generation broke the tie toward token 300 ('0').

Every other criterion passed:

- exact alignment with `generate`;
- exact repeatability;
- candidate mass 0.82–0.91;
- all six all-eight replacements moved S by 0.75–3.6.

Stages B and C were not run in run_001. See `../run_002/scoring_contract.md` for the disclosed v2 rules.

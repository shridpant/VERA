# Scoring contract: experiment02 run_002 (config v2)

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

## Measured in Stage A (run_001 and run_002 agree)

| Item | Result |
|---|---|
| Query reconstruction | the rebuilt chat query tokenizes to exactly the ids `batch_chat` passed to `generate` |
| Boundary | `...don't give me any other words.\n<|im_end|><|im_start|>assistant\n` + candidate |
| Candidates | `"0"` = token **300**, `"1"` = token **312**; single tokens, prefix-stable; the same ids as tokenized alone |
| Alignment | on all 5 cases, the scorer's boundary logits equal `generate(output_scores=True)` step-0 scores (max abs difference 0.0); argmax and generated token agree |
| Repeatability | identical input scored 3 times: identical S on every case (deterministic) |
| Candidate mass | P('0') + P('1') = 0.82–0.91 on the 5 unperturbed cases; other mass goes to `[`, ``` `` ```, `Yes` |
| Resolution | candidate logits sit near 33–37, where bf16 values are 0.25 apart, so **S moves in steps of 0.25 nats** while both logits are ≥ 32. The all-grey inputs, where logit('1') = 31.625, fall on a 0.125 grid. |

## S_hi: the finer readout (v2, post hoc)

The bf16 head output rounds each logit to 0.25 here, so many perturbation effects would collapse to 0 or ±0.25.
`S_hi` uses the same model computation:

- the final normed hidden state `h` (bf16), captured by a hook on the language model's `output` head;
- the same bf16 head rows for tokens 312 and 300;
- the dot product `w·h` kept in fp32 instead of rounded to bf16, giving `S_hi = w_312·h − w_300·h`.

Every score record also checks that each fp32 logit rounds back to the model's own bf16 logit (within half a bf16
step plus 1e-3), in `S_hi_rounds_to_model_logits`. S_hi therefore changes no model input, weight or precision. It
only stops rounding the final two-row readout.

S_hi differences below 0.25 nats are real changes in the hidden state, but they cannot change the deployed bf16
output probabilities. Every effect row reports `Delta` (on S_hi, primary for effects, interactions and rankings)
**and** `Delta_bf16` (on S), together with the flag `Delta_below_resolution`.

## Gate changes in v2 (post hoc, after seeing run_001 Stage A)

- An exact tie `S == 0` (identical bf16 logits) counts as **undecided**, not as agreement or disagreement.
  - Under v1, R-pedestrian (logits 36.0 / 36.0) failed "sign(S) = label-only generation". The tie rule sent S = 0
    to anomaly, while greedy decoding picked token 300 ('0', the lower id).
- `post_hoc_amendment: true` caps the scorer status at `provisional_post_hoc_gate`.
- run_001 (Stage A under v1, scorer "blocked") is kept unchanged as the pre-declared outcome.

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

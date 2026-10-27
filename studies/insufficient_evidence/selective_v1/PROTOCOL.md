# Frozen selective-explanation pilot v1

Freeze and push this protocol, runner, metrics and input/reference hashes before inference. No prompt revision after seeing outputs; preserve all failures. Supersedes the old conditional learning proposal for this training-free diagnostic only.

## Inputs and scope

Reuse all 24 processed inputs from proxy_v1, unchanged: 12 previously consulted UCF test sources, one annotation-included and one unannotated-gap five-second/eight-frame view each. No optimization, source selection or training from these outputs. The source IDs were originally hash-selected without response outcomes; current selection is a deliberately reused diagnostic, not a pristine test. All three variants receive identical slot tensors. Original VERA's published ten-second/full-video fusion protocol is not reproduced. No XD experiment or full-dataset detection accuracy claim.

## Three conditions

A `original`: original binary learner instruction and original five guiding questions, unchanged.

B `manual`: same model, questions and binary 0/1 task. Explain only visually observable details; explicitly list unresolved details instead of guessing. Keep a best binary decision even when details are unresolved. Emit a small JSON explanation with observations (up to three), empty frame arrays, and unresolved details. No label 2.

C `cited`: exactly B plus require each emitted observation to cite one or more supplied frame slots (1–8). If it cannot cite support, put the detail in unresolved instead. Same JSON schema as B. No additional verifier, external judge or evidence retrieval. A citation is model-proposed, not certified evidence.

Both B and C remove the assumption permission and replace the final yes/no wording with explicit binary classification plus a separate unresolved-detail channel. A retains its original free-text answer format; B/C share the structured format. Consequently A-versus-B includes a formatting/instruction bundle, while B-versus-C more narrowly isolates requesting citations. Neither B nor C forces its label to equal A: preservation is measured, not guaranteed by copying a decision. Separate model calls, no conversation history or captions, categories, timestamps or previous answers supplied.

## Execution and budget

72 calls = 24 inputs × 3 conditions, one shared frozen InternVL2-8B checkpoint, BF16/flash attention, greedy max1024 tokens, no parameter updates, no verbalized question updates, no optimizer calls. Rotate condition order by source index; reverse view order on alternate sources. Original baseline is rerun and compared to the previous run as a determinism audit. One empty Ada5 only, selected by UUID; never Adas6/7 or the occupied A16s. Check memory/process occupancy immediately before model load. Timeout1800s, no automatic retries. Budget including previous pilot:120/192 calls, elapsed allocated GPU time remains below1hour. Stop on pixel/prompt mismatch or invalid binary decision; malformed explanation JSON is a measured failure retained without retries.

## Frozen descriptive endpoints

* By condition and view: binary alert counts, decision agreement and original-alert retention relative to current A; show all24 paired decisions. Retention is not accuracy.
* B/C: valid explanation schema, number of observations, nonempty unresolved channel, alert plus unresolved details, legal/nonempty frame citations. Original JSON metrics are not applicable, never scored as zero failures.
* Positive-reference lexical coverage on the12 annotated inputs: lowercase ASCII words of length≥3, minus the fixed stopword list in analyze.py, set recall against all UCA captions whose timestamp equals the selected containing interval. Use A's answer text excluding the final Output field; B/C's observation claim text only. Save hashed reference text and derived word sets; never pass them to the model. This is content-word overlap, not entailment, event correctness or a hallucination metric. Negation, synonymy, length and unequal formats confound it. Zero-overlap/unmatched claims remain unverified. Do not label gap claims false.
* No citation/timestamp alignment score: all annotated-view frames lie inside the selected interval, so such a score would be structurally trivial; gap views have no positive target. Legal citations alone do not demonstrate support.
* Parse/cap failures, exact prompt hashes, frame and slot hashes, dependencies, runtimes, baseline repeat agreement. No p-values/population claims or automatic learned extension.

## Decision rule

If outputs are parseable and B/C produce alert-plus-unresolved responses while retaining A alerts, conclude only that the two channels can coexist operationally. C must show more than valid syntax before it is called better grounded. Positive lexical coverage may expose loss of reference content but cannot establish faithfulness. If semantic correctness remains unidentifiable or the manual control explains the behavior, stop compute and report the limitation; do not optimize the proxy or present the prompts as novel. Further research requires a differentiated mechanism and stronger existing evidence-linked labels/data, not necessarily new annotation.

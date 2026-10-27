# Research decision: no-go for current evidence-verified question-learning proposal

Decision date: October 27, 2026. Scope: the investigated original-VERA evidence-feedback direction and its temporal/caption variants. This is a resource-allocation decision under current evidence, not proof that future novelty is impossible.

## Concrete candidate assessed

Keep original InternVL2-8B frozen. Learn guiding questions with feedback from temporally aligned UCA descriptions. Compare classification-only feedback, unchanged VERA, no-sound/manual prompts and caption-informed feedback, counting all optimizer/verification calls. Use original VERA split boundaries, exclude test IDs, and report caption supervision explicitly.

## Why this candidate is not ready for compute

1. A caption-reference score measures agreement/coverage of selected descriptions. It does not verify arbitrary generated claims. Unmentioned details may be true; a fluent paraphrase may conceal unsupported causality. Neither text overlap nor an unvalidated model judge supplies the requested evidence verification.
2. Existing VERA/PrismVAU/GEPA cover question/prompt optimization and reflective feedback; LRPO covers linguistic learning with temporal feedback; VANGUARD uses UCA-informed reasoning/grounding. Switching the reference from class labels to captions is not by itself a defensible contribution. No specific mechanism with demonstrated advantage over these controls has emerged.
3. The measured 40-call temporal pilot supports engineering reproducibility but provides no decision-order effect on eight windows and no verified contradiction. Scaling it would repeat known probes without answering the central support question.
4. Source access and split design are repairable engineering issues, not the fundamental blocker. Across all UCA splits, 156 original VERA validation IDs are annotated and span all 13 anomaly categories plus normal. The prior narrow joint-split candidate was unnecessarily restrictive. Do not falsely claim data are unavailable or category coverage impossible.

## Outcome and preservation

Stop new GPU computation, training-video downloads, and prompt optimization for this proposal. Retain the original 03B repair/report, all 40 pilot outputs, negative findings, rejected candidate, source hashes, split audit, qualitative review and compiled report. Total new model compute for this feasibility study: 89.54 seconds on one Ada (~0.025 allocated GPU-hours); no A16 usage. No PR merges.

## Conditions for reopening

A concrete claim-support evaluation set with independently reviewed supported/contradicted/unknown labels, source-frame or temporal evidence, and separate development/evaluation sources; a precise feedback mechanism differentiated from the reviewed methods; and a predeclared comparison that separates extra caption supervision from the proposed mechanism. A modest human-reviewed annotation effort is one route; an accessible independently validated dataset is another. Agent-only text review and absent caption matches do not satisfy this condition.

The human can review PR5 for experiment03B and PR10 for this feasibility study. No additional question is required while the user is AFK. Future unchanged heartbeats should perform no repeated searches or compute and remain quiet. Reopen only on new evidence or human direction.

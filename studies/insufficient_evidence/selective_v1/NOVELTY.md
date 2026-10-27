# Narrow novelty assessment — 2026-10-27, before inference

Question: can original frozen VERA retain its binary anomaly decision while withholding unresolved explanation details, and does requesting frame citations add anything beyond a manual restraint prompt?

Primary sources inspected:

* VERA, https://vera-framework.github.io/ — frozen learner/optimizer VLM roles, text question optimization, binary segment decisions and scene/temporal fusion. This study changes inference instructions only, not questions or weights.
* VirtueBench, https://arxiv.org/html/2603.07071v1 — missing sampled evidence, answerability/refusal, explicit refusal prompting. Whole-answer abstention is established prior work.
* Have I Seen Enough?, https://danbenami.github.io/have-i-seen-enough/ — question-conditioned readiness, frozen-model probes and gating. Author method page inspected, not an independent replication.
* GroundedVQA, https://arxiv.org/html/2608.15574v1 Sections 3 and 5 — drafts individual claims with timestamp citations, blind frame recaptioning and NLI verification; unverified claims are explicitly tagged. Direct conceptual overlap with selective explanation and citation-based grounding. Citation presence is not verification. Reported verifier performance is not independently checked here.
* TRACE, https://arxiv.org/html/2605.16740v1 Section 3 — structured grounding, evidence localization and grounded claims with citations across videos. Grounded claims are not new in themselves.
* Learning Temporal Evidence Selection for Video Anomaly Understanding, https://arxiv.org/html/2610.16299v1 Sections 3–4 — trained temporal sampler and language-generation metrics; changing frame selection would enter an already active direction.
* Anomaly-Guided Observation and Two-Level Semantic Feedback, https://arxiv.org/html/2610.16324v1 Appendix — frozen VLM visual verification and conservative preservation of detection scores under ambiguous feedback. Detection preservation plus verification is also occupied territory.
* VALU, https://aclanthology.org/2026.acl-long.56/ — existing multilevel anomaly boundaries/detail discrimination, potentially a stronger future evaluation source. Data access and compatibility have not been established in this run.

Decision: **no established method novelty** for the proposed prompts. A cheap training-free diagnostic is justified by the user's explicit request and the unresolved engineering question of detection/explanation coupling. No paper-level contribution is inferred from adapting known techniques to VERA. No learned extension or compute scale-up follows automatically.

Available references: original UCA manual captions/timestamps, with split crosswalk already audited. These are partial event descriptions, can include nonvisual inferences, and are not exhaustive supported/contradicted/unknown claim labels. Existing test footage has already been consulted. Thus this is an exploratory paired response study, not an untouched benchmark or human-validated explanation evaluation. Without stronger existing claim/evidence labels, we cannot prove a hallucination reduction from this run. No new human annotation is required or fabricated.

# Codex handoff for controlled evidence corroboration

Begin by reading LITERATURE_AUDIT.md and PROTOCOL.md in this directory. Implement the complete frozen experiment, run its engineering smoke, and then run the full UCF pipeline when the verified resource preflight passes. Compare against the user's VERA reference scores. Preserve negative findings and do not keep redesigning against test results until a gain appears.

## Repository and branch

Repository: https://github.com/shridpant/VERA

Use branch research/rev-vad-matched-evidence. It was created from exact commit 5dde005928d42f05625ceab948b8fa2afab772f3 on research/vera-position-audit. That parent carries the baseline engine, Experiment03B, verified score-replay machinery, and existing negative feasibility records. Its position-audit stopping rules apply to that old study; this new study has its own protocol. Preserve their findings and gates.

Use a separate checkout or worktree. Do not switch or modify a checkout owned by another running chat. Commit only this study's additions. Freeze source/config/manifest hashes before new inference, create a draft PR if appropriate, attach any created PR to the execution chat, and never merge without a direct user instruction. Push from the Mac, because earlier records report Oz has no GitHub credentials. A remote bundle copied to the Mac is an existing supported route.

## Known assets

The user requested discovery through prior chats. Relevant chats are Compare CoReVAD and VERA, Run reproducible GPU experiments (2), Find VERA transferability failures, and Find SP2117 RTX 6000 Ada results. Read them if resource ownership or reference lineage is unclear. Do not send messages to those chats without user authorization.

SSH target: sp2117@oz.gccis.rit.edu. Read-only SSH succeeded during preparation on October 27. Verified live: project /home/stu11/s5/sp2117/vera_research exists; /local/datasets/sp2117/ucf_crime_test_videos contains 290 mp4 files; shared/frames/ucf_crime_sparse and shared/models/OpenGVLab/InternVL2-8B exist under that project. This checks existence and count, not fresh integrity or GPU availability.

The pinned environment is setup/environment/env/bin/python. Source setup/source/VERA is pinned at 008ed4019781c39e7cb16244b7500244f3986f92. Model revision is 6fb9ad6924f69424e57fab2ab061d707688f0296. Read MODEL_PROVENANCE.json and check_source_resolution.py before loading. The existing sparse frame cache covers old selected inputs, not the dense new 69,634-anchor run. Extract all newly required frames from the archived mp4s into a NEW study cache. Streaming decoding and bounded per-video caches are preferable to millions of permanent JPEGs.

Local reference assets exist at /Users/shrid/Documents/Codex/2026-10-26/u-h/work/vera-position-assets. Scores ZIP: VERA_prediction_scores_UCF_Crime.zip, SHA256 b7457cee812510ec2331c24cb5da451327c9500152e2437839d31af29166a3b0. Feature ZIP: ucf_crime_vision_features.zip, SHA256 6b3e0a17091524c1635b68f18f8b1c5daaf067192d194e27d39de758db5f29e4. Use the manifest in studies/position_audit/evidence/assets.json for individual arrays.

Existing replay has 290 videos and 1,111,808 evaluated frames. Released-code 15 percent retrieval gives 86.6071 percent AUC; paper 10 percent gives approximately 86.548 percent. These are released-score replays, not a fresh reproduction of all VERA generations. Recompute from raw vectors and distinguish these reference tracks. The user's own other VERA run may have different prompts or scores: search the existing experiment run manifests before equating them with author references.

Previous Experiment03B already tests blur-ranked frame deletion under grey replacement. Read its protocol and amended validation. Do not call it a new result or confuse its 433 sampled inputs with full dense benchmark inference. The old evidence-learning branch closed several prompt and temporal-order pilots; use those failures as constraints.

XD released scores and metadata were located in prior work, but raw XD videos were not established. Inventory before promising an XD model run. An official 800-score-file archive alone cannot run new visual interventions.

## Work to implement

core.py implements only the support-score transform, label-free selection, and exact corroboration formula. Eight synthetic tests cover key fallbacks and bounds. There is NO implemented video/VLM CLI yet and NO new GPU result. Do not report a runnable full inference pipeline until you implement and verify it.

Implement these stages with separate resume-safe CLI entry points:

1. Inventory assets, current free GPU UUIDs, quotas, and model/source hashes. Use only a genuinely available RTX 6000 Ada. Do not terminate, resize, or reuse another study's jobs. Old records reserve Adas 6/7 for firearm studies, but recheck current ownership. A16 cannot hold this BF16 8B stack as configured.
2. Reuse the existing numerical engine by a NEW adapter with its own prompt and readout identity. Preserve old engine.py, attr_lib.py, old source, and caches. Validate candidate tokenization and BF16/FP32 readout alignment. No inference text parsing is needed for the main detector.
3. Implement all-window manifests, disjoint A/B sampling, masks and their complements, two operators, per-request hashes, deterministic selection, and complete audit records. Check transformations on synthetic arrays before GPU use.
4. Replay VERA references exactly using studies/position_audit/reconstruct.py and verify labels and lengths. The script's successful cache/output conventions and short-video behavior are documented in REPRODUCE.md. Keep absolute output paths.
5. Run the engineering smoke defined in PROTOCOL.md. Measure actual per-view latency and storage; do not reuse historical timing as a promise. Freeze the implementation and exact full-run call counts. All comparison arms and diagnostics count toward the projection.
6. Score EVERY window, freeze the label-free selected list, then verify. Persist unselected predictions; do not evaluate only alerts. Use separate run roots, atomic records, per-shard nonoverlapping manifests, locks, and a single final coverage audit. Resume must yield identical records to uninterrupted execution.
7. Run full UCF and required controls. Compare direct versus MEC using identical frame expansion first; compare VERA and CoReVAD separately under their native documented pipelines. No tuning of formula, prompt, 0.20 guard, or selection fraction using new test outcomes.
8. Produce raw per-frame predictions, paired bootstrap tables, within-video diagnostics, coverage/fallback tables, elapsed GPU time, and visual intervention case cards. Keep dataset frames out of Git. Complete XAI transfer evaluation under the third, held-out pixelation operator; do not score faithfulness only with the construction statistic.
9. Write RESULTS.md and a self-contained report with actual measured results, all negative outcomes, and limits. The full run can complete successfully as a negative experiment. Freeze this version before a separately inventoried XD transfer.

Expected file layout to add: adapter.py, masks.py, manifest.py, run.py, evaluate.py, validate.py, configs/ucf_v1.json, records schema, and tests. The conceptual commands are plan, score, select, verify, evaluate, report; choose the concrete argparse interface and document exact working commands after implementation. Do not copy an invented command into a launch log.

## Scientific constraints

The original REV-VAD novelty claim is withdrawn. The candidate's exact combination is provisional; no priority or superiority has been established. Before large compute, inspect the closest methods' implementations where accessible and record any mechanism overlap. Do not use inability to prove novelty as a reason to fabricate a distinct claim.

This rule only boosts corroborated scores. No boost means unresolved, not normal. At fixed thresholds false positives cannot decrease; any false-alarm improvement must be at matched recall after reranking. A wrong semantic cue can still pass both tests. UCF temporal labels do not establish correct spatial explanations. Sampling B is correlated, not an independent test set. These limits must survive the final report.

The main fixed-prompt model is training-free. Reusing VERA's learned questions belongs to a separately labeled secondary condition. Compare against direct and equal-compute baselines before attributing improvement to evidence checks. Do not lower unrelated old study gates or discard failed experiments.

## First actions

Read this packet, inspect the checkout and AGENTS.md if present, run python3 studies/matched_evidence/test_core.py, reproduce baseline metadata and score integrity, and implement CPU mask/manifest fixtures. Then progress through the documented smoke and complete UCF run. Ask only for concrete missing access or a necessary resource decision that the existing records cannot resolve. Do not launch duplicate work owned by another chat.

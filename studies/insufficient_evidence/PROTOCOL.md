# Insufficient-evidence experiment: novelty and annotation gate

Human authorization: October27 request to check novelty, establish human annotation requirement, run bounded pilot if justified. Separate branch research/vera-insufficient-evidence from83b1d18. Prior study records frozen.

## Novelty assessment
- VirtueBench, CVPR2026, https://arxiv.org/html/2603.07071v1 Section3 inspected: frame sampling can change answerability; refusal evaluated under missing evidence. This directly overlaps the earlier broad recommendation. Generic sampled-frame insufficiency is NOT new.
- Have I Seen Enough?, author method page https://danbenami.github.io/have-i-seen-enough/ inspected: frozen-model evidence-readiness probes and gating distinct from confidence. Generic frozen-model readiness is NOT new. Full paper/code reproduction not performed; page says code soon.
- MM-AQA https://arxiv.org/abs/2604.14799 primary abstract: visual evidence sufficiency and abstention tradeoffs. More elaborate prompting alone insufficient novelty.
- Original VERA question learning and LRPO temporal linguistic learning remain essential comparators from prior review.

Candidate remaining question: can learned guiding questions reduce unsupported *specific explanations* while retaining alerts supported at a coarser event level? Detection and explanation need separate decisions, e.g., visible damage supports an alert while collision causality remains unknown. This is a hypothesis, NOT a verified gap; compare manual selective-explanation prompting and confidence/readiness baselines before claiming contribution. Do not claim first abstention/uncertainty method.

## Immediate pilot decision
GO for annotation feasibility calibration; NO-GO for learning or GPU performance pilot before independent labels and precise gap hold. Packet uses8 already-consulted UCF sources ×2views (ordered8, slot3 repeated8), with4fixed event questions each.64labels per reviewer. This tests rubric/interface feasibility, not a learned method or heldout performance. Interventions do NOT automatically imply insufficiency. Fixed questions avoid deriving a task from model responses. No new inference.

Packet builder verifies64source JPEG hashes and applies pinned VERA grid1 RGB448resize. Sheets contain exactly supplied frame sequence with numbered slots; no category/sourceID/model answers. Pair membership private; repetition remains visually evident. All raw dataset images stay local outside Git. Private mapping and packet kept under work/ and outputs/ respectively. Reviewer templates blank; never fabricate annotations.

Two independent reviewers needed; single reviewer supports sanity checks only. Budget43–96personminutes plus20–40adjudication minutes, provisional until timed. Question to user sent, no response yet at preparation. RUBRIC.md contains labels, citation requirements, ambiguity handling and agreement gate.

## Conditional model pilot, not launched
After calibration and focused narrow-mechanism review, freeze a separate source-grouped training/development/evaluation manifest from original VERA boundaries; do not use calibration/test sources for learning. Source-hash and scene checks required. Start12training+12development sources,2views, three prompt conditions (unchanged, manual selective-explanation, learned), max144evaluation calls plus48optimizer/selection calls=192calls, max1GPU-hour. This is feasibility only; a separate20source evaluation should remain sealed until selection finishes. Bound counts include all retries; stop on parse/mapping errors or budget exhaustion. Inspect GPUownership immediately before allocation; at mostoneAda, no6/7, noA16needed.

Use human support labels independent of outputs for fixed questions, then independently review generated claims. Ground-truth availability and correctness remain separate. Original anomaly alerts scored separately; for modified methods explanations may abstain without removing the alert. Report supported alert retention and unsupported assertion rate jointly with explanation coverage and abstention on supported examples. All-abstain baseline mandatory. Match call budgets and human supervision; declare extra supervision beyond weak labels. Do not map original binary output to 'insufficient' without evidence.

Stop if reviewer agreement fails, labels lack both support/insufficiency, manual baseline explains gains, or narrow contribution duplicates prior work. An annotation request is missing essential data, not another permission request. No GPU inference is justified before resolving it.

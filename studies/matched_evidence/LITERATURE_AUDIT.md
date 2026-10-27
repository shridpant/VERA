# Literature audit of evidence verification for video anomaly detection

October 27, 2026. Decision: retire the original REV-VAD novelty claim. Proceed to an engineering pilot of the narrower protocol in PROTOCOL.md. Its method novelty is provisional, and its detection benefit is unmeasured.

## Closest methods

| Primary source | Inspected scope | Established overlap | Implication for this study |
| --- | --- | --- | --- |
| [Mänttäri et al., ACCV 2020](https://arxiv.org/abs/2002.00367) | Author abstract and conference PDF excerpts | Meaningful perturbation extended to time to identify influential video fragments | Removing temporal blocks and measuring score changes is established |
| [Li et al., WACV 2021, STEP](https://openaccess.thecvf.com/content/WACV2021/html/Li_Towards_Visually_Explaining_Video_Understanding_Networks_With_Perturbation_WACV_2021_paper.html) | Abstract and method excerpts | Video perturbation explanations, preservation objectives, and spatiotemporal smoothness | Keeping evidence, small masks, and smooth video masks are established |
| [Zhang et al., EUSIPCO 2023](https://eurasip.org/Proceedings/Eusipco/Eusipco2023/pdfs/0001235.pdf) | Full five-page paper | Evaluates occlusion, LIME, and KernelSHAP for VAD; separates fidelity, anomaly localization, stability, and cost | Perturbation XAI specifically for VAD is established; detection and explanation endpoints must be separated |
| [RISE, 2018](https://arxiv.org/abs/1806.07421) | Author abstract | Black-box input masking and insertion/deletion evaluation | Black-box explanation and measured deletion curves are not new |
| [Lin et al., AAAI 2022](https://guanbinli.com/papers/A%20Causal%20Inference%20Look%20At%20Unsupervised%20Video%20Anomaly%20Detection.pdf) | Abstract and counterfactual temporal ensemble method | Counterfactual feature replacement blends local and long-range context after deconfounded training | Neither counterfactual VAD nor inference-time correction is broadly new; this is a trained feature-level method |
| [LAVAD, CVPR 2024](https://arxiv.org/abs/2404.01014) | Author abstract and paper excerpts | Training-free caption cleaning, anomaly reasoning, and contextual score refinement | Training-free refinement and explainability are established |
| [VERA, CVPR 2025](https://arxiv.org/html/2412.01095v2) | Full method and experiment passages | Frozen weights, learned guiding questions, scene and temporal refinement | Frozen weights do not imply training-free; VERA-conditioned extensions must be labeled separately |
| [CoReVAD, 2026](https://arxiv.org/html/2605.23116v1) | Full method, implementation, and comparison tables | Local response cleaning with CLIP and contextual refinement without task-specific training | Contextual cleaning is an essential baseline, not a novelty claim here |
| [PrismVAU, WACV workshop 2026](https://arxiv.org/html/2601.02927v1) | Paper overview, method structure, and explanation-evaluation passages | Coarse scoring followed by MLLM refinement and prompt/anchor optimization | A score-first pipeline or additional reasoning pass is insufficient novelty |
| [ASK-HINT, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.html) | Primary abstract and method excerpts | Fine-grained action questions for frozen VLMs on UCF and XD | Manually improved action prompts alone are not a new contribution |
| [CEAVAD, August 2026 preprint](https://arxiv.org/html/2608.09908v1) | Sections 2 and 3 | Hazard-versus-benign hypotheses, a semantic proposal, and MLLM adjudication using Past/Target/Future observations | Competing explanations and evidence adjudication are crowded; this proposal instead measures controlled input sensitivity |
| [AgenticVAU, August 2026 preprint](https://arxiv.org/html/2608.03779v1) | Overview and method | Adaptive exploration, dense revisits, and supporting/opposing evidence registries using four agents | Explore-then-verify and targeted observations are not new; our fixed small mask bank avoids agent orchestration |
| [CSI-VAD, July 2026 preprint](https://arxiv.org/html/2607.19077v1) | Full method and evaluation setup | Environment, tracked-object, and time branches with score fusion | Multiple context views and ensembling are established; its reported UCF task is video-level |
| [Parser-Free VLM Verification, September 2026 preprint](https://arxiv.org/html/2609.07455v1) | Abstract and method interface overview | A trained federated MIL scorer proposes suspect segments; frozen VLM next-token feedback refines their scores | Selective verification, next-token scores, and score fusion are not independently novel |
| [Song and Lee, August 2026 preprint](https://arxiv.org/html/2608.21854v1) | Abstract and evaluation analysis | Pooled frame metrics can predominantly reward cross-video ranking | Report within-video localization alongside official pooled metrics |

These are different tasks and supervision settings. This table establishes overlap, not comparable benchmark rankings. A preprint's existence does not establish the correctness of all its claims.

## The closest prior experiment is already in this repository

Experiment03B, on branch research/experiment03B-appendix, already selects a frame pair using blur-derived numerical effects and tests the original VERA alert under grey replacement against all 28 possible pairs. It also examines adaptive rankings and replacement sensitivity. Its narrow reported effect was 15.1656 percentage points, with a video-bootstrap interval [9.8599, 20.6004], on 199 alert windows across 138 videos. This is an intervention endpoint, not full UCF detection AUC. Consult its amended validation records before quoting it.

The original REV-VAD proposal is therefore not a defensible new explanation method for this project. Repeating that experiment with fewer temporal blocks would be a regression in experimental rigor.

## Candidate revisions and decisions

1. Random temporal deletion: reject. Established attribution, token-count confounding, and redundant-event failure.
2. Score-ranked temporal deletion: reject as a new contribution. Direct overlap with Experiment03B.
3. Hazard-versus-benign prompting: reject as the principal contribution. Strong overlap with CEAVAD and normal/abnormal prompt literature.
4. Adaptive exploration or question learning: reject for this simple experiment. Prior methods and the repository's evidence-learning pilots already occupy much of this space.
5. Controlled corroboration reranking: retain for a bounded engineering pilot. Select spatial evidence from scores, check it on different sampled frames, compare deletion AND retention against equal-area controls under two operators, and boost only corroborated scores. Keep ambiguous scores unchanged.

## Narrow claim to investigate

Proposed, not established: a fixed, conservative corroboration rule based on selected spatial tubes, equal-area controls, disjoint frame sampling, and agreement between deletion and retention contrasts can improve frozen-VLM anomaly ranking while producing an auditable intervention record.

The possible contribution is the precise inference rule and its validated benefit over constituent ablations and equal-compute controls. None of masking, sufficiency, necessity, disjoint sample checks, or positive-only fusion is individually claimed as new. The combination may prove incremental or ineffective. Search did not establish an identical VAD formulation; that is not proof of priority.

Search families covered temporal deletion/occlusion, minimal sufficient evidence, retention and deletion, counterfactual VAD, selective VLM verification, matched/negative controls, spatial perturbations, and training-free evidence grounding. Search snippets were used for discovery; substantive comparisons above use primary author/publisher sources. A final claim also requires checking cited successors and available implementations of the closest methods, especially STEP and selective verification, before a publication submission.

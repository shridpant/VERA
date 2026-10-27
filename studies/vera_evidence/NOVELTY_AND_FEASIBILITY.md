# Evidence-feedback question learning: first feasibility gate

October 27, 2026. This is a primary-source literature and metadata assessment, not a novelty proof or a model experiment.

## Decision

**No-go for a broad novelty claim that prompt optimization or grounded feedback improves frozen-VLM anomaly detection.** VERA, PrismVAU, ASK-HINT and LRPO already occupy substantial parts of this claim. **Conditional go for a narrower feasibility investigation**: can evidence feedback reduce unsupported claims beyond no-sound and citation-only controls, while preserving detection and avoiding reward for fragility? Do not start a large optimization sweep yet.

## Primary-source comparison

| Work | Source inspected | Established overlap | Remaining question, not a novelty claim |
|---|---|---|---|
| VERA (CVPR 2025) | Original paper/repository inspected in preceding work | Learner/optimizer interaction learns natural-language guiding questions for frozen VLMs | Can externally checked support improve learned questions beyond classification feedback? |
| PrismVAU (WACV workshop 2026) | Full method and implementation sections: https://arxiv.org/html/2601.02927v1 | Same MLLM optimizes anchors and prompts using weak labels; video-level AUC selects prompts | A new contribution cannot just be another prompt optimizer or frozen-model refinement |
| ASK-HINT (WACV 2026) | Primary paper abstract/introduction: https://openaccess.thecvf.com/content/WACV2026/papers/Zou_Unlocking_Vision-Language_Models_for_Video_Anomaly_Detection_via_Fine-Grained_Prompting_WACV_2026_paper.pdf | Fine-grained action/interaction questions for frozen VLMs on UCF and XD | Compare against structured/manual prompts before attributing benefit to learning |
| LRPO (ICML 2026) | Publisher abstract: https://proceedings.mlr.press/v306/leng26c.html | Textual experience priors from group-relative trajectories, including anomaly-alignment feedback and temporally grounded reasoning | Closest overlap requiring full algorithm/reward inspection; PDF URL returned unsupported content-type in browser, not evidence of unavailability |
| GEPA | Primary abstract and algorithm excerpt: https://arxiv.org/abs/2507.19457 | Reflective prompt evolution using trajectory/feedback information | Generic reflection or richer feedback is not new by itself |
| CPL | Author organization page: https://research.google/pubs/cpl-counterfactual-prompt-learning-for-vision-and-language-models/ | Counterfactual prompt learning already exists for vision-language tasks | Need a precise difference from generic counterfactual learning |
| CCT (ACL 2024) | Author organization abstract: https://deepmind.google/research/publications/78755/ | Counterfactual explanation evaluation considers distribution shifts beyond binary flips | Citation/deletion consistency is related evaluation, not automatically a new faithfulness metric |
| LATERN | Primary abstract: https://arxiv.org/abs/2605.15054 | Frozen-VLM temporal evidence aggregation and image-grounded memory | Adaptive context/evidence alone is crowded |
| Instance-Aligned Captions | Primary abstract/full HTML structure: https://arxiv.org/html/2601.08155v1 | Links claims to entities; reports annotated anomaly benchmarks | Potential external support labels; release access, license, coverage and independence must be verified |

Do not extrapolate absence from an abstract. Full reward and evaluation details for the closest work are outstanding. No ranking of reported SOTA numbers is attempted because protocols differ.

## Existing experiments constrain the direction

Experiment03B already establishes perturbation-ranking utility, so reproducing ranked deletion is insufficient. Its average effect also includes fragile false alerts. The original scorer and explanation-first VERA are different conditions. Sound claims on image-only input demonstrate a directly checkable modality violation but cover only one narrow kind of unsupported claim. Removing the sound question is a necessary cheap baseline; it may explain the entire improvement. A model judging its own response is not independent evidence of semantic support.

## Data inventory and reproducible metadata check

On Oz, UCF test videos, sparse frames, reference predictions and ImageBind features are staged. In the inspected user-owned locations no UCF training videos or XD videos were located. `datasets` and `datasets_qwen` contain Python loaders, not videos. Metadata paths for XD point to the original author's host.

`code/audit_metadata.py` hashes the six supplied JSON files. UCF metadata has 1,449 train, 161 validation, 290 test IDs; XD has 3,559 train, 395 validation, 800 test IDs. Exact video IDs are disjoint across all three splits for both datasets. This is not a source-byte or scene-disjointness guarantee. Extensionless XD IDs contain dots; the audit strips only known video suffixes, never arbitrary final dotted components. An initial development use of Path.stem incorrectly collapsed these IDs and was corrected before reporting counts.

UCF training QA strings sometimes include temporal intervals. If the protocol claims weak labels, derive only the declared video label and do not silently use temporal hints to select training windows. The UCF test collection is already consulted research material and must not serve as prompt-training data or be called untouched future confirmation.

## Next gates (no GPU used yet)

1. Read LRPO's exact feedback/reward mechanism and verify access to external claim-support annotations. If the proposed mechanism is already covered, pivot to a clearly scoped comparative evaluation rather than inventing novelty.
2. Verify official UCF training acquisition and create a deterministic, source-disjoint small pilot manifest before fetching selected videos. Test train/validation/test identity and source hashes. Avoid downloading the full collection by default.
3. Freeze a pilot comparison with unchanged VERA, no-sound, citation-only, and evidence-feedback variants. Account for ALL selection and verification calls. Keep detection, unsupported-claim rate, intervention robustness and invalid output rate separate; no combined metric that rewards silence or fragility.
4. Use known-modality violations as an engineering sanity check only. Broader semantic faithfulness needs reliable external labels or a documented evaluation limitation. Stop if the verifier cannot distinguish false claims, or if simple controls account for the effect.

GPU snapshot: Ada4 low memory/idle utilization, but ownership not established; Adas6/7 reserved to the other study. A16 10/11 empty at check. No device is reserved and no new inference was launched.

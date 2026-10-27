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
| LRPO (ICML 2026) | Full Section 3.4 and Appendix C: https://arxiv.org/html/2607.00654v1 | Textual experience priors from group-relative trajectories, including anomaly-alignment feedback and temporally grounded reasoning | Ordered/shuffled feedback already overlaps; inspect reward null behavior before adapting |
| GEPA | Primary abstract and algorithm excerpt: https://arxiv.org/abs/2507.19457 | Reflective prompt evolution using trajectory/feedback information | Generic reflection or richer feedback is not new by itself |
| CPL | Author organization page: https://research.google/pubs/cpl-counterfactual-prompt-learning-for-vision-and-language-models/ | Counterfactual prompt learning already exists for vision-language tasks | Need a precise difference from generic counterfactual learning |
| CCT (ACL 2024) | Author organization abstract: https://deepmind.google/research/publications/78755/ | Counterfactual explanation evaluation considers distribution shifts beyond binary flips | Citation/deletion consistency is related evaluation, not automatically a new faithfulness metric |
| LATERN | Primary abstract: https://arxiv.org/abs/2605.15054 | Frozen-VLM temporal evidence aggregation and image-grounded memory | Adaptive context/evidence alone is crowded |
| Instance-Aligned Captions | Primary abstract/full HTML structure: https://arxiv.org/html/2601.08155v1 | Links claims to entities; reports annotated anomaly benchmarks | Potential external support labels; release access, license, coverage and independence must be verified |

Do not extrapolate absence from an abstract. LRPO reward details have now been inspected; implementation equivalence remains unverified. No ranking of reported SOTA numbers is attempted because protocols differ.

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

## Second gate: analytical diagnostic (October 27, 00:15 EDT)

LRPO already uses weak-label preference text and ordered/shuffled frame rewards to update linguistic experience. Its Section 3.4 Eq.5 uses delta=a-0.8b with Appendix C thresholds 0.05 and 0.20. The prose says ordered performance must exceed shuffled performance; the printed equation does not explicitly impose that additional gate. We have not inspected author code.

**Our calculation:** equal scores a=b=0.8 yield temporal reward 0.117333 for a correctly classified output. Even a=0.7,b=0.8 yields 0.004. At equality delta=0.2a, so positive reward begins at a>0.25. This is an equation-level null diagnostic, not evidence against reported model results. An extra strict outperformance gate would remove these counterexamples. The script checks five analytical identities and produces synthetic JSON and a two-panel figure. Zero GPU-hours.

Instance-Aligned Captions Section 3.2 uses manual roles/boxes followed by SAM2 masks and VLM captions. It is not wholly human-verified claim truth. Table 1 includes UCF (295 captions) but not XD. No direct annotation download was established from the inspected paper and targeted search; this does not prove no release exists.

**Pivot:** postpone evidence-question learning. First test whether purported temporal evidence feedback can distinguish a real order effect from an order-invariant null. This is a candidate diagnostic research question, not a novelty claim. Check LRPO code availability and prior temporal-order diagnostics before any GPU pilot. A future original-VERA pilot must compare identical-input repeats, fixed frame permutations and repeated-single-frame nulls, preserving the frame multiset except the explicitly separate repeat-frame control. Treat the already-consulted UCF test videos as exploratory diagnostics only, never prompt selection or pristine confirmation. Do not equate a shuffle-induced drop with semantic faithfulness; permutations may be out of distribution. Establish null behavior before considering reward-driven learning or fresh training-data acquisition.

Reproduce: `python studies/vera_evidence/code/audit_temporal_reward.py` with matplotlib installed. Figures and JSON are deterministic numerical diagnostics; PDF metadata may vary. Existing 03B records are unchanged.

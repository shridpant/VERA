# Completed grounding measurement gate: failed V2, no Phase 2 training

Source: Oz `/local/datasets/sp2117/firearm_visilab/grounding1`. The complete corrected run is `phase1_seed_measurements.jsonl` (398 unique videos). The 109-record v0 partial run is preserved separately and is not used for the gate.

V1: 275 videos, true-minus-shuffled shared-frame lift 1.536476701591593, 95% recording-group bootstrap CI [1.4003637935857363, 1.6775087914025346], pass. V2: 274 videos, Spearman rho 0.1128229434798778, CI [-0.013136515835146016, 0.22919711414228594], fail. Both lower bounds must exceed zero; the combined gate fails. No Phase 2 training or Phase 2 test effect exists.

The gate reads all splits, including test boxes, for feasibility. It is not a held-out effect of learned grounding feedback. `DEVIATIONS.md` explains eager/FlashAttention readouts, the restart, and corrected shared-frame pairing.

The consolidated audit's `reproduce.py` independently reproduces the gate from these raw measurements using the original bootstrap settings. `source_manifest.json` records remote source hashes. Original analysis tables and full-precision numbers are preserved in `analysis/`; sources remain in `studies/firearm_grounding/report`. The consolidated report and proof README provide cross-study interpretation. No new model calls or changes to running experiments.

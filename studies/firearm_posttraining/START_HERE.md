# Start the one-GPU firearm post-training study

Read the full protocol in `POSTTRAINING_HANDOFF.md` and its defaults in `POSTTRAINING_PLAN.json` in the delivered folder. In the repository these are `studies/firearm_posttraining/HANDOFF.md` and `studies/firearm_posttraining/plan.json`.

The user approved the five-method controlled comparison and wants one GPU, LoRA supervised training first with stopping criteria, then a measured and tested RL environment and GRPO training. This chat prepared the handoff only; no new environment or training was started.

Use `/Users/shrid/Documents/research/VERA-package` as the source repository. Recheck its branch/status and applicable instructions. Connect to `sp2117@oz.gccis.rit.edu`; recheck existing jobs and the preferred Ada 6 UUID `GPU-2931eba2-9a4d-61da-6bd1-e75692edad03`. Keep Ada 7 free. Do not interfere with the ongoing Phase 1 jobs or reuse their output folders.

Begin with isolated working state, data audit, a new locked training environment, and an actual one-GPU SFT/GRPO compatibility smoke test. Then run the bounded supervised pilot, stop and save the best checkpoint, use FIT-only measurements to finalize rewards/reference handling, freeze the protocol, and run the matched methods/seeds sequentially. The protocol specifies hard caps, exact stop rules, reward definitions, paired warm starts, evaluation guards and required final tables.

The plan's numerical defaults are explicitly proposed and can be revised by the user. Resolve routine implementation details autonomously within the approved scope. Do not silently change the scientific comparison, expand GPU usage, leak test information, overwrite running work, or claim implementation/test success before it exists. Record every material amendment. No push or merge is authorized.

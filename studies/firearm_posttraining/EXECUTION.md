# Post-training execution: scientific sweep running

The coordinate amendment is implemented and all prerequisite gates passed. This is an ongoing study, not a completed comparison. No held-out improvement or RL benefit is established yet.

## Authorization and immutable source

On October 27, 2026, the human instructed chat 01a17dce-3275-78b0-b005-0f3a408d5cbe to fix the coordinate issue, rerun format checks, and continue post-training if they pass, retaining scoring rules and an audit trail. The human separately approved updating public PR9. The accepted amendment is recorded in AMENDMENTS.md. There is no pending coordinate decision.

- Local branch: experiment/firearm-posttraining; draft PR https://github.com/shridpant/VERA/pull/9. Never merge.
- Running source: clean immutable release /local/datasets/sp2117/firearm_visilab/code/posttraining-releases/8d9232c, full commit 8d9232c85e2406a25bde1e0f8c1e8f67f132eafb. Later local documentation/evidence commits must never be synced over this running release.
- SSH: sp2117@oz.gccis.rit.edu.
- ROOT: /local/datasets/sp2117/firearm_visilab/posttraining1.
- Python: /local/datasets/sp2117/envs/firearm-posttrain/bin/python.
- Driver PID300975; calibration/pipeline controller PID301220. Read ROOT/native-coordinate-driver-v1-status.json, ROOT/calibration-native-coordinates-v1/status.json and ROOT/pipeline/status.json before acting. Do not duplicate launch or resume the dead controller293624.
- Authorized independent single-GPU workers: Ada6 GPU-2931eba2-9a4d-61da-6bd1-e75692edad03 and Ada7 GPU-721a5bd6-d1e2-d5c5-af9a-6db4bdf691f7. Preserve all other experiments/environments.

## Verified attempt010

| Stage | Result |
|---|---|
| Tests |78 local firearm tests and30 focused Oz tests passed|
| Actual tokenizer/source audit |240 FIT/STOP videos,799 known boxes; unit round-trip error1.11e-16, maximum284 completion tokens,553 unknown payload tokens masked|
| Fresh engineering smokes006 |Both passed exact prediction reload, exact next-update resume, finite decreasing SFT loss, frozen-base and warm-reference checks|
| Fresh paired warm-ups010 |Both completed exactly3 passes,576 exposures,72 updates, no STOP selection; label adapter hash exactly reproduced prior label run|
| Label format010 |95/96 valid, no truncations; passed|
| Grounded format010 |92/96 valid, no truncations; passed. Three missing closing braces and one trailing code fence remain invalid|
| Label actual-reward panel010 |24 groups,96 responses,6 updates;16/24 groups with reward spread,1/96 invalid, changed policy and unchanged reference; passed|
| Grounded actual-reward panel010 |24 groups,96 responses,6 updates;20/24 groups with reward spread,2/96 invalid, changed policy and unchanged reference; passed|
| Pilot ledger |5.190671679 of6 aggregate GPU-hours settled; no pilot reservation remains|

Engineering synthetic fixtures are mechanics only. Only the true sampled-reward panels count as scientific feasibility. Read-only format reproduction validates213 accepted unit boxes and preserves all raw failures and artifact hashes in evidence/native-format-audit.json. Scoring-source and update-source equivalence are proved by coordinate_audit.py. The immutable actual audit is ROOT/data/coordinate-audit-v1.json. New target bytes are explicitly different; label prompt/targets are unchanged.

## Frozen protocol and live work

Protocol hash: 3f1f1e8a9aa78a0c5124f765df6712d2c33c0a1db770947edd3318cbec4abe80. Actual protocol.validate passed on Oz. Pipeline stage sweep_running. The first scientific paired warm-ups C0L-20261026 / C0G-20261026 are alive on GPUs6/7 (initial child PIDs306456/306476; recheck because future stages replace them). All five methods VQ/SL/RL/SG/RG and all three seeds20261026/20261027/20261028 are scheduled by the frozen pipeline. Selection manifest does not yet exist.

Model-facing coordinates are strictly0–1000, normalized to unit geometry before the same hierarchical-iou-v1 scoring functions. No unit guessing, clamping, geometry repair or rescue of previous failed outputs. Reward coefficients, IoU/matching, parsing structure, <=5% validity threshold, FIT-only96-response/24-group panels, >=20% spread requirement and all stopping rules are unchanged.

Three warm-up FIT passes count inside the five-total-pass cap. Scientific/evaluation time ceilings remain removed by prior human authorization; pilots retain their six-hour aggregate budget. FIT192/STOP48 from TRAIN240, original VAL62/test96. Eight448-square chronological full-frame letterbox views; native BF16 language-attention rank16 LoRA. Paired warm starts, exact checkpoints, common label scoring independent of box validity and all source/data/model/environment bindings remain enforced.

Measured maximum-schedule forecast:28.38 training GPU-hours, approximately15.51 wall-hours with two workers, plus2.65 evaluation GPU-hours. This includes overhead and is not an ETA; scientific stops may shorten runs. The training forecast is an estimate, not a newly introduced cap.

All15 method/seed selections must freeze before guarded val/test evaluation and report generation. RG–SG remains primary and RL–SL secondary. Paired recording-group bootstrap and conservative sparse-negative support screen remain. Prior Training1 test exposure is documented; do not call the historical test newly pristine. No held-out outcomes guided this amendment.

## Continuing work

The detached driver automatically chains successful sweep→selection freeze→guarded evaluation/report. Existing15-minute continuation may inspect and advance meaningful work after this chat yields; no new automation was created. Do not start a duplicate controller, modify an active release, or change frozen scientific rules. If the pipeline fails, diagnose the actual failure, preserve evidence and apply an explicit auditable amendment where needed; never silently broaden a finite pilot budget or retry a failed scientific candidate.

Canonical human-facing continuation record: /Users/shrid/Documents/Codex/2026-10-26/u-h/outputs/CONTINUE_EXPERIMENTS.md. This chat's delivered snapshots are /Users/shrid/Documents/Codex/2026-10-27/wha/outputs/coordinate-rerun-evidence.json and native-format-audit.json. Keep PR9 current with verified results. Deliver the eventual report, tables and figures; running workers do not mean the overall study is complete. Prior failures, old masks/samplers and the old proposed-decision state remain in AMENDMENTS.md, evidence/ and Git history.

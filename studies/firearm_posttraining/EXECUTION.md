# Actual execution status

The study is ongoing. Draft PR: https://github.com/shridpant/VERA/pull/9. No merge.
The current user authorizes both remaining Adas, commits and a review PR, no
scientific-run time ceilings, six aggregate bounded pilot GPU-hours, and all
other scientific stops. These resource changes from a side chat were directly
confirmed. HANDOFF.md is a historical proposal; current amendments govern.

## Current execution owner

CPU controller PID 283507 runs from immutable release 1b109ed. Its actual status
is ROOT/calibration-mask-v2/status.json, with log
ROOT/calibration-mask-v2-controller-001.log and launch metadata
ROOT/calibration-mask-v2-launch-001.json. Inspect live artifacts before acting;
the controller advances automatically. Do not double-submit GPU jobs.

Corrected one-pass and 1.5-pass warm-ups completed without STOP queries. Label
sampled-format gates passed 94/96 and 96/96 respectively. Grounded format failed
5/20 and 5/18 respectively, with early stopping once <=5% of 96 became impossible.
The declared final two-pass candidate 007 is running on both Adas. Strict parser
and thresholds are unchanged. Actual reward panels must pass at the same common
warm-up before protocol freeze. No comparative sweep or held-out results yet.

## Isolation and reproducibility

- Local isolated checkout/branch experiment/firearm-posttraining is based on
  85e7aba. The original research checkout is untouched.
- Remote central checkout: /local/datasets/sp2117/firearm_visilab/code/VERA-firearm-posttraining.
- Immutable releases: /local/datasets/sp2117/firearm_visilab/code/posttraining-releases/.
- ROOT: /local/datasets/sp2117/firearm_visilab/posttraining1.
- Only Ada 6 GPU-2931eba2-9a4d-61da-6bd1-e75692edad03 and Ada 7
  GPU-721a5bd6-d1e2-d5c5-af9a-6db4bdf691f7, one process/card. Preserve other jobs.
- Fresh environment: /local/datasets/sp2117/envs/firearm-posttrain/bin/python;
  78 exact distributions with wheel hashes; offline wheels/provenance ROOT/setup/.
  setup.sh refuses an existing environment; never overwrite this one.
- All runs require clean committed source and immutable specs, command, code,
  model/data/adapter hashes, seed, environment, logs, process IDs and resources.
- Checkpoints save adapter, optimizer/scheduler, Python/NumPy/Torch/CUDA RNG,
  cursor/order, selection and hashes. Base weights are referenced, not copied.
- execute.py locks authorized idle devices and reserves/settles shared pilot
  GPU-seconds. Occupied GPUs fail; stale utilization can settle for 30 seconds
  only with no process and <500 MiB. Prelaunch failures retain their logs.
- Scientific and evaluation jobs use --seconds 0 / --max-seconds 0 to remove time ceilings. Pilots remain finite;
  budget exhaustion requires an explicit human amendment, never silent extension.

## Verified data, model and engineering

FIT 192 / STOP 48 from TRAIN 240; original validation 62 / test 96 metadata. No source/
recording-group overlap among original splits or exact-frame FIT/STOP overlap.
All 1,920 development frames verified and hash-checked on every read. Eight full
chronological 448 letterbox views; no GT crops. Unknown listed_no_box is unknown,
not absence. Native Qwen3-VL-8B BF16 with rank 16 language-attention LoRA freezes
vision/projector/embeddings/head/base tensors. Same backbone/capacity across arms.

All 74 firearm tests pass locally; 26 focused also passed on Oz at 1b109ed. Real
VQ compatibility passed 16 images / 4,451 prompt tokens with a valid single-slot
revision and unchanged adapter. Both fresh engineering gates smoke-labels-004
and smoke-grounded-004 passed exact prediction reload/next-update resume,
finite/decreasing SFT loss, adapter-only updates and explicit nonzero frozen
warm-reference scoring. Fixture rewards had zero spread: distinct synthetic
strings only verified gradient plumbing, never scientific RL feasibility.

The actual pinned-tokenizer audit passed all 240 FIT/STOP videos. Target bytes
unchanged; maximum 311 completion tokens; 6,658 formerly masked known tokens
restored; 553 unknown-payload tokens masked. Known schema tokens crossing span
boundaries remain supervised; tokens touching unknown payload stay masked.
This corrected a genuine JSON-boundary masking bug. Fresh adapters replace old
ones; thresholds, target text, views, prompts and rewards are unchanged. See
AMENDMENTS.md and evidence/target-mask-diagnosis.json.

## Preserved pilot history

| Evidence | Result | Interpretation |
|---|---|---|
| Legacy pilot-sl-001 | Normal patience stop, 5 FIT passes / 120 updates / 2,470 seconds; best 3.5 passes | Exploratory selection evidence under old mask |
| Legacy STOP curve | Best category balanced accuracy 86.3%, group 70.8%, code 29.2%, invalid 0%, No_Gun 0/14 | Baseline invalidity 50%; transient 4/14 false positives at 1 pass retained; no held-out claim |
| Legacy pilot-rl-001 | 96 responses / 6 updates, 29.17% spread groups, 1.04% invalid, unchanged reference | Passed legacy feasibility, not a substitute for corrected gates |
| Legacy pilot-rg-001 | 28/48 invalid, no truncation; gracefully stopped/checkpointed | Failed; logs/operator action preserved |
| Legacy 1/1.5-pass grounded formats | 5/7 and 5/5 invalid | Led to actual-tokenizer mask diagnosis |
| Corrected 1-pass 005 | Both warm-ups 192 clips / 24 updates; label 94/96 valid, grounded 5/20 invalid | Grounded gate failed; raw FIT traces preserved |
| Corrected 1.5-pass 006 | Both warm-ups 288 clips / 36 updates; label 96/96 valid, grounded 5/18 invalid | Grounded gate failed; final declared 2-pass candidate follows |

Corrected FIT failures include non-unit coordinates, malformed/unterminated
JSON, duplicate responses/trailing prose and truncation. Do not relax parsing,
truncate answers post hoc or treat partial-panel rates as full-panel estimates.
Failed runs, traces and operator actions remain available under ROOT and evidence/.

## Remaining gates and inference policy

Finite FIT-only warm-up grid 1/1.5/2 passes, shared across all pairs/seeds, no STOP
selection. Both format gates must pass at one exposure, then both true reward
panels: 24 groups / 96 responses / 6 updates, >=20% spread groups, <=5% invalid/truncated,
finite material gradients/policy change, unchanged actual warm reference. A real
reward failure stops for diagnosis; no blind retry or fabricated feasibility.

Only verified gates permit throughput forecast and immutable protocol freeze,
then five methods (VQ/SL/RL/SG/RG) × three seeds, max 5 total FIT passes including warm-up.
Paired starts are cloned/hash-proven; max 10 STOP checks, patience and numerical/KL/
invalidity/accuracy-divergence stops remain. Best and last checkpoints are distinct.
All 15 outcomes must freeze in selection.json before guarded deterministic reporting.

Primary contrast RG–SG, secondary/exploratory RL–SL. Identical label scoring is
independent of grounded payload validity; localization is separate. Paired
recording-group bootstrap and seed spread are reported. A conservative exact-
binomial group-support upper screen additionally guards sparse No_Gun +.05
noninferiority; report raw interval and assumptions separately. Zero observed
errors alone cannot establish safety.

Original validation informed Phase 1. Metadata-only inspection found 13 nonempty
prior Training 1 test outputs; historical test is not a pristine new holdout.
No test images/boxes/predictions have been used in this execution. No held-out
recognition improvement or RL benefit is established.

The detached controller chains sweep→selection freeze→evaluation/report only
after passing gates. An active 15-minute in-chat continuation reads the canonical
outputs/CONTINUE_EXPERIMENTS.md and actual artifacts. Local follow-ups need Mac/
app running; remote jobs are independent. Deliver verified report/tables/plots,
update audit evidence/PR from Mac, and never merge. Running jobs are not completion.

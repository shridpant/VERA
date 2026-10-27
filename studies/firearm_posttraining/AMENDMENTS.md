# Protocol amendments

## 2026-10-26: current execution authorization

The current user explicitly requests both remaining RTX 6000 Ada GPUs, end-to-end
execution, Git commits, and a PR after verification. This supersedes the historical
one-GPU study / keep-Ada-7-free / no-push restrictions in the supplied handoff.

Use Ada 6 and Ada 7 only after a live idle check, with one device per independent
process, no distributed training. Use concurrency only for independent work after
the shared compatibility gate. Keep the 48 GPU-hour **aggregate** ceiling, three
GPU-hour pilot allocation, and per-run caps; concurrency does not double budgets.
Existing Phase 1 runs, environments, and output roots remain untouched. Push from
the Mac only. Open a review PR after actual checks; do not merge.

The historical HANDOFF.md remains the proposed scientific protocol. Numerical
settings remain provisional until the measured pilot and preregistration freeze.
Engineering failures and time-limited work must remain visible, with no invented
results, implicit test access, or automatic full sweep before gate approval.

## Annotation uncertainty policy

`listed_no_box` means an image record exists without an annotation; it is not
proof of audited firearm absence. Only `positive` and `negative_category` (or a
future explicitly audited `confirmed_absent`) enter box loss/reward. All other
statuses are masked. This applies the handoff's unknown-not-negative rule.

## Engineering attempt ledger

- `smoke-labels-001`, commit 09a1ed9: stopped after 25.71 GPU-seconds.
  Three fixture SFT losses fell 2.9651 → 2.8763 → 2.5573; deterministic generated
  answer reloaded exactly. Next-step resumed loss matched at 2.1716952, but
  next-step adapter hashes differed. This is an engineering failure, not a
  completed compatibility gate or a generalization result. Enable deterministic
  kernels and capture tensor-level differences before another bounded attempt.

- `smoke-labels-002`, commit afeea9e: deterministic kernels fixed exact resume;
  prediction reload and native forward/backward/reference scoring passed in
  44.26 GPU-seconds. Peak allocated memory 19,609,906,176 bytes. All four sampled
  rewards were equal. The centered synthetic advantages on identical strings
  almost canceled (gradient norm 4.6e-9), so this did **not** establish a meaningful
  policy gradient. Strengthen the engineering fixture with distinct explicitly
  synthetic completions and require gradient norm >1e-6. The real 24-video
  reward-spread/validity gate remains mandatory; synthetic fixtures never pass it.


## 2026-10-26: user removes real-run GPU-hour ceilings

This later authorization supersedes the GPU-hour limits earlier in this file and
in the historical handoff. The user said: “retain other stopping criteria but not
GPU hour. in pilots, extend but don't remove. in real, remove” and requested
frequent updates.

- Remove the 48-hour aggregate, 37.5-hour sweep, 2.5-hour method/seed, and 7.5-hour
  evaluation ceilings. Scientific sweep jobs, including fixed paired warm-ups,
  and final evaluation use `--seconds 0`; trainer/VQ use `--max-seconds 0`.
- Double the pilot allocation to six aggregate GPU-hours; supervised pilots may
  occupy at most four hours and each RL pilot at most two hours. Engineering
  smoke checks may retain their shorter bounds. Pilot reservations remain finite.
- Preserve patience, the common maximum FIT passes, maximum validation queries,
  fixed warm-up exposure, reward validity/variance gates, numerical/divergence
  checks, checkpoint integrity, idle-device checks, and sealed holdout guards.
- Throughput still informs ETA and GPU-use reporting. Do not reduce a scientific
  pass cap solely to squeeze a run inside the removed 2.5-hour allowance.
- Updates: meaningful progress during active work at least every 60 seconds;
  during long experiments target a status update about every 15 minutes, plus
  immediate notice of failures, stop gates, material results or required input.
  Report both GPUs, progress/metrics, actual GPU time, and revised ETA honestly.

Applied locally in the side conversation. Existing detached releases and already
running remote processes retain their launch-time settings. They were not killed,
restarted, remotely edited, or claimed to have new limits. The main execution must
use a new committed release for subsequent launches and explicitly reconcile any
still-running capped pilot before it reaches its old cutoff. No remote deployment
or recurring monitor was created by this side conversation.

The main chat then requested direct clarification after noticing the concurrent
edits. The user explicitly answered “Remove full-run caps and allow 6-hour
pilots” and confirmed “yes a side chat was doing that.” Authorization is now
confirmed in the main chat. Subsequent releases integrate this policy, including
unlimited-time final evaluation; existing pilot processes retain their original
shorter caps and need no restart unless their normal stopping fails to finish.

## Classification versus grounding scoring correction before comparison

Code review found the initial evaluator used full grounded-schema validity to
score category labels in SG/RG. This would conflate box-format errors with the
primary recognition metric, contrary to the handoff. Classification now uses
the identical three-label JSON parser for every arm, independently of the box
payload; full-schema invalidity is still reported separately. Malformed outer
JSON/truncation still receives no label credit. Strict RL rewards and box metrics
retain their full-schema checks. Tests prove a bad box payload cannot change
otherwise identical classification scores. The label-only supervised pilot is
unchanged; no comparative or holdout runs preceded this correction.

## Processor and VQ engineering before protocol freeze

The installed Instruct template ignores `enable_thinking=False` and warns on
every call. A CPU comparison verified byte-identical rendered templates with and
without it (SHA-256 371220223b59051bc1f36bdcf229b23f4fec18b245cddb935337180c908e937d
for the comparison fixture); remove only that unused argument. Frames are now
hash-checked against the immutable audit at every load.

VQ asks for an explicit zero-based slot and one replacement question instead of
a regenerated list. The runner applies exactly that single edit and retains all
other questions verbatim, enforcing the same maximum of five and ten candidate
revisions. No STOP examples enter this optimizer prompt. A two-FIT-video real
compatibility check exercises the optimizer context and proposal handling before
the frozen comparative VQ runs.

## FIT-only grounded format failure and warm-up calibration

`pilot-rg-001` at a fixed 0.5-pass grounded warm start produced 15 invalid answers
in its first 23 FIT responses (no truncation), including flat zero-box arrays
instead of frame objects and action descriptions instead of action codes. More
than four invalid answers makes the <=5% / 96-response gate impossible, so only
that owned pilot was signaled for a graceful checkpoint. Its traces and final
failed-gate result remain preserved; the parser and reward are not loosened.

Before protocol freeze, calibrate a stronger **shared fixed warm-up exposure**
using FIT-only output-format measurements, beginning at one pass. Apply the same
chosen exposure to label and grounded pairs and all seeds, count it inside the
unchanged five-total-pass limit, and recheck actual sampled-reward feasibility
from the resulting references. No STOP-based warm-start selection is allowed.
This revises a provisional numerical default, not supervision, split, backbone,
LoRA capacity, rewards, or the scientific comparison. Freeze the final value only
after both real-reward panels pass; do not scale a failed grounded arm.

The finite calibration grid is 1.0, 1.5, then 2.0 FIT passes. Choose the first
shared exposure for which both label and grounded sampled-format checks pass;
each check uses four samples on the same balanced 24-video FIT panel, no STOP,
and stops early if >4 invalid responses makes <=5% of 96 impossible. Then run
both actual reward panels at that same fixed exposure. A reward-panel failure
stops the controller for diagnosis, not a blind grid retry. Only two passing
real-reward panels permit protocol freeze and the downstream scientific pipeline.
All calibration time remains inside six aggregate pilot GPU-hours.

## Pre-freeze negative-sample support guard

No reporting outcomes have been opened. Code review found that a percentile
bootstrap returns [0,0] when both methods make zero observed No_Gun errors; that
cannot establish a +0.05 noninferiority margin with sparse negative groups.
Retain the paired bootstrap unchanged and additionally require a conservative
group-support upper screen <=0.05. A harmful group has any new false positive
across its clips/seeds; a uniformly beneficial group improves every paired clip
and seed. Use 97.5% exact binomial upper/lower endpoints for their probabilities,
combine them conservatively with observed maximum/minimum group-size weights,
and take the maximum of this screen and the raw bootstrap upper endpoint. Report
the raw interval, support screen and independent-group/observed-size assumptions
separately. Sparse evidence remains inconclusive; strong improvements can still
satisfy the screen. This screen is deliberately conservative, not a replacement
claim of an exact video-level interval. SciPy's exact endpoints use Clopper–Pearson:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html

The pre-freeze CPU controller was stopped to integrate this guard. Its GPU
warm-ups continue unchanged. Resume reuses matching settled/running artifacts,
does not repeat their queries, and preserves the earlier controller/action logs.

## Calibration prelaunch recovery

The first 1.5-pass label warm-up never created a job or budget reservation: a
stale 100% utilization sample remained after its preceding job released all GPU
memory. Its paired grounded warm-up completed 288 clips / 36 updates normally.
The first resumed CPU controller then stopped on the existing prelaunch log.
Preserve that log and only retry if no job, output or budget reservation exists.
Allow up to 30 seconds for utilization to settle while the device lock is held,
only if no compute process and <500 MiB are present. Occupied devices still fail
immediately. This changes orchestration, not scientific exposure or gates.
Regression checks cover cooldown, occupied-device rejection, prelaunch failure
recovery and reusing the completed paired warm-up without repeat GPU work.

## Pre-freeze target-mask correction (v2)

FIT format sampling at 1.5 passes failed 5/5 grounded responses, including
label-only JSON and duplicate frame keys. Actual pinned-tokenizer inspection
confirmed the original span-containment mask discarded known punctuation
across adjacent known spans: the label-to-grounding delimiter, frame-object
separators and closing JSON. Entire unknown frame objects were also masked,
although frame IDs and schema are known. Preserve these failures and the
tokenizer diagnosis. Pause only the CPU controller before further calibration.

Retain every token fully covered by known spans, including BPE tokens crossing
known boundaries. A token touching any unknown box payload stays fully masked.
Supervise known labels/frame schema with unit weight, known box payload with
0.5 weight, and no unknown payload. Keep the exact same target text, labels,
prompts, views, parser and rewards. This also corrects label-only closing-JSON
boundary supervision. The prior supervised curve remains exploratory legacy
evidence; do not imply its masking implementation equals the corrected one.

Run a CPU audit over all FIT/STOP targets using the actual pinned tokenizer,
proving unchanged target bytes, <=512 tokens and no masked known/unmasked
unknown tokens. Repeat label and grounded GPU engineering gates. Start a new
calibration namespace with unique attempts 005/006/007 at the same declared
1/1.5/2-pass FIT-only grid. Reuse no adapters trained under the old mask. Both
true reward panels must pass at the corrected common warm-up before protocol
freeze. All extra GPU work remains inside the existing six-hour pilot budget.

## Additional finite FIT-only warm-up candidate: three passes

The corrected 1/1.5/2-pass grid exhausted normally with no passing common format
exposure. Label validity was 94/96, 96/96, 95/96; grounded checks stopped at 5/20,
5/18, 5/64 invalid respectively. These partial-panel fractions are not complete
96-response estimates. The two-pass FIT failures contain missing JSON closure,
repeated answers/truncation and non-unit coordinates. No further target-mask
defect was found in those traces. Preserve the failed grid/controller status.

Before any scientific freeze, declare exactly one additional candidate: shared
3.0 FIT-pass warm-up, independent fresh label/grounded runs, no STOP queries.
Use the same fixed FIT panel, sampling, strict parser, rewards, thresholds,
backbone, LoRA and optimizer defaults. This is a numerical warm-up amendment
using FIT feedback, not reporting-set outcomes. Count three warm-up passes inside
the unchanged five-total-pass ceiling, leaving at most two continuation passes
for each supervised/RL pair and applying the same exposure to all three seeds.

Use new calibration namespace calibration-3pass-v2 and attempt 008. The immutable
controller config records grid [3.0]; resume rejects any grid change. No automatic
further candidate follows failure. Both format gates and both actual reward
panels must pass before freeze; failed actual reward gates still stop for
diagnosis. Existing mask-v2 adapters at shorter exposures are not reused.

Settled pilot use before this candidate is 2.91874 GPU-hours out of the authorized
six. Measured two-pass warm-up latencies forecast approximately38 combined
GPU-minutes for fresh three-pass warm-ups; reserve finite 1800 seconds each.
Format caps remain 600/900 seconds and actual panel caps 1800/3600 seconds, with
the existing aggregate reservation guard enforcing the unchanged six-hour
allowance. Do not extend that allowance silently. Full scientific-run time caps
remain removed; all non-time scientific stopping rules remain.

## Pre-freeze rollout probability correction

The three-pass candidate008 completed its SFT warm-ups without STOP queries
(label1082s, grounded1180s) and passed old-sampler format checks:95/96valid
labels and94/96valid grounded. Before protocol freeze, code audit found that
rollouts used temperature0.8/nucleus0.95 while policy/reference log-probabilities
used raw unscaled logits. The pinned checkpoint also supplied inherited top-k20.
The installed TRL1.16 GRPO trainer explicitly passes its temperature to both
policy/reference scoring (grpo_trainer.py1066-1076; trainer/utils.py699-704).
The native implementation must define one consistent probability policy.

The format check completed just before the CPU pause, so old panels008 had
started. Only that CPU controller and the two owned panel children were signaled;
SIGTERM lets them checkpoint/settle. Preserve their traces and operator actions.
Those panels are invalidated by implementation correction, not evidence of
scientific RL feasibility or failure under the corrected policy. No protocol
was frozen and no reporting outcomes were opened.

Use one full-support temperature0.8 policy: fresh explicit GenerationConfig,
num_beams1, top-p1, top-k0, repetition penalty1. Scale logits by the same0.8
for old/current/reference token probabilities and entropy. Greedy reporting
remains single-response argmax with the same cap. Full support avoids nucleus/
top-k support renormalization mismatch in clipped ratios and reference KL.
This changes the stochastic behavior distribution, so prior format gates cannot
substitute for new checks. Parser, rewards, <=5% invalidity, >=20% reward-spread,
divergence stops and six aggregate pilot GPU-hours remain unchanged.

Keep the fixed3pass candidate, without additional exposure search. Prove exact
source equality for SFT functions and trainer.py against a122820, verify its
completed warm-up source commit and zero STOP checks, then reuse those adapters
without retraining. Repeat the240-video actual-tokenizer audit as a new v3
artifact (do not overwrite v2), and both GPU engineering checks as005. New
format/reward attempt009 uses warm-up attempt008 in a fresh immutable namespace
calibration-probability-v3. No automatic retry or further warm-up candidate.
Only both new formats and both actual reward panels permit freeze and scaling.
Unit tests cover temperature-matched likelihoods/entropy/gradients, preventing
checkpoint filter inheritance, and warm-up reuse with new unique gate jobs.

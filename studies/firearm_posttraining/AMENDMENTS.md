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

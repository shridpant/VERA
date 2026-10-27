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

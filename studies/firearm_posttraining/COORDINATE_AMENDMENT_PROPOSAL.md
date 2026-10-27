# Grounded format gate: stopped, amendment proposed

The corrected-policy pilot failed its declared format gate. No scientific protocol was frozen, reward panels 009 never started, and reporting outcomes remain sealed. This is a feasibility stop, not a completed comparison.

## Verified evidence

- Label-only format: 95/96 valid, passed; 125.21 seconds.
- Grounded format: five invalid answers in 65 responses; stopped early because five invalid answers cannot meet the unchanged maximum 5% of 96. The partial 5/65 fraction is not a completed-panel error-rate estimate.
- All five failures are valid, untruncated JSON with valid label fields. Four contain coordinates outside the required unit range, all within 0–1000. One contains zero-height boxes. There are no termination failures among these five.
- Controller PID 293624 is dead. Both authorized Ada UUIDs were idle at inspection. No active pilot reservations remain.
- Settled use: 14,405.8741 GPU-seconds = 4.0016317 of six GPU-hours. Remaining: 7,194.1259 seconds = 1.9983683 aggregate GPU-hours. Full scientific-run time ceilings remain removed; other scientific stops remain.

The raw five responses, artifact SHA-256 hashes, immutable release commit, gate results and ledger totals are committed in `evidence/format-009-diagnosis.json`. Reproduce the read-only audit with `python3 evidence/audit_format_009.py ROOT RELEASE`, using release 2118b52 and the same source artifacts. No model inference or reporting data access is required.

## Interpretation

The [official Qwen3-VL grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/0dcc180d854f4b132f8059b10bbf0b5fd5dae9ed/cookbooks/2d_grounding.ipynb) documents normalized 0–1000 coordinates. The study instead prompts and supervises unit coordinates. This makes a representation mismatch a plausible explanation for four failures, but does not establish causation or prove that another pilot will pass. Unit coordinates were a valid custom contract; this is not an arithmetic conversion bug in the current parser. The degenerate-box failure would remain invalid under either convention.

The official source URL, commit, notebook hash and extracted-cell-source hash are recorded in `evidence/qwen-grounding-source.json`. No third-party mirror or held-out outcomes informed this diagnosis.

## Concrete proposed amendment — not yet implemented or launched

Use exactly one fresh, pre-freeze fixed three-pass candidate with model-facing 0–1000 coordinates for both SG and RG. Internal annotations, letterbox geometry and reward/localization calculations remain unit coordinates. Serialize training targets as the old five-decimal unit value multiplied by 1000, preserving equivalent spatial precision. The prompt declares the new scale; the strict parser validates finite coordinates, ordered positive-area boxes, unique frame IDs and all existing schema constraints, then divides coordinates by 1000 before the unchanged IoU/reward calculations.

Do not detect units from answer magnitude, clamp boxes, accept malformed JSON, repair degenerate boxes or retroactively rescore old failed gates as passes. Small unit-looking values are also legal on a 0–1000 scale, so a passing format gate alone cannot establish correct localization; actual reward spread and subsequent guarded localization reporting remain necessary.

Fresh paired label/grounded warm-ups would avoid relying on old-target adapters. Keep three warm-up passes inside five total FIT passes, the same seeds/backbone/LoRA/optimizer, the same FIT-only panel and full-support temperature-0.8 sampling, and all current thresholds/stops. No exposure grid expansion and no reporting-guided tuning. Old failures remain preserved. A failed new candidate stops again; no further automatic candidate.

Before GPU work: record the accepted amendment; implement the model-boundary conversion; test target-to-parser round trips, unchanged internal reward/IoU, boundaries, degenerate/nonfinite/duplicate cases, label-only equivalence and unknown masks; then audit the actual pinned tokenizer on all 240 development targets. Write a new immutable audit with new target hashes and source binding. Never claim target bytes or SFT source are unchanged. New GPU engineering checks and both format/reward gates are required before freeze. Deploy only a clean committed immutable release. Selection/reporting remain blocked until their gates pass.

Keep the existing six-hour aggregate pilot allowance and finite per-job limits. Expected fresh paired warm-ups cost about 0.63 GPU-hours from the completed three-pass measurements, plus engineering/format/panel work. This is a forecast, not a guarantee. The shared reservation guard is authoritative. Run reward panels sequentially if concurrent reservations do not fit; do not shrink panels, relax gates or bypass budget accounting. If a required reservation no longer fits, preserve state and request the exact additional allowance rather than silently extending it. Do not launch the full scientific sweep until all new gates and protocol freeze pass.

## Decision required

Approve this specific output-representation amendment within the existing pilot allowance, or retain the current contract and preserve the calibration stop. The fixed candidate's format stop remains in force while this decision is pending. No further GPU jobs are authorized by this proposal alone. The study and PR remain incomplete; no held-out benefit or scientific RL feasibility is established.

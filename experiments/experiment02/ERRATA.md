# experiment02 errata

These are corrections recorded after the runs. The run reports, scoring contracts and raw records stay as they
were written.

## E1 (2026-10-23): a sub-step change in S_hi can still move the deployed bf16 score

**Statements this corrects:**
- [`runs/run_002/scoring_contract.md`](runs/run_002/scoring_contract.md), *S_hi: the finer readout*: "S_hi
  differences below 0.25 nats are real changes in the hidden state, but they cannot change the deployed bf16 output
  probabilities."
- [`runs/run_002/REPORT.md`](runs/run_002/REPORT.md), §10 *Limitations*: "S_hi refines the readout but cannot change what the
  bf16 model would output."
- The REPORT's "sub-resolution" wording for N-middle: "every per-frame effect is smaller than the model's numerical
  resolution" (*Plain-language summary*), and "every single-frame and group effect is below the model's numeric resolution"
  (§7 *Answers to the brief's questions*).
  These are true of S_hi's magnitude only.

**Correction.** The model rounds each logit to bf16 on its own: S = logit('1') − logit('0') with both logits
rounded to the 0.25 grid. A perturbation that moves S_hi by less than 0.25 can still push one logit across a
rounding boundary. The native S, and the bf16 probabilities P('0') and P('1'), then move by a full step. The
reverse also happens: an S_hi change of 0.25 or more can leave the native S unchanged. The two deltas can even
have opposite signs.

Example: R-accident, blur on slot 0. S_hi goes 2.855 → 2.986 (Δ −0.131, below one step), while the native S goes
2.75 → 3.00 (Δ −0.25).

"Below resolution" (the `Delta_below_resolution` flag) therefore describes the size of the S_hi change. It does
not guarantee that the deployed output is unchanged.

**Measured on run_002.** This is computed from the existing `results/temporal_effects.csv`, CPU only, with no new
inference. Script: [`errata/precision_crossings.py`](errata/precision_crossings.py); output:
[`errata/precision_crossings.json`](errata/precision_crossings.json).

| Case (54 non-all-eight masks each) | \|ΔS_hi\| ≥ 0.25 | native S moved | crossings: \|ΔS_hi\| < 0.25 but native S moved | \|ΔS_hi\| ≥ 0.25 but native S unchanged | max \|ΔS_hi\| |
|---|---:|---:|---:|---:|---:|
| R-accident | 37 | 43 | 6 | 0 | 3.95 |
| R-pedestrian | 16 | 33 | 19 | 2 | 0.71 |
| N-middle | 0 | 11 | 11 | 0 | 0.23 |

Over all 168 effect rows (all-eight masks included):
- 109 have |ΔS_hi| below one step, and 36 of those still moved the native S.
- 2 rows have |ΔS_hi| of a step or more while the native S is unchanged.
- In 4 rows (two R-pedestrian, two N-middle; all with |ΔS_hi| ≤ 0.115) the native and S_hi deltas have opposite
  signs.
- The stored `Delta_below_resolution` flag agrees with a recomputation in 168/168 rows.

**What this changes:**
- **N-middle.** S_hi effects are below 0.25 (max 0.227). However, the native S moved by one step in 11 of 54
  non-all-eight masks (3 blur, 8 constant). No label flipped.
- **R-pedestrian.** The REPORT's "16 of 54 masks exceed 0.25" counts S_hi changes. The native S moved in 33 of
  54.
- **S_hi's role.** Effects, interactions, rankings and τ_b were declared on S_hi, and those computations stand.
  Read S_hi as a finer readout of the hidden state, not as a finer view of an unchanged deployed output. Native
  bf16 differences (`Delta_bf16`) are already reported alongside in every effect row.

**What it does not change.** No raw record, CSV or summary is edited. The binary-label results come from the
original generation, not from S, and are unaffected. The run_002 validator (18/18) checks implementation
properties, and none of them depends on the corrected sentence.

**For later work.** Report native bf16 differences next to any fp32 readout. Do not justify thresholds by saying
a sub-step change "cannot change the output".

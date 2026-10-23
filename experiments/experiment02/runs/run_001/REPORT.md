# experiment02 / run_001: Stage A under config v1 (superseded by run_002)

**Status: Stage A only.** The pre-declared v1 gate classed the numeric scorer as **blocked**.

- Under the v1 code and the brief ("continue original-label tests"), Stage B would still have run original-prompt
  generations without numeric scores, and Stage C would have been skipped.
- **Stopping run_001 after Stage A was my decision, not a rule.** I reran everything under config v2 in run_002 instead.

## Pre-declared rules (`resolved_config.yaml`)

- The v1 rules were fixed before any model output.
- Gate criteria were tightened after the code review, before any model call (`run_manifest.json` → `config_amendments`).
- The code that produced this run is in `code_snapshot/`. Its hashes match `code_sha256_frozen_at_first_model_invocation`.

## Results

**A1 reproduction.** Rerunning VERA's original prompt reproduced experiment01 exactly for R-start, R-pedestrian,
R-accident and R-end: raw text, labels and token ids. N-middle got label 0.

**A2 scorer gate.** Exactly one criterion failed: "sign(S) equals the label-only generation label on all 5 cases".

- On R-pedestrian both candidate logits were 36.0, so S = 0.
- v1 maps S ≥ 0 to anomaly; greedy decoding chose '0' (the lower token id).

All other criteria passed:

- alignment with `generate()` exact (max logit difference 0.0);
- repeatability exact;
- candidate mass 0.82–0.91;
- all six all-eight replacements moved S by 0.75–3.62.

Stage A also showed that S moves in 0.25-nat steps while both logits are ≥ 32 (bf16 at |logit| ≈ 36). That holds
for every unperturbed input. The three all-grey inputs push logit('1') to 31.625, which is on a 0.125 grid; hence the
change of 3.625.

**A3 sound ablation.** Removing only the sound question flipped R-pedestrian from 1 to 0; the other labels were unchanged.

## Cost

- Job: `exp02_A_20261022T200546_18974`, 218 s, device peak 20,422 MiB (19.9 GiB).
- Calls: 15 generations, 21 scores, 5 alignment checks.
- 6 transformed sequences (the all-eight inputs).

## What happened next

run_002 reran everything under config v2. Its post-hoc changes are disclosed in `../run_002/REPORT.md` §3:

- exact ties are undecided;
- the scorer status is capped at provisional;
- effects use the S_hi readout.

This run was not passed through `validate_attr.py`, because it has no Stage B/C outputs by design. Its Stage A records
remain in `results/` unchanged.

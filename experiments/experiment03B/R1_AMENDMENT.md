# R1: post-data integrity audit and reporting

Authorized October 26, 2026. This is an additive CPU-only audit, not a new model run or a change to the prespecified scientific endpoint. Original `runs/full_v1` scores, labels, rankings, tables, configuration, code hashes, failed validation and ledger remain byte-identical. The historical handoff is retained as historical text; its pending-action sections are superseded here.

## Four explicitly amended checks

1. **score_rounding:** original validator required exact BF16 rounding, whereas the already-frozen experiment02 engine at lines 250–251 checks `abs(hi-native) <= bf16_spacing(native)/2 + 1e-3`. R1 checks the historical tolerance, finite values, representable native BF16 logits, flags, and stored S_hi subtraction. All 45 exact-rounding mismatches are retained in the audit JSON. Maximum half-step excess is 0.0001716614. We do not relabel these as exact rounding or modify scores. The original engine flag name is misleading and remains unchanged for provenance.
2. **slot_hashes_present:** verify eight valid SHA256 strings and recompute the legacy flag (exactly masked slots changed); a correct false flag is meaningful, not missing provenance.
3. **slot_identity:** hash-verified frame zero from Explosion039 and RoadAccidents128 is black in every channel. Blur radius 17.92 preserves it exactly. Permit only such explicitly proven black-frame blur no-ops, affecting 20 records. Changed unmasked slots, unproved masked no-ops, grey no-ops, and inconsistent replacement hashes still fail.
4. **mask_hash_consistency:** names are not content identity. Require identical source video SHA, frame indices, source frame hashes, replacement family/mask and resulting tensor hashes whenever sequence keys are shared. This explains 88 keys associated with Normal_Videos_936 and 937. Both archived MP4 files were independently SHA256-hashed on Oz on the audit date: `9655db1270be2e0add48e627caeb2cc1767b7ca13c2ef84baf9e7898a9f7e8a9`.

R1 re-executes the original independent validator, retains its 16-pass/4-fail result in `original_validation_recomputed.json`, and combines its sixteen unchanged checks with these four amended checks: **20 passed, zero failed or skipped**. The R1 reconciliation is authored with knowledge of the failures; it must not be described as a fresh blinded independent review.

## Result impact

Primary remains +15.16563147 pp, 95% CI [9.85992495, 20.60041408], 199 alerts / 138 videos. No score, label, rank or original result changed. Duplicate sources are outside the primary population. A separately labeled source-deduplicated sensitivity reduces secondary independent source count 288→287 and numeric-margin mean 0.35563037→0.35591913 nats. Its post-data bootstrap CI is [0.31832414, 0.39474595]. Source-hash ordering changes bootstrap draw assignment, so sensitivity intervals need not match the original Monte Carlo draws. Other sensitivity metrics and fixed/adaptive rates are in `duplicate_sensitivity.json`.

## Preservation and provenance

`audits/full_v1_r1/source_sha256.json` covers all 31 portable record files of the source-run copy, including cache, frozen metadata, raw requests, original results and failed validation. The audit asserts identical hashes before and after. The ephemeral run.lock and four dataset-derived example files are excluded from the portable manifest and Git, as required by the existing repository rules; all 36 local files are checked before/after and retained in source_allfiles_sha256.json; the first snapshot is also retained. No files in Oz's original run were written. Source frames used for pixel proofs remain outside Git. Raw original run records are now committed, closing the previously untracked-result gap.

## Validation performed

- Original independent validator tests: 37 passed.
- R1 regression tests: 11 passed, including deliberately corrupt readouts, unmasked changes, false legacy flags, grey/unproved no-ops, and same-key/different-content collisions.
- Workflow suite: 16 substantive checks passed locally; its legacy harness prints 17/17 even when the pixel check returns a skip dictionary. The missing pixel check was actually run on Oz with its source frames and passed (separate JSON).
- Initial local workflow attempt failed because the pinned source checkout was absent. Restoring the documented 008ed401 worktree fixed the setup; no workflow code change was needed. The first attempt is preserved in `workflow_tests_initial.json`.
- Report .tex compiles in the desktop LaTeX compiler. Figures are generated from stored results, not manually drawn estimates.

## Reproduce (from repository root)

Dependencies: Python 3, Pillow for source-frame proofs; matplotlib for report figures. Exact versions from the audit/report environment are recorded in `software.json`. No model/GPU is required.

```sh
# Restore the historical source for workflow tests (ignored dependency).
git worktree add --detach setup/source/VERA 008ed4019781c39e7cb16244b7500244f3986f92
python3 experiments/experiment03B/code/tests/test_validate3b.py
python3 experiments/experiment03B/code/tests/test_audit3b_r1.py
python3 experiments/experiment03B/code/tests/run_tests.py --out /tmp/workflow-tests.json

# Use a NEW audit output directory; existing outputs are refused.
python3 experiments/experiment03B/code/audit3b_r1.py \
  --run-dir experiments/experiment03B/runs/full_v1 \
  --out /tmp/experiment03b-r1-reproduction \
  --frames-root shared/frames/ucf_crime_sparse \
  --exp02-run-dir experiments/experiment02/runs/run_002
python3 experiments/experiment03B/code/report3b_r1.py \
  --run-dir experiments/experiment03B/runs/full_v1 \
  --audit-dir /tmp/experiment03b-r1-reproduction \
  --out /tmp/experiment03b-r1-report
```

On a Mac without dataset frames, copy the two implicated `000000.jpg` files into a separate directory preserving their video subfolders and pass that root. The audit verifies hashes against original inputs; no trust in filenames alone. On Oz the complete sparse root is already staged.

## Scope of conclusion

Evidence supports local frame-ranking utility under specified perturbations. It does not demonstrate detection improvement, human-valid support, semantic faithfulness or natural missing-frame behavior. Adaptive ranking incurs extra calls. A subsequent question-learning study needs distinct development material and its own novelty assessment; this amendment launches no new experiment.

Frozen CSV files use CRLF and historical logs contain final blank lines. These produce whitespace-lint findings; they are intentionally preserved. `.gitattributes` disables text normalization for this frozen run. Source/document changes are checked separately, and every staged scientific file is verified byte-for-byte against its SHA256 manifest before commit.

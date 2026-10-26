# Firearm VERA package: execution update (2026-10-26, 19:31 ET)

Setup is complete. Phase 1 is running; no gate decision is available yet. Phase 2, importance, package assembly and the report remain unbuilt. Nothing was pushed or merged.

## Verified

- Both environments report CUDA available; all 36 InternVL3.5-38B files passed verification.
- The non-test manifest has 1,003 frames: 394 Handgun, 433 Machine_Gun, 176 No_Gun, with nested 300/100 subsets.
- All seven initial format settings completed (60 real calls). InternVL3.5 probes were repeated after the SDPA correction; the 38B off-probe also completed (12 calls).
- Qwen3-VL Instruct completed 1,003/1,003 E1 frames and InternVL2 packed completed 302/302 E2 videos, with zero call failures. These are completion counts, not gate conclusions.
- All 48 firearm tests pass. The completion checker was verified against missing records, failed jobs and complete inputs.

## Corrections and declared decisions

- 61ae6e9 corrected the 38B thinking ranking exclusion, per-category target indexing, and duplicate retry accounting before Phase 1.
- 019766e and 47c867a fixed observed A16 memory failures: explicit weight-memory ceilings and the intended SDPA language backend. InternVL remote code had silently selected eager attention. Preliminary InternVL records are preserved under `package/diagnostics/pre_sdpa/` and excluded from analysis.
- The repeated SDPA probes returned a parseable firearm box with the detection prompt for 8B thinking off, but none for thinking on. Under the predeclared rule, the thinking arm and its repeat now use the referring prompt (1cc7086). The existing parser still marks a thinking response without a closed trace as truncated; this can mean protocol noncompliance rather than token-limit exhaustion. See DEVIATIONS.md.
- The 38B off-probe returned parseable firearm boxes and retains the detection prompt. Its originally omitted thinking-on probe is now required before either 38B ablation arm. Pipeline 087aa42 applies the same predeclared prompt rule, records both decisions before detection, and checks successful record completeness between retries.
- No threshold, seed, model weight, frame selection or scoring parser was changed.

## Active jobs on oz

The run checkout is `/local/datasets/sp2117/firearm_visilab/code/VERA-package-runs`, at 087aa42. The Ada 5/6 jobs began at 61ae6e9 and continued through the compatible loader/analysis corrections; current InternVL jobs use 087aa42. Local later handoff commits do not change running code.

All job directories are under `/local/datasets/sp2117/firearm_visilab/package/logs/`.

- Ada 4, InternVL detector: `firearm_package_e1a_probed_20261026T193049_258688`.
- Ada 5, recognition: `firearm_package_e2_20261026T191611_256791`.
- Ada 6, Qwen detectors: `firearm_package_e1b_20261026T191612_256838`.
- A16 indices 0, 1, 2, 3, 8, 9, 38B probes and detectors: `firearm_package_e1c_probed_20261026T193050_258735`.
- CPU dependency job: `firearm_package_gates_probed_20261026T193051_258781`.
- Ada 7 remains free. The A16 job records usage samples for all six cards.

The CPU job waits for all four jobs, verifies every expected frame/video, then runs `analyze1`. Its status is `package/analysis/completion.json`; results will be `package/analysis/phase1.json`. The committed checker is `jobs/finish_phase1.py`; the launched copy is `package/finish_phase1.py`.

`package/phase1_submission.json` records current jobs and superseded attempts. Do not double-launch. Inspect job exit_status, records and completion.json before interpreting gates or restarting. The 38B pipeline will append its probe decisions to the remote DEVIATIONS.md before ablation; preserve those notes and bring them back to the Mac before any future code sync. That expected documentation change can make the remote checkout dirty.

## Next

Allow the active jobs to complete, inspect completion.json and phase1.json, then apply G1/G2 before building Phase 2. The running CPU dependency job does not send notifications.

---

## Original handoff (2026-10-26, about 17:45 ET; historical)


## Where things are

- **Branch:** `experiment/firearm-package` in the Mac worktree `~/Documents/research/VERA-package`.
  - It was cut from Training 1 (`c239dc0`) and is at `a13b3c1`.
  - Not pushed.
  - `.venv` is a symlink to VERA-training's and is excluded from git.
- **Plans:**
  - [`docs/superpowers/specs/2026-10-26-firearm-package-design.md`](../../docs/superpowers/specs/2026-10-26-firearm-package-design.md):
    the full plan, approved by the user.
  - [`…-phase1-rules.md`](../../docs/superpowers/specs/2026-10-26-firearm-package-phase1-rules.md):
    the gates G1/G2, declared before any result.
- **Oz:**
  - run checkout `/local/datasets/sp2117/firearm_visilab/code/VERA-package-runs`, at `2812bdd`;
    re-sync it before running;
  - outputs in `/local/datasets/sp2117/firearm_visilab/package/`;
  - logs in `package/logs/`.
- **Sync:** the session scratchpad script `sync_package.sh [runs|analysis]` bundles the branch to
  `code/VERA-package-<which>`.

## Done

- **Phase 0 downloads:** started at about 16:40 ET, pinned and verified (`setup/fetch_all.sh`).
  - Done: Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking and Qwen3.5-9B.
  - Still running: InternVL3.5-38B (71.5 GiB).
  - InternVL3.5-8B and InternVL2-8B were already on oz.
- **Environments (`setup/make_envs.sh`, still running):**
  - `vlm4`: torch 2.8 + transformers 4.57.6, for InternVL3.5;
  - `vlm5`: torch 2.14.1 + transformers 5.21, for Qwen;
  - both on `/local/datasets/sp2117/envs`, no flash-attn (SDPA).
  - The `vlm4` pip install was slow, sharing bandwidth with the download; check `logs/make_envs.log`.
- **Code (46 firearm tests pass on the Mac):**
  - `vera_runner/vlm.py`: model access;
  - `vera_runner/detector.py`: prompts, parsers, IoU;
  - `studies/firearm_package/{manifest,e1,e2,analysis_phase1}.py`;
  - `run_firearm_package.py`: `manifest`, `probe`, `detect`, `recognise`, `analyze1`;
  - jobs: `probes.sh`, `phase1_e1a.sh` (Ada 4), `phase1_e2.sh` (Ada 5), `phase1_e1b.sh` (Ada 6),
    `phase1_e1c.sh` (A16 pool).

## Next steps, in order

1. **Wait for setup:** both environments print `torch … True` in `make_envs.log`, and
   `fetch_all.log` ends with DONE for the 38B.
2. **Re-sync and build the manifest:** run `sync_package.sh runs`, then on oz:
   `$PY4 run_firearm_package.py manifest`.
3. **Probes:** launch `jobs/probes.sh` on one Ada via `launch.sh` with an absolute path.
   - These are the first real calls, never run before.
   - Check InternVL3.5's box format, the Qwen3.5 coordinate scale and the thinking split.
   - Fix the parser or prompt as the rules allow, and record each change in
     `studies/firearm_package/DEVIATIONS.md`.
4. **Phase 1, launched separately with `launch.sh`:**
   - `phase1_e1a.sh` on Ada 4 (GPU-6a490dc2);
   - `phase1_e2.sh` on Ada 5 (GPU-b6901777);
   - `phase1_e1b.sh` on Ada 6 (GPU-2931eba2);
   - the A16 pool: `--gpu none -- env CUDA_VISIBLE_DEVICES=<6 A16 UUIDs> bash .../phase1_e1c.sh`;
   - Ada 7 stays free.
5. **Gates:** run `run_firearm_package.py analyze1`, which writes `package/analysis/phase1.json`.
6. **Not built yet:**
   - Phase 2: loop v2 (`verbalized_v2.py`), 3 arms × 3 seeds;
   - importance (`importance.py`, reusing `vera_runner/reliance.py` from `experiment/firearm-grounding`);
   - package assembly;
   - Phase 3: one test run, the LaTeX report, VERA Lab, and the PR into `feature/configurable-runner`.

## Untested risks

- The vlm.py paths have never touched a real model:
  - InternVL `chat`/`batch_chat` with `num_patches_list` on 3.5;
  - Qwen `apply_chat_template` with PIL images and `enable_thinking`.
- E2's `iv2_*` arms run in VERA's environment (transformers 4.37) through vlm.py; untested there.
- Thinking arms cap at 4,096 tokens; watch the truncation counts.
- Memory rules still apply: push only from the Mac; never merge into configurable-runner or
  vera-lab without the user.

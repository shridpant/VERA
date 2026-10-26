# Firearm VERA package: handoff (2026-10-26, about 17:45 ET)

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

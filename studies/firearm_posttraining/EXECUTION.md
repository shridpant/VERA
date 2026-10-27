# Actual execution status

The current user authorized **both idle Adas**, commits and a review PR. See
AMENDMENTS.md. The original HANDOFF.md is the historical scientific proposal,
not proof of implementation or permission to override the current request.

## Verified

- Isolated Mac checkout and branch `experiment/firearm-posttraining`, based on
  85e7aba. The user's original checkout has not been modified.
- Separate oz checkout `/local/datasets/sp2117/firearm_visilab/code/VERA-firearm-posttraining`.
  Runs use immutable detached releases under `code/posttraining-releases/`.
- Outputs `/local/datasets/sp2117/firearm_visilab/posttraining1`; original Phase 1
  jobs, environments and output paths are separate. Remote Phase 1 probe decisions
  have been preserved in the branch's prior-study deviations file.
- FIT 192 / STOP 48 from original TRAIN 240; no recording/source overlap among
  original train/validation/test metadata; no exact-frame FIT/STOP overlap.
  Validation has 62 videos and has historically informed Phase 1.
- All 1,920 FIT/STOP frame files verified, original base-weight files hashed,
  longest target 311 tokens (512 cap). Unknown box spans are masked.
- Native Qwen3-VL-8B BF16, rank-16 language-attention LoRA fits on one Ada.
  Label and grounded engineering checks passed on Ada 6 and Ada 7, respectively:
  finite backward, reduced fixture loss, adapter-only updates, exact prediction
  reload and next-update resume, four sampled responses and generated-token
  log-probabilities, and explicit frozen nonzero warm-start reference.
- Four fixture sampled rewards were identical. Distinct **synthetic engineering
  strings** verified materially nonzero GRPO gradients. This does not establish
  scientific RL feasibility; the 24-video FIT reward-spread/validity panel remains
  mandatory after the supervised pilot.
- Engineering results and data audit are in `evidence/`; resolved dependency lock
  includes 78 exact distributions and wheel hashes. Offline wheels stay on oz.

## In progress / pending

- `pilot-sl-001`: bounded exploratory supervised pilot on Ada 6, code 635e418,
  baseline STOP then checks each 0.5 FIT pass, best checkpoint and fixed stopping.
  Hard allocation 5,400 seconds, below the proposed two-hour ceiling.
- `pilot-c0g-001`: completed fixed 0.5-pass exploratory grounded warm-up on Ada 7,
  96 distinct clips, 12 updates, 203 seconds; no STOP selection.
- The pilot baseline has 50% strict-format invalid outputs on STOP. Keep this
  visible alongside accuracy; do not broaden the parser after seeing STOP.
- First half-pass STOP balanced category accuracy 70.6%, invalid outputs 0%.
  At one pass balanced accuracy 74.8% but No_Gun false positives 28.6% (4/14),
  versus 0% at baseline/half-pass. These are exploratory selection-set results;
  higher overall accuracy does not establish a safer or better final model.
- Real-reward RL panels, shared-pass throughput forecast, protocol freeze,
  comparative methods/seeds, final selection freeze, holdouts and report are
  **not complete**. No recognition improvement or RL benefit is claimed.

## Reproduction and audit

`python -m studies.firearm_posttraining --help` lists the implemented entrypoints.
Use `setup.sh` with the committed hash lock on matching Linux/Python 3.12;
`setup.sh --offline` uses `posttraining1/setup/wheels`. The script refuses an
existing environment. Do not run it over the current environment.

`execute.py` checks an authorized UUID is idle, locks the device, reserves shared
aggregate GPU-seconds, enforces a hard subprocess deadline, and preserves command,
commit, process IDs, stdout/stderr, GPU samples, exit reason and budget settlement.
The aggregate ceiling is 48 GPU-hours, with three for all pilots combined;
concurrent occupancy sums across cards. A crash leaves its full reservation
charged until explicitly reconciled. There is no automatic failure resubmission.

Every run writes its spec before loading weights, and requires a clean committed
checkout. Checkpoints contain adapters, optimizer/scheduler state, Python/NumPy/
Torch/CUDA RNG, sampler cursor and order seed, selection state and hashes. Base
weights are referenced, not copied. Native training freezes vision/projector,
embeddings, head and all non-LoRA tensors. The reference copies the actual warm
adapter and is independently hash-checked.

The comparison and holdout entrypoints fail closed on missing protocol/selection
freezes. STOP remains a selection set. Test identity metadata has been audited;
test images, boxes and predictions have not been used in this execution. Nonempty prior Training 1 `test.jsonl` outputs establish historical test access
from file metadata only. This is not a newly pristine test.

## Validation

The existing firearm suite passed 60 tests after the initial 12 additions, using
the source repository's provisioned interpreter outside the sandbox because its
OpenMP runtime could not open shared memory inside the sandbox. The focused suite subsequently passed 18 tests locally; the earlier 15 passed
on oz. The full firearm suite then passed 66 tests at the review snapshot.
VQ, sweep orchestration, and final holdout paths remain unexercised on real data
until their protocol gates pass.

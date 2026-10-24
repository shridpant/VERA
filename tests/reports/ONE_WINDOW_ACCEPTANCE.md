# Real-model acceptance: PASSED on CUDA and MPS (2026-10-24)

`tests/run_tests.py --suite real-model`, unchanged pass criteria: one completed
nonsynthetic run, one raw response, a strict 0/1 label, captured input token IDs and
all eight saved input images. Scope: RoadAccidents127_x264, opening window (anchor 0),
original eight frames, 448-square inputs, baseline, unchanged VERA prompt, greedy
generation (maximum 1024 new tokens), verified InternVL2-8B checkpoint at revision
6fb9ad69 with its MODEL_PROVENANCE.json checks.

| Machine | Device, precision | Wall time | Generation | Label | Input tokens | `real_model_passed` |
|---|---|---|---|---|---|---|
| Oz, RTX 6000 Ada | CUDA, BF16 | 78 s | 4.9 s | 0 | 2487 | true |
| MacBook Pro, M3 Max, 128 GB | MPS, FP16, CPU fallback | 28 s | 16.5 s | 0 | 2487 | true |

Both runs gave the same label and the same tokenized model input. The reports are local
and ignored by Git (`tests/reports/real-model/20261024T201741184491Z/` on Oz,
`tests/reports/real-model/20261024T202210383031Z/` on the Mac). The first Oz attempt
completed inference but errored afterwards: the test wrote `last_console.log` into
`tests/reports/real_model/`, which a fresh checkout does not have. The test now creates
that folder.

## Beyond one window (VERA Lab, same runner)

- Full video on Oz with VERA's original settings (CUDA, BF16, FlashAttention, greedy,
  1024 new tokens), the lab's defaults since 03e72a5: RoadAccidents127, all 162 responses
  are word for word identical to experiment01, with the same 13 windows labelled 1,
  frame ROC-AUC 0.9949414519906323 and AP 0.9183852264069265, in 7.7 minutes.
  Evidence: `~/vera_research/experiments/vera_lab_acceptance/2026-10-24_paper_defaults_RoadAccidents127_x264/` on Oz.
- The same video without FlashAttention and with 512 new tokens (the lab's earlier
  settings) took 17 minutes and differed in one window (anchor 496: 0 instead of 1), with
  the same AUC; VERA rounds refined scores to 0.1. Evidence:
  `~/vera_research/experiments/vera_lab_acceptance/2026-10-24_full_video_RoadAccidents127_x264/`.
- Warm model, two consecutive windows at 72 s and 74 s (labels 0 and 1 on both machines):
  Oz CUDA 31.0 s then 4.6 s; Mac MPS 35.3 s then 11.1 s. The second window reuses the
  loaded model.

## History: not passed on 2026-10-23

Earlier attempts on a Mac, with a RoadAccidents011 window and BF16, did not pass. MPS was
inaccessible to the agent process, and a CPU run timed out at the 300-second limit
without a complete response. An older vanilla run returned a label after about 97
minutes; it was never counted as a pass. Those artifacts stayed local to that machine.

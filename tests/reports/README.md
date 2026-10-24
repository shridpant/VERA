# Test evidence

`REPORT.md` and `runner-tests.log` are historical evidence from the first seven-test suite, before categorization. They are retained as history.

New invocations of tests/run_tests.py save timestamped tests.log and summary.json under a suite directory. Generated report directories and real-model artifacts are ignored by Git; no model data or video frames are pushed.

Latest results (2026-10-24, Mac and Oz):

- Categorized default suite: 40 discovered, 39 passed, real-model acceptance explicitly skipped. It now includes the `lab` suite (`tests/demo`).
- Explicit real-model suite: passed on Oz CUDA (BF16) and on the Mac with MPS (FP16). See ONE_WINDOW_ACCEPTANCE.md.

Read the exact timestamped log for details. Passing mock integration does not establish real-model end-to-end success.

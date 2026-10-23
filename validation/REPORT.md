# Validation

Base: vanilla fork commit 15b8bcb8574a977c229c577f50bfe6f06d07106e.

Executed locally on 2026-10-23:

- 7/7 automated tests passed; output is in runner-tests.log.
- The actual upstream forward method with a fake model produces the exact prompt used by the adapter.
- Default windows match the upstream arithmetic for four lengths, including boundary windows.
- Blur and grey pixels match the specified PIL operations; only selected slots change in probe mode.
- Default refinement equals unmodified compute_auc.py on a synthetic 700-frame input.
- A synthetic video completed preprocess/infer/probe and a second new run completed preprocess/infer/refine/evaluate with the mock backend. These are plumbing checks, not model predictions.
- RoadAccidents011 decoded completely: 2159 frames at 30 FPS. The new runner produced its coverage and video identity records under ignored runs/preprocessing_validation/run 1.
- Original generate_initial_score.py, compute_auc.py, Preprocessing/ and Inference/ are identical to vanilla origin/main.
- Real MPS prerequisite check failed explicitly because MPS is inaccessible in this agent's process. No new large model inference was launched. The prior Mac inference issue remains unresolved; this branch does not claim to fix hardware/precision performance.

Not validated: real InternVL2 generation through this new adapter, CUDA/MPS numerical equivalence, other model families, or full-dataset metrics. Optional refinement requires correctly aligned precomputed features. The Mock backend does not substitute for these checks.

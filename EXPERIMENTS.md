# Experiment registry

| ID | Purpose | Config | Runs | Status | Result |
|---|---|---|---|---|---|
| experiment01 | One-video end-to-end validation of the VERA setup (Steps 1–3) on UCF-Crime `RoadAccidents127_x264` (appendix B.5). Setup check, not a research result. | [`experiments/experiment01/config.yaml`](experiments/experiment01/config.yaml) | `run_001` (primary), `run_002` (fresh rerun) | **Complete** (2026-10-22) | 162/162 windows parsed; 13 labelled 1 (two isolated, at 16.5 s and 23.5 s; the rest a cluster at 70.9–78.9 s). The UCF annotation is 72.0–76.7 s. Final frame scores peak at 0.388 (73.6 s). Frame AUC on this video alone: 0.995. Validator 15/15. run_002 reproduced run_001 exactly. See [`run_001/REPORT.md`](experiments/experiment01/runs/run_001/REPORT.md). |

`experiment02` and later IDs are reserved. Add a row and a folder only when an experiment is actually requested.

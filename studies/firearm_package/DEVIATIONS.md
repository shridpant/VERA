# Phase 1 execution notes

## 2026-10-26: analysis corrections before Phase 1

Code review during the format probes found three implementation errors. Corrected the analysis to exclude `internvl35_38b_on` from detector ranking as declared, compare per-category E2 predictions with the category target rather than the action-code target, and count each successful frame/video only once if retry records exist. These corrections implement the existing rules; no gate, prompt, dataset, or threshold changed. The firearm unittest suite passes (47 tests).

The oz checkout was fast-forwarded to cb0c5ed. Both environments reported CUDA available, and all 36 InternVL3.5-38B files were verified. The manifest contains 1,003 frames: 394 Handgun, 433 Machine_Gun, 176 No_Gun; subsets contain 300 and 100 frames.

Format probes started on Ada 4 with launcher job `firearm_package_probes_20261026T191120_255757`. The detection prompt returned a parseable firearm box for InternVL3.5-8B with thinking off, so the predeclared prompt-switch condition was not met for that arm.

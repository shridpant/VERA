# Phase 1 execution notes

## 2026-10-26: analysis corrections before Phase 1

Code review during the format probes found three implementation errors. Corrected the analysis to exclude `internvl35_38b_on` from detector ranking as declared, compare per-category E2 predictions with the category target rather than the action-code target, and count each successful frame/video only once if retry records exist. These corrections implement the existing rules; no gate, prompt, dataset, or threshold changed. The firearm unittest suite passes (47 tests).

The oz checkout was fast-forwarded to cb0c5ed. Both environments reported CUDA available, and all 36 InternVL3.5-38B files were verified. The manifest contains 1,003 frames: 394 Handgun, 433 Machine_Gun, 176 No_Gun; subsets contain 300 and 100 frames.

Format probes started on Ada 4 with launcher job `firearm_package_probes_20261026T191120_255757`. The detection prompt returned a parseable firearm box for InternVL3.5-8B with thinking off, so the predeclared prompt-switch condition was not met for that arm.

## 2026-10-26: probes and A16 memory placement

All seven initial probe settings completed (60 calls, launcher exit 0) without runtime exceptions. Every InternVL setting produced at least one parseable firearm box with the detection prompt, so no referring-prompt substitution was made. No Qwen probe returned coordinates above 1,000, so no coordinate scaling was introduced. The main thinking prompts produced closed thinking traces; the alternative InternVL referring prompt sometimes returned no trace and was marked truncated under the existing parser. Qwen3.5 uses its reference PyTorch linear-attention kernels because optional fused kernels are absent; this affects runtime.

Phase 1 started at 19:16 ET from 61ae6e9. The three Ada jobs produced successful records. The first 38B A16 job failed during weight loading: the default automatic placement filled one 14.66 GiB card and a further 250 MiB allocation failed. Stopped only that job and its dependent analysis waiter, preserving their logs. Added explicit automatic-placement memory ceilings of 12 GiB on the first visible card and 13 GiB on each other card (also capped at 90% of current free memory), leaving space for vision, KV cache and allocator overhead. Prompts, model weights, frame sets, seeds and decision rules are unchanged. The replacement run will retain the existing arm specifications.

### SDPA correction before the replacement runs

The memory-capped 38B checkpoint loaded, but inference still exhausted an A16 while forming the eager-attention softmax. The pinned InternVL3.5 remote code explicitly sets the language model to `eager` when `use_flash_attn=False`; the earlier setup description incorrectly called this SDPA. Corrected the loader to call the native language model's `set_attn_implementation('sdpa')` and verify the resulting backend for non-FlashAttention arms. The vision encoder retains its own supported implementation. InternVL2 arms still use their original FlashAttention configuration.

Stopped and archived the initial InternVL3.5-8B detector records and 38B attempt under `package/diagnostics/pre_sdpa/` before restarting those arms. They are excluded from the analysis. The two other Ada jobs (Qwen detectors and InternVL2 recognition) continue unchanged. InternVL3.5 format probes will be repeated with SDPA before its replacement detector runs. The 38B job now stops if its probe fails rather than moving on to the ablation. No gate threshold, seed, prompt, model weight, or frame selection changed.

### Predeclared prompt switch after the SDPA probes

The repeated SDPA probes returned one parseable firearm box for `internvl35_8b_off` but zero for `internvl35_8b_on` with the detection prompt. Following the previously declared probe rule, set the thinking arm and its `_r2` repeat to the model-card referring prompt. The non-thinking arm retains the detection prompt. The runner records the actual selected prompt in its immutable spec and sends that same prompt to the model. This substitution was made before either replacement 8B detector arm collected Phase 1 records.

The referring probe answers sometimes omit the required thinking trace; the existing thinking parser marks those calls truncated with no final answer, even when the raw text contains a box and is shorter than the token cap. This protocol limitation is retained and must be distinguished from token-limit exhaustion in the report. No parser/scoring rule was broadened to recover those boxes.

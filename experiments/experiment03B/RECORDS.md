# experiment03B raw-record contract (v1.1)

This is the interface between the runner (`code/run_workflow.py`), the analyzer (`code/analyze3b.py`) and the
independent validator (`code/validate3b.py`). It is written so that the validator can be implemented from this file
and `PROTOCOL.md` alone, without reading the runner or the analyzer.

**Conventions:**
- All JSON is UTF-8. JSONL files hold one object per line and are split on `"\n"` only.
- Slots are integers 0..7. A mask is a list of distinct slots sorted ascending, and `[]` means the unmasked input.
- Families are `none` (only with `mask = []`), `blur` and `grey`.

## Run folder `runs/<run_id>/`

```text
frozen/config.json          resolved configuration (never edited after the run starts)
frozen/code_sha256.json     {"<path relative to repo root>": "<sha256>"} for every file that computes results
frozen/manifest.json        the locked window manifest used by this run (see below)
inputs.json                 the unique inputs of this run, with their identities (see below)
ledger.jsonl                stage ledger: {"stage", "event": "start"|"done"|"failed"|"skipped", "at", "detail"}
requests/<stage>[.shardKofN].jsonl   append-only request logs (see below)
cache/forward_cache[.shardKofN].jsonl   model outputs keyed by result_key (append-only; never edited)
derived/rankings.jsonl      one record per label-available input, merged by window_uid across invocations
results/                    analyzer outputs (see below); results/prompts/ holds the exact question strings, the
                            no-sound diff and a comparison with experiment02; results/example_inputs/ (local only)
                            holds replacement examples
validation/validation.json  independent validator output
gpu_time/<invocation>.json  GPU seconds per invocation, rewritten after every model call (survives hard kills)
retrieval.jsonl             retrieval ledger (fetch, extracted, purge, failed, fidelity_check events)
fidelity_check.json         sparse-extraction fidelity result
preflight.json              read-only preflight checks
integration_fresh_recompute.json   integration only: cached-versus-fresh comparison
```

JSONL files may end with a torn fragment after a hard kill. Readers ignore a final segment without a newline, and
the next append moves it to `<file>.torn.<n>`.

## `inputs.json`

`{"inputs": [ INPUT, ... ]}`. Each INPUT has these fields:

| field | type | meaning |
|---|---|---|
| `window_uid` | str | `"<video_id>@<anchor>"`, unique |
| `video_id` | str | e.g. `RoadAccidents127_x264` |
| `category` | str | the anomaly category, or `Normal` |
| `anchor` | int | VERA anchor |
| `frame_indices` | [8 int] | the source frame IDs, in slot order |
| `frame_sha256` | [8 str] | sha256 of each JPEG used |
| `video_sha256` | str | the source video's sha256 |
| `strata` | [str] | a subset of `["event", "negative", "normal"]` (deduplicated) |
| `window_record_ids` | [str] | links to the manifest window records sharing this input |
| `development` | bool | true for `RoadAccidents127_x264` and `Normal_Videos_018_x264` |
| `metadata_flags` | [str] | e.g. `["fps_not_30"]`, `["decoded_count_mismatch"]`; `[]` if clean |
| `annotated_slots` | [int] | slots whose source frame lies inside an annotated interval |
| `clipped` | bool | the window was clipped at the start or end |
| `status` | str | `ok`, `fetch_failed` (cause in `retrieval.jsonl`) or `invalid_index` |

## Request logs `requests/*.jsonl`

Each line is one attempt at one logical request:

| field | type | meaning |
|---|---|---|
| `request_id` | str | unique per logical request (the same id is repeated across attempts) |
| `window_uid` | str | the input |
| `phase` | str | `baseline`, `fixed`, `adaptive` or `evaluation` |
| `purpose` | str | free text (`single`, `pair`, `fixed_prefix`, `adaptive_candidate`, ...) |
| `target` | str | `original_generation`, `nosound_generation` or `label_only_score` |
| `family` | str | `none`, `blur` or `grey` |
| `mask` | [int] | sorted slots |
| `repeat` | int | 0; or 1 and 2 for parse-failure re-generations of the identical input |
| `cache_status` | str | `fresh`, `cached`, `prior_cache`, `error` or `skipped_budget` |
| `result_key` | str | the cache key (equal requests share it) |
| `sequence_key` | str | the identity of the 8-slot pixel input, independent of the prompt |
| `attempt` | int | 1, 2, ... for this `request_id` |
| `slot_tensor_sha256` | [8 str] | hash of each slot's normalised tensor (absent on error) |
| `slot_identity_ok` | bool | exactly the masked slots differ from the unmasked input |

**Score fields** (`label_only_score`): `S` (log-softmax margin as stored), `S_hi` (FP32 head readout), `logit_0`,
`logit_1`, `hi_logit_0`, `hi_logit_1`, `S_resolution`, `S_hi_rounds_to_model_logits`, `candidate_mass` and
`finite`.

**Native margin for analysis (v1.1).** It is `logit_1 − logit_0`, the exact BF16 logit difference. The stored `S`
differs from it by up to about 1e-7 of log-softmax rounding. This margin is used for the native ranking, the
scorer's class (> 0 → 1, < 0 → 0, = 0 undecided) and every native difference.

**Generation fields** (`original_generation`, `nosound_generation`): `raw_response`, `label` (0, 1 or null) and
`parse_status` (`ok` or a reason).

**Error records** carry `error` (a string) and no result fields.

**Authoritative value rule.** For each tuple `(window_uid, target, family, mask, repeat)`:
- **Value.** It is the value of the **latest** line whose `cache_status` is `fresh`, `cached` or `prior_cache`.
- **Consistency.** All successful lines that share a `result_key` must carry identical result fields.
- **Missing.** A tuple with no successful line is missing.
- **Labels.** For generation, the label used is that of `repeat 0` if parsed. Otherwise it is the first parsed label
  among repeats 1 and 2, otherwise missing.

## `derived/rankings.jsonl`

There is one record for each input whose baseline original label `L0` is parsed:

```json
{"window_uid": "...", "L0": 1, "q": 1,
 "baseline": {"S_hi": 2.85, "S": 2.75},
 "fixed": {
   "blur":        RANKING,     // from S_hi of blur singles
   "grey":        RANKING,     // from S_hi of grey singles
   "blur_native": RANKING      // from native S of blur singles
 },
 "adaptive": {"family": "blur", "readout": "S_hi", "max_budget": 4,
   "states": [ {"step": 1, "masked": [], "state_score": 2.85,
                "candidates": [{"slot": 0, "mask": [0], "score": 2.98, "effect": -0.13}, ...],
                "selected": 3, "tied_max": []}, ... ],
   "order": [3, 4, 2, 5]},
 "prefixes": {"fixed": {"1": [..], "2": [..], "3": [..], "4": [..]},
              "adaptive": {"1": [..], "2": [..], "3": [..], "4": [..]}},
 "T_w": [3, 4]}
```

`blur_native` uses the native margin `logit_1 − logit_0`. The record also carries `ranking_mode` (the
`--ranking-mode` of the invocation that wrote it).

**RANKING** = `{"effects": {"0": D, ..., "7": D}, "order": [8 slots], "ties": [[slots sharing one exact value], ...],
"tie_spans_cut": {"1": bool, "2": bool, "3": bool, "4": bool}, "top2_margin": float, "all_tied": bool,
"all_nonpositive": bool}`.

- **Effects.** `effects[i] = q × (baseline − single_i)` on the stated readout and family, unrounded.
- **Order.** By effect descending; exact ties go to the lower slot.
- **Tie at a cut.** `tie_spans_cut[k]` is true iff `effects[order[k-1]] == effects[order[k]]`.
- **Margin.** `top2_margin = effects[order[1]] − effects[order[2]]`.
- **Incomplete rankings.** If any single is missing, the ranking is `{"status": "incomplete", "missing": [...]}`.

**Adaptive states.**
- **Score.** Each state's `state_score` is `S_hi(x_masked)`, where `x_[] = x`.
- **Candidates.** Every remaining slot is a candidate, with `mask = sorted(masked + [slot])` and
  `effect = q × (state_score − score)`.
- **Selection.** `selected` is the maximum effect, with the lower slot on exact ties; `tied_max` lists the other
  slots tied at the maximum.
- **States.** There are 4 states (steps 1..4), which produce `order`.
- **Call accounting.** Each state also carries `state_sequence_key` and `model_calls` = `{"new_adaptive",
  "shared_with_fixed", "cached", "missing"}`. Each candidate carries `sequence_key` and `cache_status`. These are
  taken from the request log, so they are the same whichever stage computed the request.

**Prefixes.** `prefixes.fixed[k] = sorted(order_blur_fixed[:k])` and `prefixes.adaptive[k] = sorted(order[:k])`.
They are evaluated under **grey**. `T_w = prefixes.fixed["2"]`.

## Analyzer outputs `results/`

| file | content |
|---|---|
| `inputs_metrics.csv` | one row per input: identities, strata, flags, L0, nosound label, q, ranking summaries, `d_w` (blank if missing), `F_top2`, `F_random_mean`, `n_pair_labels`, `e_w`, overlaps, τ_b values, trajectory flips |
| `pairs.csv` | one row per (input, family, pair): `S_hi`, `S`, `D`, `J`, and on grey the original-generation label and flip |
| `trajectories.csv` | one row per (input, mode ∈ {fixed, adaptive}, k ∈ 1..4): mask, label, flip, `q·ΔS_hi`, `q·ΔS`, the number of new model calls attributable to the mode |
| `primary.json` | the primary estimate and sensitivities (see below) |
| `secondary.json` | the §7.2 results |
| `table_a1.json` | Table A1 rows and note |

`primary.json` has these fields:

```json
{"estimate_pp": float|null, "ci95_pp": [lo, hi]|null, "bootstrap": {"seed": 0, "B": 10000,
 "stability": [{"seed": 1, "ci95_pp": [...]}, ...]}, "n_videos": int, "n_inputs": int, "n_inputs_missing_dw": int,
 "video_means_pp": {"<video_id>": float}, "d_w": {"<window_uid>": float|null},
 "bounds_pp": [lo, hi], "sensitivity": {"tie_free": {...}, "scorer_agreement": {...},
 "with_development": {...}, "metadata_clean": {...}}}
```

The `pp` values are `100 ×` the mean difference.

**Counts and flags.**
- `n_inputs` is the population size (confirmation alert inputs), and `n_inputs_missing_dw` is how many of them
  lack `d_w`.
- `n_videos` counts videos with at least one `d_w` (the bootstrap's V), and `n_videos_population` counts all videos
  in the population.
- `run_flags` = `{partial, skipped_budget_lines, error_lines, record_conflicts}`.
- `code_sha256` gives the analyzer files' hashes.
- `bounds_note` explains that the bounds refer to the full-data estimate.

## Frozen manifest `frozen/manifest.json`

**Cases populations** (integration and synthetic runs) instead write `{"version": "cases", "cases": [...],
"videos": [{"video_id", "length", "development"}], "unique_inputs": [{"window_uid", "window_record_ids",
"strata"}]}`.

**Manifest populations** copy the locked manifest:

```json
{"version": "...", "inputs_sha256": {...}, "counts": {...},
 "videos": [{"video_id", "category", "length", "zip", "zip_member", "video_bytes", "development", "status", ...}],
 "windows": [{"window_record_id", "stratum", "video_id", "anchor", "window": [s, e], "frame_indices": [...],
              "clipped": bool, "interval": [s, e]|null, "interval_index": int|null, "annotated_slots": [...],
              "gap_to_annotation": int|null, "window_uid", "status"}],
 "unique_inputs": [{"window_uid", "window_record_ids", "strata"}]}
```

# Deviations from the pre-registration

Each entry: what changed, why, the evidence, and when (before or after which result).

## D1: category probabilities are read with FlashAttention; attention with eager (2026-10-26, after Phase 0, before Phase 1)

**Pre-registered:**
- One eager-attention scoring pass gives both the decision attention (M1) and the category
  probabilities (M2).
- Phase 0 checks (b2) every greedy token reproduced by the eager replay.
- Phase 0 checks (b3) eager and FlashAttention category log-probabilities within 0.05 nats.

**Phase 0** (`grounding1/phase0/phase0.json`, 4 validation videos):

| Check | Result |
|---|---|
| b1 generate == batch_chat | 4/4 |
| b2 eager replay reproduces the greedy tokens | 207/209, failed |
| b3 eager vs FlashAttention log-probabilities | max difference 0.120 nats, failed |
| c grid orientation | 9/10 within one cell |

**Diagnostic** (`diagnostics/phase0_diag.py`, `grounding1/phase0/phase0_diagnostic.json`):
- The two b2 mismatches ("in" and ",", video Handgun__PAH1_C1_P1_V1_HB_3, positions 14 and 29)
  appear identically in a FlashAttention replay of the same sequence.
- Their margins are −0.19 and −0.42 there, too: near-ties that flip between incremental greedy
  decoding and a one-pass replay, not an effect of eager attention.
- The 0.12-nat b3 gap is on Machine_Gun at p ≈ 0.002. For the chosen class the gap is at most
  0.0007 nats.

**Change:**
- M1 attention still comes from the eager pass; attention weights are not available otherwise.
- Every category probability (the decision, p_true, and the occlusion readouts) is read from a
  FlashAttention replay: the same kernels that generate the answers.
- The eager p_true is kept in each record (`eager_p_true`) for comparison.

**Effect on the plan:** none on the arms, the outcomes or the gate. Phase 1 had not run.

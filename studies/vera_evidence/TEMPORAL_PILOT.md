# Temporal diagnostic v1: frozen before inference

Question: is original VERA deterministic on repeated identical input, and how often does its decision change under fixed frame-order interventions? This is an engineering/exploratory diagnostic, not a novel benchmark, LRPO reproduction, learning experiment, or proof of temporal faithfulness.

Prior art: VBenchComp (https://arxiv.org/abs/2505.14321) explicitly separates shuffled-frame-answerable questions; TVBench (https://daniel-cores.github.io/tvbench/) uses shuffled/single-frame controls. LRPO code was not located by exact-title/author searches or publisher/PDF links; OpenReview returned a verification challenge. This is a bounded access finding, not proof code is unavailable.

Selection: 4 event and 4 negative windows from previously consulted UCF test metadata; SHA256 ordering with fixed prefix, unique video-byte hashes, exclude development videos, no response-based selection. Manifest committed before inference. Event strata are not precise labels for every sampled slot. No detection accuracy or novel-method improvement will be inferred from 8 windows.

Five conditions per window, 40 generation calls: original chronological order, identical repeat, reversal, fixed permutation [3,0,6,1,7,4,2,5], repeat slot3 eight times (separate information-removal control). Original pinned InternVL2-8B, BF16, eight frames, original VERA explanation-first prompt, greedy decoding max1024 tokens. Positional Frame1..8 labels stay fixed. Verify processed tensor hashes exactly follow each mapping. Save every raw response and parse status. Do not silently drop failed parses or truncation. Same model instance; repeated calls are not independent samples.

Analysis: exact text/label repeat agreement, paired decision changes relative to ordered condition, parse failures, output lengths/truncation indicators, runtime/memory. No statistical significance claims at this size. Report source-level outcomes. Do not convert greedy binary labels into LRPO's stochastic probability reward or treat shuffle sensitivity as correctness. Stop if repeat consistency/input mapping fails. No optimizer, no prompt selection, no extension without reviewing results.

Compute: one empty RTX6000 Ada (currently index5, UUID GPU-b6901777-c94c-e9ac-b295-041178270888), recheck immediately before launch, 1-hour process timeout for this 40-call pilot. No other GPUs used. Read existing model/frames/source, write only separate research output directory. Do not touch existing jobs or source records.

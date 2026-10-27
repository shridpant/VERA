# Reproduction

Protocol/runner/analysis/reference sets were frozen at 7778d09 before inference. Read PROTOCOL.md and NOVELTY.md first. No new human labels, weight training or question optimization.

The referenced external videos and original UCA JSONs are not redistributed. Original source/model revisions are recorded in the prior run provenance and new run provenance. Reconstruct frame inputs using proxy_v1/select_pairs.py and prepare_frames.py; require the input digest in freeze.json. Paths in the inputs are recorded Oz paths and need relocation if reproducing elsewhere. Never silently change the model or input preprocessing.

On Oz, run check_contract.py with the original environment first. The runner is intentionally pinned to the reserved Ada5 UUID and refuses an occupied GPU. To reproduce on a different host, make a documented amendment rather than bypassing this check. Invoke run.py with --base pointing to vera_research, --inputs pointing to frames_v1/inputs.json, and a NEW --out directory. Use timeout1800 and CUDA_VISIBLE_DEVICES from the frozen protocol. Never overwrite run_001.

Local analysis:

    python3 studies/insufficient_evidence/selective_v1/analyze.py references --uca ../work/uca-official --manifest studies/insufficient_evidence/proxy_v1/manifest.json --out /tmp/vera_references.json
    python3 studies/insufficient_evidence/selective_v1/analyze.py analyze --run studies/insufficient_evidence/selective_v1/evidence/run_001 --references studies/insufficient_evidence/selective_v1/references.json --prior studies/insufficient_evidence/proxy_v1/evidence/run_001/results.jsonl

Compare the rebuilt references to the frozen JSON before analysis. plot.py regenerates report tables/figure from analysis.json. None of these metrics validates semantic support or establishes novelty.

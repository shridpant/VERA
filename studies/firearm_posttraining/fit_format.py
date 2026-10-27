"""FIT-only sampled output-schema calibration; no STOP or weight updates."""
import argparse
import json
import time
from pathlib import Path
from .data import balanced_order, load_split, write_json
from .model import NativeModel
from .rewards import parse
from .runtime import provenance, append, resource_metrics


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    provenance(out, vars(args))
    started = time.monotonic()
    book = json.loads(Path(args.codebook).read_text())
    fit = load_split(args.data, 'fit')
    by_id = {r['id']: r for r in fit}
    ids = balanced_order(fit, args.seed, 0)[:24]
    native = NativeModel(args.model, book, args.adapter, args.seed)
    before = native.adapter_hash()
    invalid = count = 0
    for video in ids:
        prompt, _ = native.prompt(by_id[video], args.grounded)
        for j in range(4):
            answer = native.generate(prompt, sample=True)
            valid = parse(answer['text'], book, args.grounded, answer['truncated']) is not None
            append(out / 'responses.jsonl', {'video': video, 'sample': j, 'valid': valid, **answer})
            invalid += not valid
            count += 1
            # More than four invalid answers can never satisfy 5% of 96.
            if invalid > 96 * .05:
                break
        if invalid > 96 * .05:
            break
    if native.adapter_hash() != before:
        raise RuntimeError('format calibration mutated weights')
    passed = count == 96 and invalid / count <= .05
    result = {'status': 'passed' if passed else 'failed_gate', 'responses': count, 'invalid': invalid,
              'invalid_fraction_observed': invalid / count, 'planned_responses': 96,
              'stopped_early_for_irreversible_gate_failure': count < 96,
              'weights_unchanged': True, 'grounded': args.grounded, 'adapter': args.adapter,
              'elapsed_seconds': time.monotonic() - started, **resource_metrics()}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'adapter', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--grounded', action='store_true')
    run(p.parse_args())

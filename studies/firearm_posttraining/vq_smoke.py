"""Real FIT-only two-video optimizer compatibility, without adaptation or STOP access."""
import argparse
import json
import time
from pathlib import Path
from .data import load_split, balanced_order, write_json
from .model import NativeModel, DEFAULT_QUESTIONS
from .rewards import parse
from .runtime import provenance, append, resource_metrics
from .verbalized import optimizer_instruction, decode_proposal


def run(args):
    started = time.monotonic()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    provenance(out, vars(args))
    book = json.loads(Path(args.codebook).read_text())
    fit = load_split(args.data, 'fit')
    by_id = {r['id']: r for r in fit}
    rows = [by_id[i] for i in balanced_order(fit, args.seed, 0)[:2]]
    native = NativeModel(args.model, book, seed=args.seed)
    initial_hash = native.adapter_hash()
    paired = []
    for index, row in enumerate(rows):
        prompt, _ = native.prompt(row)
        answer = native.generate(prompt)
        append(out / 'fit-predictions.jsonl', {'video': row['id'], **answer})
        paired.append({'video_in_this_prompt': index, 'predicted': parse(answer['text'], book, False, answer['truncated']),
                       'target': {k: row[k] for k in ('category', 'action_group', 'action_code')}})
    text = optimizer_instruction(book, list(DEFAULT_QUESTIONS), paired)
    prompt, actual_text = native.encode(rows, text)
    proposal = native.generate(prompt, sample=True)
    questions = decode_proposal(proposal['text'], list(DEFAULT_QUESTIONS), proposal['truncated'])
    if native.adapter_hash() != initial_hash:
        raise RuntimeError('VQ engineering changed weights')
    write_json(out / 'proposal.json', {'fit_ids': [r['id'] for r in rows], 'prompt': actual_text,
               'response': proposal, 'decoded_questions': questions})
    result = {'status': 'engineering_passed', 'scientific_role': 'FIT_only_engineering',
              'proposal_valid': questions is not None, 'proposal_truncated': proposal['truncated'],
              'prompt_tokens': int(prompt['input_ids'].shape[1]), 'images': len(prompt['image_grid_thw']),
              'weights_unchanged': True, 'elapsed_seconds': time.monotonic() - started, **resource_metrics()}
    write_json(out / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    p.add_argument('--seed', type=int, default=20261026)
    run(p.parse_args())

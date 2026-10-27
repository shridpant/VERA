"""Text-only adaptation using FIT examples; STOP targets never enter optimizer input."""
import json
import signal
import time
from pathlib import Path
from .data import balanced_order, load_split, write_json, canonical_hash
from .evaluation import evaluate
from .model import NativeModel, DEFAULT_QUESTIONS, instruction
from .protocol import validate
from .rewards import parse
from .runtime import provenance, append, resource_metrics


def valid_revision(old, new):
    if not isinstance(new, list) or not 1 <= len(new) <= 5:
        return False
    if any(not isinstance(x, str) or not 1 <= len(x) <= 240 for x in new):
        return False
    if len(new) == len(old):
        return sum(a != b for a, b in zip(old, new)) == 1
    return len(new) == len(old) + 1 and new[:-1] == old


def run(args):
    validate(args.frozen_protocol, args.data, args.model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    provenance(out, vars(args))
    started = time.monotonic()
    deadline = started + args.max_seconds - 35
    stopping = []
    old_handler = signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    book = json.loads(Path(args.codebook).read_text())
    fit, stop = load_split(args.data, 'fit'), load_split(args.data, 'stop')
    by_id = {r['id']: r for r in fit}
    order = balanced_order(fit, args.seed, 0)
    native = NativeModel(args.model, book, seed=args.seed)
    for parameter in native.model.parameters():
        parameter.requires_grad_(False)
    base_adapter_hash = native.adapter_hash()
    questions = list(DEFAULT_QUESTIONS)
    best_questions = out / 'questions-initial.json'
    write_json(best_questions, questions, immutable=True)
    best_key = (-1., -1., -1.)
    rejected, checks, fit_queries, optimizer_queries = 0, 0, 0, 0
    reason = 'maximum_candidate_revisions'
    try:
        measured = evaluate(native, stop, out / 'stop-initial', questions=questions, deadline=deadline, stop_requested=lambda: bool(stopping))
        if measured is None:
            reason = 'time_cap_during_baseline'
        else:
            best_key = tuple(measured[k] for k in ('balanced_category_accuracy', 'action_group_accuracy', 'action_code_accuracy'))
        for candidate_index in range(10) if measured is not None else ():
            if time.monotonic() >= deadline or stopping:
                reason = 'time_cap' if not stopping else 'external_signal'
                break
            panel = [by_id[order[(candidate_index * 12 + i) % len(order)]] for i in range(12)]
            errors, correct = [], []
            for row in panel:
                prompt, _ = native.prompt(row, questions=questions)
                answer = native.generate(prompt)
                fit_queries += 1
                predicted = parse(answer['text'], book, False, answer['truncated'])
                entry = {'row': row, 'answer': answer, 'parsed': predicted}
                if predicted is None or predicted['category'] != row['category']:
                    errors.append(entry)
                else:
                    correct.append(entry)
                append(out / 'fit-predictions.jsonl', {'candidate': candidate_index, 'video': row['id'], **answer})
            chosen = (errors[:1] + correct[:1]) if errors and correct else (errors or correct)[:2]
            paired = []
            for example_index, example in enumerate(chosen):
                row = example['row']
                paired.append({'video_in_this_prompt': example_index, 'predicted': example['parsed'],
                               'target': {k: row[k] for k in ('category', 'action_group', 'action_code')}})
            optimizer_text = ('Improve a short list of visual guiding questions for this video classification task. '
                'The displayed videos are FIT examples only. Their prediction and target labels follow. '
                'Return only JSON with one key questions (a list of at most five concise questions). '
                'Change exactly one existing question, keeping all others verbatim, or append exactly one question. '
                'Retain useful questions. Do not include filenames or labels for a particular example in a question.\n'
                + '\nTask codebook: ' + '\n'.join(f'{code}: {v["category"]}, {v["group"]}, {v["meaning"]}'
                                                       for code, v in sorted(book['codes'].items()))
                + '\nCurrent questions: ' + json.dumps(questions)
                + '\nPaired examples: ' + json.dumps(paired))
            prompt, actual_text = native.encode([e['row'] for e in chosen], optimizer_text)
            proposal = native.generate(prompt, sample=True)
            optimizer_queries += len(chosen)
            append(out / 'optimizer.jsonl', {'candidate': candidate_index, 'fit_ids': [e['row']['id'] for e in chosen],
                    'prompt': actual_text, 'response': proposal['text'], 'truncated': proposal['truncated']})
            try:
                proposed = json.loads(proposal['text'])['questions'] if not proposal['truncated'] else None
            except (ValueError, TypeError, KeyError):
                proposed = None
            if not valid_revision(questions, proposed):
                rejected += 1
                append(out / 'selection.jsonl', {'candidate': candidate_index, 'accepted': False, 'reason': 'invalid_revision'})
            else:
                file = out / f'questions-{candidate_index:02d}.json'
                write_json(file, proposed, immutable=True)
                measured = evaluate(native, stop, out / f'stop-{candidate_index:02d}', questions=proposed,
                                    deadline=deadline, stop_requested=lambda: bool(stopping))
                if measured is None:
                    reason = 'time_cap_during_validation'
                    break
                checks += 1
                key = tuple(measured[k] for k in ('balanced_category_accuracy', 'action_group_accuracy', 'action_code_accuracy'))
                accepted = key > best_key
                primary_progress = key[0] >= best_key[0] + .005
                if accepted:
                    best_key, best_questions, questions = key, file, proposed
                rejected = 0 if primary_progress else rejected + 1
                append(out / 'selection.jsonl', {'candidate': candidate_index, 'accepted': accepted,
                       'primary_progress': primary_progress, 'questions_hash': canonical_hash(proposed), **measured})
            if candidate_index >= 1 and rejected >= 3:
                reason = 'three_nonimproving_candidates'
                break
        if native.adapter_hash() != base_adapter_hash:
            raise RuntimeError('VQ base adapter unexpectedly changed')
        result = {'status': 'stopped', 'stop_reason': reason, 'best_questions_file': str(best_questions),
                  'best_metric_key': best_key, 'STOP_checks': checks, 'FIT_prediction_queries': fit_queries,
                  'FIT_optimizer_video_queries': optimizer_queries, 'censored': 'time_cap' in reason,
                  'elapsed_seconds': time.monotonic() - started, **resource_metrics()}
        write_json(out / 'result.json', result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        signal.signal(signal.SIGTERM, old_handler)

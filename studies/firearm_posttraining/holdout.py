"""Explicitly frozen, deterministic final evaluation. No holdout access before guards."""
import json
from pathlib import Path
from .data import enrich, write_json, digest
from .evaluation import evaluate
from .model import NativeModel
from .protocol import verify_selection
from .runtime import provenance


def run(args):
    selection, protocol = verify_selection(args.frozen_manifest, args.data)
    # The first access to original reporting targets occurs only below this guard.
    source = Path(args.original_data)
    audit = json.loads((Path(args.data) / 'audit.json').read_text())
    expected = audit['source_hashes'][args.split + '.json']
    if digest(source / f'{args.split}.json') != expected:
        raise ValueError('original holdout identity file changed')
    rows = json.loads((source / f'{args.split}.json').read_text())
    grounded = args.method in ('SG', 'RG')
    if grounded:
        annotation_path = source / f'boxes_{args.split}.json'
        annotations = json.loads(annotation_path.read_text())
    else:
        annotations = {r['id']: {'statuses': ['unknown'] * r['frames_decoded'], 'boxes': {}} for r in rows}
    rich = [enrich(r, annotations[r['id']], args.frames_root) for r in rows]
    questions, adapter = None, None
    if args.method != 'B0':
        entry = next(e for e in selection['entries'] if e['method'] == args.method and e['seed'] == args.seed)
        if args.method == 'VQ':
            questions = json.loads(Path(entry['selected']).read_text())
        else:
            adapter = entry['selected']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    provenance(out, vars(args))
    write_json(out / 'holdout_manifest.json', {'split': args.split, 'selection_hash': selection['selection_hash'],
                'ids': [r['id'] for r in rich], 'source_sha256': expected,
                'annotation_sha256': digest(annotation_path) if grounded else None,
                'frames': {v['path']: v['sha256'] for r in rich for v in r['views']}})
    book = json.loads(Path(args.codebook).read_text())
    native = NativeModel(protocol['model'], book, adapter=adapter, seed=args.seed)
    measured = evaluate(native, rich, out / 'evaluation', grounded, questions)
    write_json(out / 'result.json', {'status': 'complete', 'method': args.method, 'seed': args.seed,
               'split': args.split, 'metrics': measured,
               'predictions_sha256': digest(out / 'evaluation/predictions.jsonl')})
    return measured

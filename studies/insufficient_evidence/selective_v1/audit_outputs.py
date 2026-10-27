"""Post-run content/format audit, separate from the frozen descriptive analysis."""
import argparse
import json
from pathlib import Path

PLACEHOLDERS = ('a concrete visible observation', 'a detail not resolved by these frames')


def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); a = p.parse_args()
    out = {}
    for run in ('run_001', 'run_002'):
        path = a.root/'evidence'/run
        rows = [json.loads(z) for z in (path/'results.jsonl').read_text().splitlines()]
        variants = {}
        for variant in ('original', 'manual', 'cited'):
            chosen = [r for r in rows if r['variant'] == variant]
            objs = [r['explanation'] for r in chosen if r['explanation'] is not None]
            obs = [x for o in objs for x in o['observations']]
            variants[variant] = {'calls': len(chosen), 'invalid_binary': sum(r['decision'] is None for r in chosen),
                'at_token_cap': sum((r['result']['n_new_tokens'] or 0) >= 1024 for r in chosen),
                'valid_schema': len(objs) if variant != 'original' else None,
                'empty_observation_lists': sum(not o['observations'] for o in objs) if variant != 'original' else None,
                'placeholder_observations': sum(x['claim'].strip().lower() in PLACEHOLDERS for x in obs),
                'placeholder_unresolved': sum(x.strip().lower() in PLACEHOLDERS for o in objs for x in o['unresolved']),
                'nonplaceholder_observation_count': sum(x['claim'].strip().lower() not in PLACEHOLDERS for x in obs),
                'responses_with_nonplaceholder_observations': sum(any(x['claim'].strip().lower() not in PLACEHOLDERS for x in o['observations']) for o in objs),
                'manual_nonempty_frame_arrays': sum(bool(x['frames']) for x in obs) if variant == 'manual' else None,
                'generation_seconds': sum(r['result']['runtime_s'] for r in chosen)}
        out[run] = {'calls': len(rows), 'complete': (path/'completion.json').exists(), 'conditions': variants,
                    'last_case': rows[-1]['case_id'], 'last_variant': rows[-1]['variant'],
                    'status': 'completed; template placeholders contaminate explanations' if run == 'run_001' else 'stopped after six calls: missing Output field; no retry'}
    out['accounting'] = {'previous_proxy_calls': 48, 'selective_calls': sum(v['calls'] for v in out.values()), 'call_cap': 192}
    out['accounting']['actual_total_calls'] = 48 + out['accounting']['selective_calls']
    out['warning'] = 'Post-run implementation audit only. Nonplaceholder or syntactically valid claims are not verified true.'
    (a.root/'evidence/output_audit.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

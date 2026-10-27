"""Frozen descriptive metrics. No semantic correctness inferred from citations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

STOP = set('the and that this there with from into then after before while were was are have has had for its his her their them they she him who which when where what not any all some one two three video frame frames scene visible person people man woman men women wearing also can could would indicates indicate anomaly abnormal normal appears appear seen see based observation observations'.split())


def words(text):
    return sorted(set(re.findall(r'[a-z]{3,}', text.lower())) - STOP)


def references(uca, manifest, out):
    u, hashes = {}, {}
    for split in ('Train', 'Val', 'Test'):
        path = uca/'UCF Annotation/json'/f'UCFCrime_{split}.json'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        u.update(json.loads(path.read_text()))
    assert hashes == manifest['source_hashes']
    refs = {}
    for v in manifest['videos']:
        d = u[v['video_id']]
        interval = v['views']['annotated']['container_seconds']
        sentences = [s for t, s in zip(d['timestamps'], d['sentences']) if t == interval]
        assert sentences
        refs[v['video_id']] = {'interval': interval, 'reference_sha256': hashlib.sha256(json.dumps(sentences, ensure_ascii=False).encode()).hexdigest(),
                               'word_set': words(' '.join(sentences)), 'caption_count': len(sentences)}
    out.write_text(json.dumps({'source_sha256': hashes, 'references': refs, 'definition': 'exact lowercase ASCII content-word set; no entailment'}, indent=2)+'\n')


def analyze(run, refs, prior):
    rows = [json.loads(x) for x in (run/'results.jsonl').read_text().splitlines()]
    assert len(rows) == 72
    bycase = defaultdict(dict)
    for r in rows:
        assert r['variant'] not in bycase[r['case_id']]
        bycase[r['case_id']][r['variant']] = r
    assert len(bycase) == 24 and all(len(x) == 3 for x in bycase.values())
    old = {(x['video_id'], x['view']): x for x in [json.loads(z) for z in prior.read_text().splitlines()] if x['variant'] == 'original'}
    summary = {}
    for variant in ('original', 'manual', 'cited'):
        selected = [r for r in rows if r['variant'] == variant]
        obs = [o for r in selected if r['explanation'] for o in r['explanation']['observations']]
        lexical = []
        for r in selected:
            if r['view'] != 'annotated':
                continue
            target = set(refs['references'][r['video_id']]['word_set'])
            text = r['result']['raw_response'].rsplit('Output:', 1)[0] if variant == 'original' else ' '.join(o['claim'] for o in (r['explanation'] or {}).get('observations', []))
            matched = sorted(target & set(words(text)))
            lexical.append({'case_id': r['case_id'], 'matched_words': matched, 'target_word_count': len(target), 'recall': len(matched)/len(target) if target else None})
        original_alerts = [r for r in selected if bycase[r['case_id']]['original']['decision'] == 1]
        summary[variant] = {
            'n': len(selected), 'alerts': sum(r['decision'] == 1 for r in selected),
            'alerts_by_view': {v: sum(r['decision'] == 1 for r in selected if r['view'] == v) for v in ('annotated', 'gap')},
            'binary_agreement_with_original': sum(r['decision'] == bycase[r['case_id']]['original']['decision'] for r in selected),
            'retained_original_alerts': sum(r['decision'] == 1 for r in original_alerts), 'original_alert_count': len(original_alerts),
            'invalid_binary': sum(r['decision'] is None for r in selected), 'at_token_cap': sum((r['result']['n_new_tokens'] or 0) >= 1024 for r in selected),
            'valid_explanation_schema': sum(r['explanation'] is not None for r in selected) if variant != 'original' else None,
            'nonempty_unresolved': sum(bool(r['explanation'] and r['explanation']['unresolved']) for r in selected) if variant != 'original' else None,
            'alert_and_unresolved': sum(r['decision'] == 1 and bool(r['explanation'] and r['explanation']['unresolved']) for r in selected) if variant != 'original' else None,
            'observation_count': len(obs) if variant != 'original' else None,
            'observations_with_legal_nonempty_citations': sum(bool(o['frames']) for o in obs) if variant != 'original' else None,
            'positive_reference_word_recall_mean': sum(x['recall'] for x in lexical if x['recall'] is not None)/sum(x['recall'] is not None for x in lexical),
            'lexical_details': lexical, 'generation_seconds': sum(r['result']['runtime_s'] for r in selected)}
    a = [r for r in rows if r['variant'] == 'original']
    return {'conditions': summary, 'all_cases': {k: {v: r['decision'] for v, r in values.items()} for k, values in bycase.items()},
            'original_repeat_label_matches': sum(r['decision'] == old[(r['video_id'], r['view'])]['decision'] for r in a),
            'original_repeat_text_matches': sum(r['result']['raw_response'] == old[(r['video_id'], r['view'])]['result']['raw_response'] for r in a),
            'completion': json.loads((run/'completion.json').read_text()),
            'limitations': ['No correctness, hallucination or novelty claim.', 'Existing captions are partial references, not exhaustive visual truth.', 'Content-word overlap is confounded by negation, wording, length and unequal explanation formats.', 'Legal citations are syntax, not evidence verification.', 'Consulted UCFtest sources; no training or untouched evaluation.']}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('references'); q.add_argument('--uca', type=Path, required=True); q.add_argument('--manifest', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q = sub.add_parser('analyze'); q.add_argument('--run', type=Path, required=True); q.add_argument('--references', type=Path, required=True); q.add_argument('--prior', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'references':
        references(a.uca, json.loads(a.manifest.read_text()), a.out)
    else:
        result = analyze(a.run, json.loads(a.references.read_text()), a.prior)
        (a.run/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'all_cases'}, indent=2))


if __name__ == '__main__':
    main()

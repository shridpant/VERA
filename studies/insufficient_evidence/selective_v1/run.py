"""Three frozen inference instructions; one shared model; no learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

FLAG = "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
OLD_FORMAT = 'Answers to Prompt Questions: [Provide your analysis by answering the questions listed in Prompt Questions.]\nOutput:\n[ONLY the integer class label; make necessary assumptions if needed]'
RULE = ('Keep the anomaly decision binary: 0 normal or 1 anomaly. Unresolved explanation details do not constitute a third decision label. '
        'Use only the supplied visual evidence for explanation details; do not invent sounds, intent, causes or unseen events. '
        'List up to three concrete visible observations, and list any unresolved event details separately. '
        'Make your best binary anomaly decision independently of whether all details can be resolved. ')
FORMAT = ('Answers to Prompt Questions: [Answer the original five questions.]\n'
          'Explanation: {"observations": [{"claim": "a concrete visible observation", "frames": []}], "unresolved": ["a detail not resolved by these frames"]}\n'
          'Output:\n[ONLY one integer: 0 or 1]')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_label(text):
    fields = re.split(r'(?i)\bOutput\s*:', text)
    if len(fields) != 2:
        return None
    tail = fields[1].strip().strip('`').strip()
    return int(tail) if tail in ('0', '1') else None


def explanation(text):
    """Validate schema without silently repairing the model's answer."""
    try:
        payload = text.split('Explanation:', 1)[1].rsplit('Output:', 1)[0].strip().strip('`').strip()
        obj = json.loads(payload)
        assert set(obj) == {'observations', 'unresolved'}
        assert isinstance(obj['observations'], list) and len(obj['observations']) <= 3
        assert isinstance(obj['unresolved'], list)
        assert all(isinstance(x, str) and x.strip() for x in obj['unresolved'])
        for obs in obj['observations']:
            assert set(obs) == {'claim', 'frames'}
            assert isinstance(obs['claim'], str) and obs['claim'].strip()
            assert isinstance(obs['frames'], list)
            assert all(type(i) is int and 1 <= i <= 8 for i in obs['frames'])
            assert len(set(obs['frames'])) == len(obs['frames'])
        return obj
    except (ValueError, IndexError, TypeError, AssertionError, AttributeError):
        return None


def instruction(original, qs, cited):
    assert original.count(FLAG) == 1 and original.count(OLD_FORMAT) == 1
    citation = ('Each observation must cite one or more supporting supplied frame slots as integers 1 through 8 in frames. '
                'If you cannot identify supporting slots, omit that observation and put the detail in unresolved. ' if cited else
                'Leave frames as an empty array for each observation; frame citations are not requested. ')
    return original.replace(FLAG, qs + RULE + citation).replace(OLD_FORMAT, FORMAT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    gpu = os.environ['CUDA_VISIBLE_DEVICES']
    assert gpu == 'GPU-b6901777-c94c-e9ac-b295-041178270888'
    status = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert int(status.strip()) < 100, 'GPU is occupied; do not load model'
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    os.environ['VERA_PROJECT_ROOT'] = str(a.base)
    sys.path.insert(0, str(a.base / 'experiments/experiment03B/code'))
    import config3b
    sys.path.insert(0, str(a.base / 'experiments/experiment02/code'))
    import engine
    cfgpath = a.base / 'experiments/experiment03B/runs/full_v1/frozen/config.json'
    cfg = config3b.exp02_compat(json.loads(cfgpath.read_text()))
    cases = json.loads(a.inputs.read_text())['cases']
    assert len(cases) == 24 and len({x['case_id'] for x in cases}) == 24
    deps = [cfgpath, a.base/'experiments/experiment02/code/engine.py', a.base/'experiments/experiment02/code/attr_lib.py', a.base/'setup/source/VERA/generate_initial_score.py', a.base/'setup/source/VERA/VERA_learner_instruct.txt']
    provenance = {'start_unix': start, 'runner_sha256': sha(__file__), 'inputs_sha256': sha(a.inputs), 'gpu_uuid': gpu,
                  'dependency_sha256': {str(x): sha(x) for x in deps}, 'model': cfg['model'], 'generation': cfg['generation'],
                  'call_limit': 72, 'seconds_limit': 1800, 'learning': 'none; no weights, questions or optimizer updated'}
    (a.out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    em = a.out/'eval_manifest.json'
    seen = {x['manifest_entry']: x['manifest_length'] for x in cases}
    em.write_text(json.dumps([{'video': k, 'length': v} for k, v in seen.items()]))
    eng = engine.Engine(cfg, em)
    original, qs = eng.prompt_parts['original']
    for variant in ('manual', 'cited'):
        ins = instruction(original, qs, variant == 'cited')
        eng.prompt_parts[variant] = (ins, '')
        eng.wrappers[variant] = eng.gis.VideoAnomalyDetectionModelInference(model=eng.model, tokenizer=eng.tok, optimizer_instruct=None, model_instruct=ins, new_qs='', generation_config=dict(cfg['generation'])).eval().cuda()
        assert eng.wrappers[variant].model is eng.model
    assert not any(p.requires_grad for p in eng.model.parameters())
    variants = ['original', 'manual', 'cited']
    prompts = {v: eng.effective_question(v) for v in variants}
    for v, text in prompts.items():
        (a.out/f'prompt_{v}.txt').write_text(text)
    records, hashes = [], {}
    videos = list(dict.fromkeys(x['video_id'] for x in cases))
    with (a.out/'results.jsonl').open('x') as f:
        for vi, video in enumerate(videos):
            vc = [x for x in cases if x['video_id'] == video]
            if vi % 2:
                vc.reverse()
            order = variants[vi % 3:] + variants[:vi % 3]
            for case in vc:
                for variant in order:
                    assert len(records) < 72 and time.time()-start < 1700, 'Budget stop'
                    result = eng.generate(variant, case, (), {'family': 'none'})
                    assert result['question_sha256'] == hashlib.sha256(prompts[variant].encode()).hexdigest()
                    if case['case_id'] in hashes:
                        assert hashes[case['case_id']] == result['slot_tensor_sha256']
                    hashes[case['case_id']] = result['slot_tensor_sha256']
                    label = parse_label(result['raw_response'])
                    obj = explanation(result['raw_response']) if variant != 'original' else None
                    rec = {'case_id': case['case_id'], 'video_id': video, 'view': case['condition'], 'variant': variant, 'decision': label, 'explanation': obj, 'result': result}
                    records.append(rec)
                    f.write(json.dumps(rec)+'\n'); f.flush(); os.fsync(f.fileno())
                    print(json.dumps({'case': case['case_id'], 'variant': variant, 'label': label, 'schema_valid': obj is not None if variant != 'original' else None}), flush=True)
                    if label is None:
                        raise RuntimeError('Invalid binary output; partial run preserved')
    summary = {'calls': len(records), 'wall_seconds': time.time()-start, 'generation_seconds': sum(x['result']['runtime_s'] for x in records),
               'engine': eng.summary(), 'paired_tensor_checks': len(hashes), 'dependencies_unchanged': all(sha(p) == h for p, h in provenance['dependency_sha256'].items())}
    assert summary['calls'] == 72 and summary['dependencies_unchanged']
    (a.out/'completion.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()

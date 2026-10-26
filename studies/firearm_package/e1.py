"""E1: the detector ablation (package plan, Module 1). One resumable run folder per arm.

runs/e1/<arm>/spec.json (immutable; checked on resume) and records.jsonl (one line per frame; a failed call is
recorded and the run moves on). Seeds are per frame, so a resumed or repeated arm makes the same calls.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from vera_runner.detector import INTERNVL_PROMPT, INTERNVL_REC_PROMPT, QWEN_PROMPT, parse, score_frame
from vera_runner.ledger import JsonlAppender, now, read_jsonl
from .manifest import select

PROMPTS = {'internvl': INTERNVL_PROMPT, 'qwen3vl': QWEN_PROMPT, 'qwen35': QWEN_PROMPT}


def call_seed(seed, arm, frame_id):
    return int(hashlib.sha256(f'{seed}|{arm.get("seed_offset", 0)}|{frame_id}'.encode()).hexdigest()[:8], 16)


def code_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return None


def load_model(cfg, arm):
    from vera_runner import vlm
    path = cfg['models'][arm['model']]
    if arm['family'] == 'internvl':
        return vlm.InternVL(path, thinking=arm['thinking'], flash=arm.get('flash', False), device_map=arm.get('device_map'))
    return vlm.Qwen(path, thinking=arm['thinking'], family=arm['family'])


def frame_image(cfg, f):
    from PIL import Image
    with Image.open(Path(cfg['frames_dir']) / f['video'] / f"{f['frame_index']:06d}.jpg") as im:
        return im.convert('RGB')


def ground(model, family, image, prompt, arm, seed):
    if family == 'internvl':
        from vera_runner.vlm import dynamic_preprocess
        tiles = dynamic_preprocess(image, max_num=arm.get('max_num', 6))
        return {**model.chat([tiles], '<image>\n' + prompt, arm['max_new_tokens'], seed=seed), 'n_tiles': len(tiles)}
    return model.chat([image], prompt, arm['max_new_tokens'], seed=seed)


def spec_for(cfg, name, arm, manifest_path, prompt):
    spec = {'experiment': 'package_e1', 'arm': name, **arm, 'model_path': cfg['models'][arm['model']], 'prompt': prompt,
            'seed': cfg['seed'], 'manifest_sha256': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}
    spec['spec_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def run_arm(cfg, name, arm, manifest_path, out_dir, model=None, limit=None, prompt=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompt = prompt or PROMPTS[arm['family']]
    spec = spec_for(cfg, name, arm, manifest_path, prompt)
    spec_file = out / 'spec.json'
    if spec_file.exists():
        old = json.loads(spec_file.read_text())
        if old['spec_sha256'] != spec['spec_sha256']:
            raise ValueError(f'{name}: spec changed since the run started; use a new arm name')
    else:
        spec_file.write_text(json.dumps({**spec, 'created_at': now(), 'code_commit': code_commit()}, indent=1) + '\n')
    frames = select(json.loads(Path(manifest_path).read_text()), arm['frames'])
    done = {r['frame_id'] for r in read_jsonl(out / 'records.jsonl')[0] if r['status'] != 'failed'}
    todo = [f for f in frames if f['frame_id'] not in done][:limit]
    model = model or load_model(cfg, arm)
    log = JsonlAppender(out / 'records.jsonl')
    failed = 0
    try:
        for f in todo:
            seed = call_seed(cfg['seed'], arm, f['frame_id'])
            base = {'frame_id': f['frame_id'], 'video': f['video'], 'category': f['category'], 'seed': seed, 'started_at': now()}
            try:
                result = ground(model, arm['family'], frame_image(cfg, f), prompt, arm, seed)
                parsed = parse('internvl' if arm['family'] == 'internvl' else 'qwen', result)
                log.write({**base, **parsed, 'score': score_frame(parsed['found'], f['gt']), **result, 'finished_at': now()})
            except Exception:
                failed += 1
                log.write({**base, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                if failed >= 20:
                    raise
    finally:
        log.close()
    return {'arm': name, 'frames': len(frames), 'done_before': len(done), 'ran': len(todo), 'failed': failed}


def probe(cfg, name, arm, manifest_path, out_path, n=6, model=None, prompts=None):
    """Format probe: a few fixed manifest frames (firearm and No_Gun), every candidate prompt, raw text kept."""
    frames = json.loads(Path(manifest_path).read_text())
    pick = [f for f in frames if f['category'] == 'Handgun'][:2] + [f for f in frames if f['category'] == 'Machine_Gun'][:2] + \
           [f for f in frames if f['category'] == 'No_Gun'][:n - 4]
    model = model or load_model(cfg, arm)
    prompts = prompts or [PROMPTS[arm['family']]] + ([INTERNVL_REC_PROMPT] if arm['family'] == 'internvl' else [])
    out = []
    for prompt in prompts:
        for f in pick:
            result = ground(model, arm['family'], frame_image(cfg, f), prompt, arm, call_seed(cfg['seed'], arm, f['frame_id']))
            parsed = parse('internvl' if arm['family'] == 'internvl' else 'qwen', result)
            out.append({'prompt': prompt, 'frame_id': f['frame_id'], 'category': f['category'], 'gt': f['gt'], **parsed,
                        'score': score_frame(parsed['found'], f['gt']),
                        'max_coordinate': max((max(b['box']) for b in parsed['found']), default=None), **result})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({'arm': name, **arm, 'calls': out}, indent=1) + '\n')
    return [{k: c[k] for k in ('frame_id', 'category', 'status', 'tokens', 'seconds')} | {'best_iou': c['score']['best_iou'],
            'prompt': c['prompt'][:30]} for c in out]

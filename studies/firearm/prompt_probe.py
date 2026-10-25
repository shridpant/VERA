#!/usr/bin/env python3
"""Bounded prompt-format probe on pilot windows (one model load, several prompt directories).

For each prompt directory (a candidate version of studies/firearm/prompts) and each pilot window,
it asks G and AG with the window's saved inputs and records raw text, the tagged decode and the
parsed slot statuses. It computes no localization score: the question is only whether the answer
has the declared format and contains coordinates at all. Every output goes into the prompt log.

  python studies/firearm/prompt_probe.py --config studies/firearm/experiment_A_G_AG.json \
      --run-dir PILOT_RUN --prompt-dirs DIR[,DIR] --windows VIDEO:ANCHOR[,...] --out OUT.json
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--prompt-dirs', required=True)
    ap.add_argument('--windows', required=True)
    ap.add_argument('--conditions', default='G,AG')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from PIL import Image
    from vera_runner.backends import BACKENDS
    from vera_runner.conditions import build_prompt, sha
    from vera_runner.config import prepare_environment
    from vera_runner.slots import final_output_label, parse_slots
    from vera_runner.study import load_study
    study = load_study(args.config)
    cfg = study['runner']
    prepare_environment(cfg)
    backend = BACKENDS[cfg['model']['backend']](cfg)
    results = []
    for prompt_dir in args.prompt_dirs.split(','):
        for condition in args.conditions.split(','):
            question = build_prompt(cfg, condition, prompt_dir)
            for item in args.windows.split(','):
                video, anchor = item.rsplit(':', 1)
                inputs = Path(args.run_dir) / 'videos' / video / 'inputs' / anchor
                images = [Image.open(inputs / f'{i:02d}.png').convert('RGB') for i in range(8)]
                prediction = backend.predict(images, question=question)
                call = backend.last_call
                parsed = parse_slots(call['tagged_text'])
                results.append({'prompt_dir': prompt_dir, 'prompt_sha256': sha(question), 'condition': condition,
                                'video': video, 'anchor': int(anchor), 'raw_response': prediction['raw_response'],
                                'tagged_text': call['tagged_text'], 'eos': call['eos'], 'generated_tokens': call['generated_tokens'],
                                'statuses': [s['status'] for s in parsed['slots']],
                                'label': final_output_label(prediction['raw_response']) if condition == 'AG' else None})
                print(Path(prompt_dir).name, condition, video, anchor, results[-1]['statuses'], flush=True)
    Path(args.out).write_text(json.dumps({'results': results}, indent=1) + '\n')


if __name__ == '__main__':
    main()

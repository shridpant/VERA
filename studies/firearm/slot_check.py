#!/usr/bin/env python3
"""Diagnostic: can InternVL2 tie 'FrameK' to the K-th image under VERA's packing?

batch_chat with num_patches_list=[8] puts all eight images at 'Frame1: <image>' and leaves
'Frame2..8: <image>' as literal text. For windows from a finished (pilot) run, this keeps one slot
k and replaces the other seven with the runner's grey image, then asks the G prompt. If slot
references are interpretable, boxes appear on Frame(k+1) only; an all-grey window is the control.
The outputs are a check of the input convention, never study data.

  python studies/firearm/slot_check.py --config studies/firearm/experiment_A_G_AG.json \
      --run-dir RUN --windows VIDEO:ANCHOR[,VIDEO:ANCHOR] --keep 0,3,7 --out OUT.json
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
    ap.add_argument('--windows', required=True)
    ap.add_argument('--keep', default='0,3,7')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from PIL import Image
    from vera_runner.backends import BACKENDS, intervene
    from vera_runner.conditions import build_prompt
    from vera_runner.config import prepare_environment
    from vera_runner.slots import parse_slots
    from vera_runner.study import load_study
    study = load_study(args.config)
    cfg = study['runner']
    prepare_environment(cfg)
    question = build_prompt(cfg, 'G', study['conditions']['prompt_dir'])
    backend = BACKENDS[cfg['model']['backend']](cfg)
    grey = {**cfg['intervention'], 'mode': 'grey'}
    results = []
    for item in args.windows.split(','):
        video, anchor = item.rsplit(':', 1)
        inputs = Path(args.run_dir) / 'videos' / video / 'inputs' / anchor
        images = [Image.open(inputs / f'{i:02d}.png').convert('RGB') for i in range(8)]
        variants = [('keep', int(k)) for k in args.keep.split(',')] + [('all_grey', None)]
        for kind, k in variants:
            shown = [im if i == k else intervene(im, grey) for i, im in enumerate(images)]
            prediction = backend.predict(shown, question=question)
            parsed = parse_slots(backend.last_call['tagged_text'])
            results.append({'video': video, 'anchor': int(anchor), 'variant': kind, 'kept_slot': k,
                            'raw_response': prediction['raw_response'], 'tagged_text': backend.last_call['tagged_text'],
                            'statuses': [s['status'] for s in parsed['slots']],
                            'slots_with_boxes': [s['slot'] for s in parsed['slots'] if s['status'] == 'boxes'],
                            'eos': backend.last_call['eos'], 'generated_tokens': backend.last_call['generated_tokens']})
            print(json.dumps(results[-1]['slots_with_boxes']), kind, k, video, anchor, flush=True)
    Path(args.out).write_text(json.dumps({'question': question, 'results': results}, indent=1) + '\n')


if __name__ == '__main__':
    main()

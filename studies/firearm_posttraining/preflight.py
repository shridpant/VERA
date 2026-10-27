"""CPU-only target/view/model inventory. Hashes original weights once, never copies them."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from .data import digest, load_split, write_json, canonical_hash
from .model import NativeModel, instruction


def run(args):
    from transformers import AutoProcessor
    from PIL import Image
    book = json.loads(Path(args.codebook).read_text())
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    fake = NativeModel.__new__(NativeModel)
    fake.tokenizer = processor.tokenizer
    report = {'model': args.model, 'python': sys.version, 'platform': platform.platform(),
              'codebook_sha256': digest(args.codebook), 'splits': {}}
    for name in ('fit', 'stop'):
        rows = load_split(args.data, name)
        lengths, unknown = [], 0
        for row in rows:
            for grounded in (False, True):
                ids, kinds, _ = fake.completion(row, grounded)
                lengths.append(len(ids))
                unknown += sum(k == 0 for k in kinds)
            for view in row['views']:
                with Image.open(view['path']) as image:
                    image.verify()
                with Image.open(view['path']) as image:
                    if image.size != (row['width'], row['height']):
                        raise ValueError('frame/source dimensions differ')
                if digest(view['path']) != view['sha256']:
                    raise ValueError('frame changed after audit')
        report['splits'][name] = {'videos': len(rows), 'maximum_target_tokens': max(lengths),
                                 'unknown_tokens_masked': unknown, 'frames_verified': len(rows) * 8}
    files = sorted(p for p in Path(args.model).iterdir() if p.is_file())
    report['model_files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}
    report['model_manifest_hash'] = canonical_hash(report['model_files'])
    report['prompt_hashes'] = {kind: canonical_hash(instruction(book, kind == 'grounded')) for kind in ('labels', 'grounded')}
    report['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv'], text=True)
    write_json(args.output, report, immutable=True)
    print(json.dumps(report['splits']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    p.add_argument('--output', required=True)
    run(p.parse_args())

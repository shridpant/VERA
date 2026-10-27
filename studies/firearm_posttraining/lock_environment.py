"""Create a hash-locked offline wheelhouse from the exact tested environment.

Do not run in parallel with package installation. Restores via pip install
--no-index --find-links WHEELS --require-hashes -r requirements.lock.
"""
import argparse
import hashlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from packaging.utils import canonicalize_name, parse_wheel_filename
from .data import digest, write_json


def run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
    (root / 'requirements-resolved.txt').write_text(freeze)
    wheels = root / 'wheels'
    wheels.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
                    '-r', str(root / 'requirements-resolved.txt'), '--dest', str(wheels)], check=True)
    entries = {}
    for path in sorted(wheels.glob('*.whl')):
        name, version, _, _ = parse_wheel_filename(path.name)
        entries.setdefault(f'{canonicalize_name(name)}=={version}', []).append(digest(path))
    wanted = {canonicalize_name(p.metadata['Name']) + '==' + p.version for p in importlib.metadata.distributions()}
    if set(entries) != wanted:
        raise RuntimeError(f'wheelhouse coverage differs: missing={wanted-set(entries)} extra={set(entries)-wanted}')
    lock = '\n'.join(key + ' ' + ' '.join('--hash=sha256:' + h for h in hashes) for key, hashes in sorted(entries.items())) + '\n'
    (root / 'requirements.lock').write_text(lock)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    write_json(root / 'wheelhouse.json', {'python': sys.version, 'lock_sha256': digest(root / 'requirements.lock'),
               'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in wheels.glob('*.whl')}})
    print('LOCKED', len(entries), 'distributions', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    run(p.parse_args().root)

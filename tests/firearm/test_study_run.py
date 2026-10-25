import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
import numpy as np
from vera_runner.backends import Mock
from vera_runner.firearm_data import build_inventory, stage, write_inventory, write_registry
from vera_runner.ledger import read_jsonl
from vera_runner.study import Timeout, load_study, reconcile, run_study
from tests.firearm.fixtures import write_video


def bright_after(n):
    return lambda i: np.full((48, 64, 3), 220 if i >= n else 20, np.uint8)


def make_study(td, prompt_dir='studies/firearm/prompts', **execution):
    td = Path(td)
    original = td / 'original'
    write_video(original / 'Handgun' / 'PAH1_C1_P1_V1_HB_1' / 'video.mp4', 40, value=bright_after(20))
    write_video(original / 'No_Gun' / 'N1_C1_P2_V1_HB_1' / 'video.mp4', 20, value=bright_after(100))
    rows = stage(build_inventory(original, 1), original, td / 'staged')
    write_inventory(rows, td)
    write_registry(td / 'registry.json', td / 'staged', td / 'features')
    return load_study({'study_id': 'test', 'runner': {'dataset': 'firearm_visilab', 'registry': str(td / 'registry.json'),
                                                      'model': {'backend': 'mock'}},
                       'conditions': {'prompt_dir': prompt_dir},
                       'data': {'inventory': str(td / 'inventory.json')},
                       'execution': {'run_dir': str(td / 'run'), **execution}})


class Recording(Mock):
    """Mock that keeps every question and can fail on chosen call numbers."""
    def __init__(self, cfg, fail_on=(), always_fail=False):
        super().__init__(cfg)
        self.questions, self.calls, self.fail_on, self.always_fail = [], 0, set(fail_on), always_fail

    def predict(self, images, on_text=None, question=None):
        self.calls += 1
        if self.always_fail or self.calls in self.fail_on:
            raise RuntimeError(f'injected failure at call {self.calls}')
        self.questions.append(question)
        return super().predict(images, on_text, question)


def records(run_dir, condition):
    return read_jsonl(Path(run_dir) / 'records' / f'{condition}.jsonl')[0]


class StudyRunTests(unittest.TestCase):
    def test_full_run_identical_inputs_and_no_history(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td)
            backend = Recording(study['runner'])
            self.assertEqual(run_study(study, backend=backend), 0)
            spec = json.loads((Path(td) / 'run/spec.json').read_text())
            self.assertEqual(spec['units'], (3 + 2) * 3)
            result = reconcile(study, Path(td) / 'run')
            self.assertEqual(result['counts'], {'complete': 15})
            by = {c: {(r['video'], r['anchor']): r for r in records(Path(td) / 'run', c)} for c in ('A', 'G', 'AG')}
            for key, a in by['A'].items():
                self.assertEqual(a['rgb_sha256'], by['G'][key]['rgb_sha256'])
                self.assertEqual(a['rgb_sha256'], by['AG'][key]['rgb_sha256'])
                self.assertEqual(a['hashes']['input'], by['AG'][key]['hashes']['input'])
            # Every question is exactly its condition's saved prompt: no earlier answer ever reaches a call.
            self.assertEqual(set(backend.questions), set(spec['prompts'].values()))
            self.assertEqual(backend.questions[:3], [spec['prompts'][c] for c in ('A', 'G', 'AG')])
            hand = [r for r in records(Path(td) / 'run', 'G') if r['video'].startswith('Handgun')]
            self.assertEqual(hand[0]['grounding']['response_status'], 'ok')
            # Both clips are shorter than the 300-frame window, so every window samples the same frames.
            self.assertEqual([r['label'] for r in records(Path(td) / 'run', 'A')], [0] * 5)
            self.assertEqual([s['status'] for s in hand[0]['grounding']['slots']], ['empty'] * 4 + ['boxes'] * 4)
            self.assertEqual(len(list((Path(td) / 'run/videos/Handgun__PAH1_C1_P1_V1_HB_1/inputs/16').glob('*.png'))), 8)
            self.assertTrue((Path(td) / 'run/model_input_AG.txt').exists())

    def test_resume_after_failure_keeps_the_attempt_and_matches_a_clean_run(self):
        with tempfile.TemporaryDirectory() as clean_dir, tempfile.TemporaryDirectory() as td:
            clean = make_study(clean_dir)
            run_study(clean, backend=Recording(clean['runner']))
            study = make_study(td, max_consecutive_failures=1)
            self.assertEqual(run_study(study, backend=Recording(study['runner'], fail_on={5})), 1)
            self.assertEqual(run_study(study, backend=Recording(study['runner'])), 0)
            for c in ('A', 'G', 'AG'):
                rows = records(Path(td) / 'run', c)
                complete = [r for r in rows if r['status'] == 'complete']
                units = [(r['video'], r['anchor']) for r in complete]
                self.assertEqual(len(units), len(set(units)), c)
                want = {(r['video'], r['anchor']): r['raw_response'] for r in records(Path(clean_dir) / 'run', c)}
                self.assertEqual({u: r['raw_response'] for u, r in zip(units, complete)}, want)
            failed = [r for c in ('A', 'G', 'AG') for r in records(Path(td) / 'run', c) if r['status'] == 'failed']
            self.assertEqual(len(failed), 1)
            self.assertIn('injected failure at call 5', failed[0]['error'])
            events = read_jsonl(Path(td) / 'run/ledger.jsonl')[0]
            unit = f"{failed[0]['video']}|{failed[0]['anchor']}|{failed[0]['condition']}"
            self.assertEqual([e['status'] for e in events if e.get('unit') == unit and e['event'] == 'end'], ['failed', 'complete'])
            self.assertEqual(len(list((Path(td) / 'run/sessions').iterdir())), 2)

    def test_changed_prompt_refuses_to_resume(self):
        with tempfile.TemporaryDirectory() as td:
            prompts = Path(td) / 'prompts'
            shutil.copytree('studies/firearm/prompts', prompts)
            study = make_study(td, prompt_dir=str(prompts))
            run_study(study, backend=Recording(study['runner']))
            (prompts / 'grounding_core.txt').write_text('Changed.\n')
            with self.assertRaises(ValueError):
                run_study(study, backend=Recording(study['runner']))

    def test_changed_frames_rerun_only_their_windows(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td)
            run_study(study, backend=Recording(study['runner']))
            frame = Path(td) / 'run/videos/No_Gun__N1_C1_P2_V1_HB_1/frames/000005.jpg'
            from PIL import Image
            Image.new('RGB', (64, 48), (255, 255, 255)).save(frame)
            backend = Recording(study['runner'])
            self.assertEqual(run_study(study, backend=backend), 0)
            self.assertEqual(backend.calls, 6)  # both windows of that 20-frame clip sample frame 5; the other video is untouched
            self.assertEqual(reconcile(study, Path(td) / 'run')['counts'], {'complete': 15})

    def test_exhausted_units_get_a_disposition(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td, max_attempts=2, max_consecutive_failures=100)
            for _ in range(2):
                run_study(study, backend=Recording(study['runner'], always_fail=True))
            self.assertEqual(run_study(study, backend=Recording(study['runner'])), 0)
            self.assertEqual(reconcile(study, Path(td) / 'run')['counts'], {'exhausted': 15})

    def test_timeout_fires_its_callback(self):
        fired = threading.Event()
        with Timeout(0.05, fired.set):
            fired.wait(1)
        self.assertTrue(fired.is_set())


class CommandLineTests(unittest.TestCase):
    def test_smoke_run_and_reconcile_commands(self):
        import subprocess, sys
        from vera_runner.config import ROOT
        with tempfile.TemporaryDirectory() as td:
            make_study(td)
            config = {'study_id': 'cli', 'runner': {'dataset': 'firearm_visilab', 'registry': str(Path(td) / 'registry.json'),
                                                    'model': {'backend': 'mock'}},
                      'data': {'inventory': str(Path(td) / 'inventory.json')}, 'execution': {'run_dir': str(Path(td) / 'main')}}
            (Path(td) / 'study.json').write_text(json.dumps(config))
            common = ['--config', str(Path(td) / 'study.json'), '--run-dir', str(Path(td) / 'smoke'),
                      '--videos', 'Handgun__PAH1_C1_P1_V1_HB_1', '--max-windows', '1', '--study-id', 'cli_smoke']
            for command in ('run', 'reconcile'):
                out = subprocess.run([sys.executable, str(ROOT / 'run_firearm_study.py'), command, *common],
                                     cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)
            spec = json.loads((Path(td) / 'smoke/spec.json').read_text())
            self.assertEqual((spec['study_id'], spec['units']), ('cli_smoke', 3))
            self.assertFalse((Path(td) / 'main').exists())

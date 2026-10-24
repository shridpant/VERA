"""HTTP tests that exercise the educational UI without loading a VLM."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from tools import demo_server
from tools.stage_view import stage_manifest


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.library = cls.root / 'library'
        cls.library.mkdir()
        for name in ('sample.mp4', 'other.mov', 'third.avi', 'notes.txt'):
            (cls.library / name).write_bytes(b'not a real video: API selection test')
        demo_server.LIBRARY = cls.library
        demo_server.RUNS = cls.root / 'runs'
        demo_server.CURRENT = demo_server.RUNS / 'current'
        demo_server.JOBS.clear()
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), demo_server.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def call(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        request_headers = dict(headers or {})
        if method == 'POST':
            request_headers.setdefault('X-VERA-Token', demo_server.SESSION_TOKEN)
        client.request(method, path, body=body, headers=request_headers)
        response = client.getresponse()
        data = response.read()
        code = response.status
        client.close()
        return code, data

    def test_pages_and_assets(self):
        for path, marker in [('/', b'Run VERA'),
                             ('/how-it-works', b'Eight source frames'),
                             ('/frame-game', b'Frame game'),
                             ('/scores', b'Score timeline'),
                             ('/static/model.js', b'api/upload'),
                             ('/api/defaults', b'missing_dependencies')]:
            code, content = self.call('GET', path)
            self.assertEqual(code, 200, path)
            self.assertIn(marker, content)

    def test_lucky_uses_random_index_over_sorted_videos(self):
        videos = [str(self.library / name) for name in ('other.mov', 'sample.mp4', 'third.avi')]
        for index in range(3):
            with mock.patch.object(demo_server.secrets, 'randbelow', return_value=index) as pick:
                code, content = self.call('GET', '/api/lucky')
            self.assertEqual(code, 200)
            result = json.loads(content)
            self.assertEqual(result['path'], videos[index])
            # The die covers every video and only videos; notes.txt is never a candidate.
            self.assertEqual(pick.call_args_list[0].args, (3,))
            self.assertGreaterEqual(result['anchor_seconds'], 0)

    def test_lucky_for_refinement_requires_matching_features(self):
        code, content = self.call('GET', '/api/lucky?target=refine')
        self.assertEqual(code, 400)
        self.assertIn(b'required for this stage', content)
        feature_dir = self.root / 'features'
        feature_dir.mkdir(exist_ok=True)
        (feature_dir / 'third.npy').write_bytes(b'fixture')
        with mock.patch.object(demo_server, 'FEATURE_DIR', feature_dir), \
             mock.patch.object(demo_server.secrets, 'randbelow', return_value=0) as pick:
            code, content = self.call('GET', '/api/lucky?target=refine')
        self.assertEqual(code, 200)
        self.assertEqual(pick.call_args_list[0].args, (1,))
        result = json.loads(content)
        self.assertEqual(result['path'], str(self.library / 'third.avi'))
        self.assertEqual(result['feature_file'], str(feature_dir / 'third.npy'))
        code, _ = self.call('GET', '/api/lucky?target=everything')
        self.assertEqual(code, 400)

    def test_post_requires_same_origin_token(self):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        client.request('POST', '/api/run', body=b'{}',
                       headers={'Content-Type': 'application/json'})
        response = client.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        client.close()
        code, content = self.call('POST', '/api/run', b'{}',
                                  {'Content-Type': 'application/json',
                                   'Origin': 'https://another.example'})
        self.assertEqual(code, 403)
        self.assertIn(b'Cross-origin', content)
        code, _ = self.call('GET', '/api/defaults', headers={'Host': 'another.example'})
        self.assertEqual(code, 403)

    def test_upload_accepts_video_and_rejects_other_suffix(self):
        payload = b'example bytes'
        code, content = self.call('POST', '/api/upload', payload,
                                  {'X-Filename': 'clip.mp4', 'Content-Type': 'application/octet-stream'})
        self.assertEqual(code, 201)
        result = json.loads(content)
        self.assertEqual(Path(result['path']).read_bytes(), payload)
        code, content = self.call('POST', '/api/upload', payload,
                                  {'X-Filename': '../bad.txt', 'Content-Type': 'application/octet-stream'})
        self.assertEqual(code, 400)
        self.assertIn(b'Upload must be', content)

    def test_run_rejects_missing_video_without_model_launch(self):
        request = json.dumps({'video': str(self.root / 'missing.mp4')}).encode()
        code, content = self.call('POST', '/api/run', request,
                                  {'Content-Type': 'application/json'})
        self.assertEqual(code, 400)
        self.assertIn(b'existing video path', content)

    def test_preflight_names_missing_cv2_before_start(self):
        request = json.dumps({'video': str(self.library / 'sample.mp4')}).encode()
        with mock.patch.object(demo_server, 'missing_dependencies', return_value=['cv2']), \
             mock.patch.object(demo_server.subprocess, 'Popen') as launched:
            code, content = self.call('POST', '/api/run', request,
                                      {'Content-Type': 'application/json'})
        self.assertEqual(code, 400)
        self.assertIn(b'missing cv2', content)
        launched.assert_not_called()

    def test_full_video_stage_config_and_scratch_replacement(self):
        feature = self.root / 'sample.npy'
        feature.write_bytes(b'test metadata only')
        demo_server.CURRENT.mkdir(parents=True, exist_ok=True)
        old = demo_server.CURRENT / 'previous-run.txt'
        old.write_text('scratch')
        request = json.dumps({'video': str(self.library / 'sample.mp4'),
                              'target_stage': 'refine', 'feature_file': str(feature)}).encode()
        fake_process = mock.Mock()
        fake_process.poll.return_value = 0
        fake_process.returncode = 0
        with mock.patch.object(demo_server, 'missing_dependencies', return_value=[]), \
             mock.patch.object(demo_server.WARM, 'stop') as release, \
             mock.patch.object(demo_server.subprocess, 'Popen', return_value=fake_process) as launched:
            code, content = self.call('POST', '/api/run', request,
                                      {'Content-Type': 'application/json'})
        self.assertEqual(code, 202, content)
        release.assert_called_once()
        self.assertFalse(old.exists())
        cfg = json.loads((demo_server.CURRENT / 'config.json').read_text())
        self.assertEqual(cfg['stages'], ['preprocess', 'infer', 'refine'])
        self.assertEqual(cfg['selection']['video_ids'], ['sample'])
        self.assertTrue((demo_server.CURRENT / 'input' / 'sample.mp4').is_symlink())
        self.assertIn('run_vera.py', launched.call_args.args[0][1])
        job = json.loads(content)['job']
        output = demo_server.CURRENT / 'pipeline' / 'run 1' / 'output'
        (output / 'sample').mkdir(parents=True)
        (output / 'status.json').write_text(json.dumps({'stage': 'inference', 'completed_requests': 2}))
        (output / 'sample' / 'coverage.json').write_text(json.dumps({'selected_anchors': [0, 16, 32]}))
        code, content = self.call('GET', '/api/status?job=' + job)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(content)['mode'], 'video')
        self.assertEqual(json.loads(content)['progress']['total_windows'], 3)
        self.assertTrue((demo_server.CURRENT / 'stage_manifest.json').is_file())
        rows = [{'anchor': anchor, 'start': 0, 'end': 48, 'condition': 'baseline',
                 'frame_indices': list(range(8)), 'strict_label': label, 'raw_response': 'Output: %s' % label,
                 'seconds': 1.5, 'input_token_ids': [[1, 2, 3]]} for anchor, label in ((0, 0), (16, 1))]
        (output / 'sample' / 'raw_responses.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (output / 'sample' / 'video.json').write_text(json.dumps({'fps_actual': 25.0}))
        code, content = self.call('GET', '/api/windows?job=' + job)
        self.assertEqual(code, 200)
        windows = json.loads(content)
        self.assertEqual([row['strict_label'] for row in windows['windows']], [0, 1])
        self.assertNotIn('input_token_ids', windows['windows'][0])
        self.assertEqual((windows['fps'], windows['inputs']), (25.0, 'pipeline/run 1/output/sample/inputs'))

    def test_stage_inventory_and_artifact_confinement(self):
        folder = self.root / 'inventory'
        (folder / 'probe' / 'frames').mkdir(parents=True)
        (folder / 'probe' / 'frames' / '00_source_000000.png').write_bytes(b'png')
        (folder / 'probe' / 'prompt.txt').write_text('prompt')
        stages = stage_manifest(folder, 'window', 'sample')
        self.assertEqual(stages[1]['stage'], '01_sampled_input')
        self.assertEqual(stages[1]['artifacts'][0]['kind'], 'directory')
        with self.assertRaises(ValueError):
            demo_server.output_file({'folder': folder}, '../outside.txt')

    def test_directory_listing_stays_inside_active_run(self):
        folder = self.root / 'listing'
        (folder / 'frames').mkdir(parents=True)
        (folder / 'frames' / '00.png').write_bytes(b'png')
        demo_server.JOBS.clear()
        finished = mock.Mock()
        finished.poll.return_value = 0
        finished.returncode = 0
        demo_server.JOBS['list'] = {'folder': folder, 'mode': 'window',
                                    'video_id': 'sample', 'process': finished}
        code, content = self.call('GET', '/api/list?job=list&dir=frames')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(content)['items'][0]['path'], 'frames/00.png')
        code, _ = self.call('GET', '/api/list?job=list&dir=../library')
        self.assertEqual(code, 400)

    def test_completed_scratch_run_restores_after_server_restart(self):
        demo_server.CURRENT.mkdir(parents=True, exist_ok=True)
        (demo_server.CURRENT / 'lab_job.json').write_text(json.dumps({
            'job': 'restored', 'mode': 'window', 'video_id': 'sample',
            'target_stage': 'window'}))
        (demo_server.CURRENT / 'probe').mkdir(exist_ok=True)
        (demo_server.CURRENT / 'probe' / 'result.json').write_text('{}')
        demo_server.JOBS.clear()
        demo_server.restore_current_job()
        self.assertEqual(demo_server.JOBS['restored']['process'].poll(), 0)
        code, content = self.call('GET', '/api/defaults')
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(content)['current_job'], 'restored')

    def test_uploaded_video_moves_into_single_scratch_run(self):
        code, content = self.call('POST', '/api/upload', b'video bytes',
                                  {'X-Filename': 'my_clip.mp4',
                                   'Content-Type': 'application/octet-stream'})
        self.assertEqual(code, 201)
        uploaded = json.loads(content)['path']
        request = json.dumps({'video': uploaded, 'target_stage': 'window'}).encode()
        with mock.patch.object(demo_server, 'missing_dependencies', return_value=[]), \
             mock.patch.object(demo_server.WARM, 'submit', return_value=finished_job()):
            code, content = self.call('POST', '/api/run', request,
                                      {'Content-Type': 'application/json'})
        self.assertEqual(code, 202, content)
        self.assertFalse(Path(uploaded).exists())
        self.assertEqual((demo_server.CURRENT / 'uploaded' / 'my_clip.mp4').read_bytes(), b'video bytes')
        self.assertFalse((demo_server.RUNS / 'uploads').exists())

    def test_window_model_settings(self):
        video = str(self.library / 'sample.mp4')
        cases = [({}, 'auto', True, True),
                 ({'device': 'cuda', 'dtype': 'bfloat16'}, 'bfloat16', False, True),
                 ({'model_path': 'OpenGVLab/InternVL2-2B'}, 'auto', True, False),
                 ({'model_path': str(self.root)}, 'auto', True, True)]
        for extra, dtype, fallback, pinned in cases:
            request = json.dumps({'video': video, 'target_stage': 'window', **extra}).encode()
            with mock.patch.object(demo_server, 'missing_dependencies', return_value=[]), \
                 mock.patch.object(demo_server.WARM, 'submit', return_value=finished_job()) as submit:
                code, content = self.call('POST', '/api/run', request,
                                          {'Content-Type': 'application/json'})
            self.assertEqual(code, 202, content)
            model = submit.call_args.args[0]
            self.assertEqual(model, json.loads((demo_server.CURRENT / 'config.json').read_text())['model'])
            self.assertEqual(model['dtype'], dtype, extra)
            self.assertEqual(model['mps_cpu_fallback'], fallback, extra)
            # Only the baseline repository and local copies keep the pinned baseline commit.
            self.assertEqual(model['revision'] is not None, pinned, extra)
            self.assertEqual(submit.call_args.args[1]['output'], str(demo_server.CURRENT / 'probe'))

    def test_window_probe_rejects_non_finite_anchor(self):
        from tools.window_probe import fresh_backend, probe
        from vera_runner.config import load
        cfg = load(demo_server.BASE_CONFIG)
        for anchor in (float('nan'), float('inf'), -1.0):
            with self.assertRaisesRegex(ValueError, 'finite'):
                probe(cfg, self.library / 'sample.mp4', anchor, self.root / 'never', fresh_backend)
        self.assertFalse((self.root / 'never').exists())


def finished_job():
    job = demo_server.WorkerJob()
    job.finish(0)
    return job


if __name__ == '__main__':
    unittest.main()

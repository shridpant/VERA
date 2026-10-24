"""The warm window process with the mock backend: real subprocesses, no model weights."""
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tools import demo_server

ROOT = Path(__file__).resolve().parents[2]


def write_video(path, frames=320):
    import cv2
    import numpy as np
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 24))
    for index in range(frames):
        writer.write(np.full((24, 32, 3), index % 255, np.uint8))
    writer.release()


def mock_config(folder):
    cfg = json.loads(demo_server.BASE_CONFIG.read_text())
    cfg['model']['backend'] = 'mock'
    path = Path(folder) / 'mock_baseline.json'
    path.write_text(json.dumps(cfg))
    return path


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'Decoding a synthetic video needs OpenCV')
class WarmModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / 'tiny.avi'
        write_video(self.video)
        patches = [mock.patch.object(demo_server, 'BASE_CONFIG', mock_config(self.root)),
                   mock.patch.object(demo_server, 'RUNS', self.root / 'runs'),
                   mock.patch.object(demo_server, 'CURRENT', self.root / 'runs' / 'current'),
                   mock.patch.object(demo_server, 'missing_dependencies', return_value=[])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        demo_server.RUNS.mkdir()
        demo_server.JOBS.clear()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), demo_server.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        demo_server.WARM.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def call(self, method, path, body=None):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        headers = {'Content-Type': 'application/json', 'X-VERA-Token': demo_server.SESSION_TOKEN}
        client.request(method, path, body=body, headers=headers)
        response = client.getresponse()
        data = json.loads(response.read())
        client.close()
        return response.status, data

    def run_window(self, **settings):
        request = {'video': str(self.video), 'target_stage': 'window', 'anchor_seconds': 4.0, **settings}
        code, started = self.call('POST', '/api/run', json.dumps(request).encode())
        self.assertEqual(code, 202, started)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            code, status = self.call('GET', '/api/status?job=' + started['job'])
            if status['state'] != 'running':
                self.assertEqual(status['state'], 'complete', status['log'])
                return status
            time.sleep(0.1)
        self.fail('Window request did not finish')

    def test_second_window_reuses_loaded_model(self):
        first = self.run_window()
        pid = demo_server.WARM.process.pid
        self.assertFalse(first['result']['model_reused'])
        self.assertIn('Started a model process', first['log'])
        warm = self.call('GET', '/api/defaults')[1]['warm_model']
        # The status names the device the worker resolved, not the requested 'auto'.
        self.assertEqual((warm['loaded'], warm['device']), (True, 'mock'))
        second = self.run_window(questions='Is anything on fire?')
        self.assertEqual(demo_server.WARM.process.pid, pid)
        self.assertTrue(second['result']['model_reused'])
        self.assertIn('Reusing the loaded model', second['log'])
        # The new questions reach the reused backend, not only the saved prompt file.
        prompt = (demo_server.CURRENT / 'probe' / 'prompt.txt').read_text()
        self.assertIn('Is anything on fire?', prompt)
        self.assertEqual(len(list((demo_server.CURRENT / 'probe' / 'frames').glob('*.png'))), 8)

    def test_changed_model_settings_start_a_new_process(self):
        self.run_window()
        pid = demo_server.WARM.process.pid
        changed = self.run_window(dtype='float32')
        self.assertNotEqual(demo_server.WARM.process.pid, pid)
        self.assertFalse(changed['result']['model_reused'])

    def test_worker_near_idle_exit_is_replaced_not_raced(self):
        self.run_window()
        pid = demo_server.WARM.process.pid
        # Pretend the last reply was almost a full idle period ago: the worker may exit any moment.
        demo_server.WARM.last_reply -= demo_server.WORKER_IDLE_SECONDS - demo_server.WarmModel.IDLE_MARGIN
        self.assertFalse(self.call('GET', '/api/defaults')[1]['warm_model']['loaded'])
        fresh = self.run_window()
        self.assertNotEqual(demo_server.WARM.process.pid, pid)
        self.assertFalse(fresh['result']['model_reused'])

    def test_worker_answers_requests_sent_together(self):
        self.run_window()
        config = str(demo_server.CURRENT / 'config.json')
        requests = ''.join(json.dumps({'config': config, 'video': str(self.video), 'anchor_seconds': anchor,
                                       'output': str(self.root / f'together{anchor}')}) + '\n'
                           for anchor in (1.0, 2.0))
        process = subprocess.Popen([sys.executable, str(ROOT / 'tools' / 'window_probe.py'), '--serve',
                                    '--idle-seconds', '5'], cwd=ROOT, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        replies, _ = process.communicate(requests, timeout=60)
        self.assertEqual([json.loads(line)['ok'] for line in replies.splitlines()], [True, True])

    def test_timeline_prompt_and_manual_boxes(self):
        first = self.run_window()
        job = self.call('GET', '/api/defaults')[1]['current_job']
        timeline = self.call('GET', '/api/timeline?job=' + job)[1]
        events = [event['event'] for event in timeline['events']['run']]
        self.assertEqual(timeline['base']['run'], 'probe')
        self.assertEqual(events[0], 'run')
        self.assertIn('answer', events)
        self.assertEqual(events[-1], 'done')
        window = next(event for event in timeline['events']['run'] if event['event'] == 'window')
        self.assertIsNone(first['result']['boxes'])
        # Event paths are relative to the timeline's base folder.
        self.assertTrue((demo_server.CURRENT / timeline['base']['run'] / window['images'][0]).is_file())
        # The page shows the exact prompt for the questions being typed, from the runner's prompt().
        code, rendered = self.call('GET', '/api/prompt?questions=Is%20anything%20on%20fire%3F')
        self.assertEqual(code, 200)
        self.assertIn('Is anything on fire?', rendered['prompt'])
        self.assertIn('<ref>{}</ref>', rendered['ground_prompt'])
        # A dark mock window is labelled 0, so nothing is boxed until the user asks for frames.
        windows = self.call('GET', '/api/windows?job=' + job)[1]
        self.assertEqual((windows['windows'][0]['strict_label'], windows['windows'][0]['boxes']), (0, []))
        anchor = windows['windows'][0]['anchor']
        body = json.dumps({'job': job, 'phrase': 'the car', 'frames': [{'anchor': anchor, 'slot': 0},
                                                                         {'anchor': anchor, 'slot': 3}]}).encode()
        code, started = self.call('POST', '/api/boxes', body)
        self.assertEqual(code, 202, started)
        deadline = time.monotonic() + 30
        while self.call('GET', '/api/windows?job=' + job)[1]['boxing'] and time.monotonic() < deadline:
            time.sleep(0.1)
        manual = self.call('GET', '/api/windows?job=' + job)[1]['windows'][0]['boxes']
        self.assertEqual([(record['mode'], record['phrase']) for record in manual], [('manual', 'the car')])
        self.assertEqual([(row['slot'], row['status']) for row in manual[0]['frames']], [(0, 'box'), (3, 'box')])
        manual_events = self.call('GET', '/api/timeline?job=' + job)[1]['events']['manual']
        self.assertEqual([event['event'] for event in manual_events].count('box'), 2)
        code, refused = self.call('POST', '/api/boxes', json.dumps({'job': job, 'phrase': 'x', 'frames': [
            {'anchor': anchor, 'slot': 9}]}).encode())
        self.assertEqual(code, 400)

    def test_failed_request_keeps_worker_serving(self):
        self.run_window()
        process = demo_server.WARM.process
        bad = {'config': str(demo_server.CURRENT / 'config.json'), 'video': str(self.root / 'missing.avi'),
               'anchor_seconds': 1.0, 'output': str(self.root / 'bad')}
        job = demo_server.WARM.submit(demo_server.WARM.settings, bad, self.root / 'bad.log')
        self.assertTrue(job.done.wait(30))
        self.assertEqual(job.poll(), 1)
        self.assertIn('Video not found', (self.root / 'bad.log').read_text())
        self.assertIs(demo_server.WARM.process, process)
        self.assertIsNone(process.poll())


class IdleExitTests(unittest.TestCase):
    def test_worker_exits_when_idle(self):
        process = subprocess.Popen([sys.executable, str(ROOT / 'tools' / 'window_probe.py'), '--serve',
                                    '--idle-seconds', '0.5'], cwd=ROOT, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(process.wait(timeout=20), 0)
        finally:
            process.kill()
            process.communicate()


if __name__ == '__main__':
    unittest.main()

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from vera_runner.config import ROOT


REVISION='6fb9ad6924f69424e57fab2ab061d707688f0296'
PROBE='''import json,sys
from vera_runner.backends import resolve_checkpoint
settings=json.loads(sys.argv[1])
try:print(json.dumps({'source':resolve_checkpoint(settings)}))
except ValueError as e:print(json.dumps({'error':str(e)}))
'''


class CheckpointResolutionTests(unittest.TestCase):
    def resolve(self,home,**settings):
        # HF_HOME is read when huggingface_hub is imported, so each case uses a fresh process.
        settings={'path':'OpenGVLab/InternVL2-8B','revision':REVISION,'local_files_only':True,**settings}
        env={**os.environ,'HF_HOME':str(home),'HF_HUB_OFFLINE':'1'}
        out=subprocess.run([sys.executable,'-c',PROBE,json.dumps(settings)],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
        return json.loads(out.stdout)

    def test_cached_hub_snapshot_is_local(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot=Path(td)/'hub/models--OpenGVLab--InternVL2-8B/snapshots'/REVISION;snapshot.mkdir(parents=True)
            (snapshot/'config.json').write_text('{}')
            self.assertEqual(Path(self.resolve(td)['source']).resolve(),snapshot.resolve())

    def test_uncached_hub_id_needs_download_permission(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIn('cached Hub snapshot',self.resolve(td,path='OpenGVLab/InternVL2-2B')['error'])
            self.assertEqual(self.resolve(td,path='OpenGVLab/InternVL2-2B',local_files_only=False)['source'],'OpenGVLab/InternVL2-2B')

    def test_local_directory_wins(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(self.resolve(td,path=td)['source'],str(Path(td).resolve()))

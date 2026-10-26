import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vera_runner.ledger import JsonlAppender
from vera_runner.study import final_records, reconcile, run_study
from studies.firearm.refinement import reasons_for, run_refinement
from tests.firearm.fixtures import write_video
from tests.firearm.test_study_run import Recording, make_study

CFG = {'neighbor_fraction': 0.15, 'similarity_temperature': 10.0, 'smoothing_size': 15, 'smoothing_sigma': 10.0,
       'round_decimals': 1, 'position_weighting': True}


class ReasonTests(unittest.TestCase):
    def test_every_applicable_reason(self):
        good = [{'label': 0}] * 15
        self.assertEqual(reasons_for(good, np.ones((15, 4)), CFG), [])
        self.assertEqual(reasons_for(good[:14], np.ones((14, 4)), CFG), ['too_few_windows'])
        self.assertEqual(reasons_for(good[:14] + [{'label': None}], np.ones((15, 4)), CFG), ['invalid_labels'])
        self.assertEqual(reasons_for(good[:14] + [None], None, CFG), ['incomplete_inference', 'missing_features'])
        self.assertEqual(reasons_for(good, np.ones((16, 4)), CFG), ['missing_features'])


class RefineRunTests(unittest.TestCase):
    def test_a_and_ag_share_features_and_short_clips_are_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td)
            # A third, long clip: 250 frames = 16 windows, enough for the 15-tap smoothing.
            write_video(Path(td) / 'original/Machine_Gun/PAW1_C1_P1_V1_HB_1/video.mp4', 250)
            from vera_runner.firearm_data import build_inventory, stage, write_inventory
            write_inventory(stage(build_inventory(Path(td) / 'original', 1), Path(td) / 'original', Path(td) / 'staged'), td)
            run_study(study, backend=Recording(study['runner']))
            reconcile(study, Path(td) / 'run')
            final, spec = final_records(study, Path(td) / 'run')
            feats = Path(td) / 'features'
            feats.mkdir()
            rng = np.random.default_rng(1)
            for v in spec['videos']:
                np.save(feats / f"{v['id']}.npy", rng.normal(size=(len(v['anchors']), 8)))
            rows = run_refinement(study, spec, final, feats, Path(td) / 'refined')
            by = {(r['video'], r['condition']): r for r in rows}
            long = 'Machine_Gun__PAW1_C1_P1_V1_HB_1'
            self.assertTrue(by[(long, 'A')]['refined'] and by[(long, 'AG')]['refined'])
            self.assertEqual(by[(long, 'A')]['feature_sha256'], by[(long, 'AG')]['feature_sha256'])
            self.assertEqual(by[('Handgun__PAH1_C1_P1_V1_HB_1', 'A')]['reasons'], 'too_few_windows')
            self.assertEqual(len(np.load(Path(td) / 'refined/A' / long / 'frame_scores.npy')), 250)
            with (Path(td) / 'refined/refinement.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)

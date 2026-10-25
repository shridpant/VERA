import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vera_runner.firearm_data import build_inventory, stage, write_inventory, write_registry
from vera_runner.study import final_records, load_study, reconcile, run_study
from studies.firearm.refinement import run_refinement
from studies.firearm.analysis.run import analyze
from tests.firearm.fixtures import coco, write_video
from tests.firearm.test_study_run import Recording


def bright_after(n, size):
    return lambda i: np.full((size[1], size[0], 3), 220 if i >= n else 20, np.uint8)


class Garbling(Recording):
    """Mock whose AG answers lose their Output field on chosen calls (invalid labels)."""
    def __init__(self, cfg, garble):
        super().__init__(cfg)
        self.garble, self.ag_calls = set(garble), 0

    def predict(self, images, on_text=None, question=None):
        out = super().predict(images, on_text, question)
        if 'Firearm localization' in question and 'Is there any anomaly' in question:
            self.ag_calls += 1
            if self.ag_calls in self.garble:
                out['raw_response'] = out['raw_response'].replace('Output:', 'Result:')
        return out


def build(td):
    td = Path(td)
    original = td / 'original'
    hand, mg, neg = (64, 48), (96, 54), (64, 48)
    # The mock boxes every bright slot at [100, 200, 600, 700] on the 0-1000 grid.
    exact = [0.1 * 64, 0.2 * 48, 0.5 * 64, 0.5 * 48]
    write_video(original / 'Handgun/PAH1_C1_P1_V1_HB_1/video.mp4', 250, hand, 25, bright_after(100, hand))
    (original / 'Handgun/PAH1_C1_P1_V1_HB_1/label.json').write_text(json.dumps(coco(250, hand, {k: exact for k in range(101, 251)})))
    write_video(original / 'Machine_Gun/PAW1_C2_P3_V1_HB_1/video.mp4', 40, mg, 30, bright_after(0, mg))
    (original / 'Machine_Gun/PAW1_C2_P3_V1_HB_1/label.json').write_text(
        json.dumps(coco(40, mg, {k: [80, 40, 10, 10] for k in range(1, 41)}, 'Machine_Gun')))
    write_video(original / 'No_Gun/N1_C1_P2_V1_HB_1/video.mp4', 250, neg, 25, bright_after(50, neg))
    rows = stage(build_inventory(original, 2), original, td / 'staged')
    write_inventory(rows, td)
    write_registry(td / 'registry.json', td / 'staged', td / 'features')
    study = load_study({'study_id': 'analysis', 'runner': {'dataset': 'firearm_visilab', 'registry': str(td / 'registry.json'),
                                                           'model': {'backend': 'mock'}},
                        'data': {'inventory': str(td / 'inventory.json')},
                        'execution': {'run_dir': str(td / 'run')}, 'features': {'dir': str(td / 'features')},
                        'analysis': {'bootstrap_resamples': 50}})
    run_study(study, backend=Garbling(study['runner'], garble={2, 3}))
    reconcile(study, td / 'run')
    final, spec = final_records(study, td / 'run')
    (td / 'features').mkdir()
    rng = np.random.default_rng(0)
    for v in spec['videos']:
        np.save(td / 'features' / f"{v['id']}.npy", rng.normal(size=(len(v['anchors']), 8)))
    run_refinement(study, spec, final, td / 'features', td / 'run/refined')
    return study, original


def read(out, tid):
    with (Path(out) / 'tables' / f'{tid}.csv').open() as f:
        return list(csv.DictReader(f))


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.study, original = build(cls.td.name)
        cls.out = Path(cls.td.name) / 'analysis'
        cls.manifest = analyze(cls.study, Path(cls.td.name) / 'inventory.json', original, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_every_table_figure_and_number_file_exists(self):
        for tid in [f'T{i}' for i in range(1, 12)]:
            self.assertTrue((self.out / 'tables' / f'{tid}.csv').exists(), tid)
        for name in ('P1_predictions', 'P2_paired_video_fractions', 'P3_transitions', 'P4_refined_scores', 'P5_camera_strata',
                     'P6_box_status', 'P7_localization', 'P9_runtime_tokens'):
            self.assertTrue((self.out / 'figures' / f'{name}.pdf').exists(), name)
        self.assertTrue(list((self.out / 'figures').glob('P8_case*.pdf')))
        self.assertIn(r'\csname val@', (self.out / 'numbers.tex').read_text())

    def test_coverage_counts_add_up(self):
        for r in read(self.out, 'T2'):
            self.assertEqual(int(r['scheduled']), int(r['complete']) + int(r['exhausted']) + int(r['pending']))
        a = {r['category']: r for r in read(self.out, 'T2') if r['condition'] == 'AG'}
        self.assertEqual(int(a['All']['unparsed_labels']), 2)  # the two garbled AG answers
        b = {(r['condition'], r['category']): r for r in read(self.out, 'T2b')}
        self.assertEqual(int(b[('AG', 'All')]['final_vs_runner_differ']), 0)
        self.assertEqual(int(b[('AG', 'All')]['legacy_0_where_final_missing']) + int(b[('AG', 'All')]['legacy_1_where_final_missing']), 2)

    def test_denominators_of_t3_t4_t5(self):
        t3 = {r['category']: r for r in read(self.out, 'T3')}
        for r in t3.values():
            self.assertEqual(int(r['scheduled_windows']), int(r['valid_A_windows']) + int(r['not_valid']))
            self.assertEqual(int(r['not_valid']), int(r['unparsed']) + int(r['truncated']) + int(r['no_record']))
        t4 = {r['category']: r for r in read(self.out, 'T4')}
        self.assertEqual(int(t4['All']['paired_windows']), int(t3['All']['valid_A_windows']) - 2)
        for r in read(self.out, 'T5'):
            cells = sum(int(r[k]) for k in ('normal_normal', 'normal_anomaly', 'anomaly_normal', 'anomaly_anomaly'))
            self.assertEqual(cells, int(r['paired']))
        self.assertEqual(sum(int(r['AG_only_unparsed']) for r in read(self.out, 'T5') if r['category'] != 'All'), 2)

    def test_localization_on_known_geometry(self):
        t8 = [r for r in read(self.out, 'T8') if r['origin'] == 'primary' and not r['size']]
        hand = {(r['condition'], r['iou']): r for r in t8 if r['category'] == 'Handgun'}
        # Every boxed Handgun frame is bright, and the mock box equals the GT box exactly.
        self.assertEqual(float(hand[('G', '0.5')]['recall']), 1.0)
        self.assertEqual(float(hand[('G', '0.75')]['precision']), 1.0)
        mg = {r['condition']: r for r in t8 if r['category'] == 'Machine_Gun' and r['iou'] == '0.5'}
        self.assertEqual((float(mg['G']['recall']), float(mg['G']['precision'])), (0.0, 0.0))
        neg = {r['condition']: r for r in t8 if r['category'] == 'No_Gun' and r['iou'] == '0.5'}
        self.assertEqual(neg['G']['recall'], '')
        self.assertTrue(float(neg['G']['negative_box_rate']) > 0)
        allr = {r['condition']: r for r in t8 if r['category'] == 'All' and r['iou'] == '0.5'}
        # Precision on boxed frames ignores No_Gun boxes; the 'incl' variant counts them.
        self.assertTrue(float(allr['G']['precision_incl_no_gun_frames']) < float(allr['G']['precision']))
        self.assertEqual(allr['AG']['paired_executed_slots'], allr['AG']['evaluable_slots'])
        for c in ('G', 'AG'):  # identical evaluable instances in both conditions
            self.assertEqual(hand[(c, '0.5')]['evaluable_slots'], hand[('G', '0.5')]['evaluable_slots'])

    def test_refinement_and_qualitative_rule(self):
        t6 = {(r['condition'], r['category']): r for r in read(self.out, 'T6')}
        self.assertEqual(int(t6[('A', 'Machine_Gun')]['excluded_too_few_windows']), 1)
        self.assertEqual(int(t6[('A', 'Handgun')]['eligible']), 1)
        cases = json.loads((self.out / 'qualitative.json').read_text())
        self.assertIn('Rule (seed 20261025)', cases['rule'])
        self.assertTrue(all(c['window_choice'] for c in cases['cases']))

import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.firearm_data import (build_inventory, expected_windows, parse_folder, stage, staged_id,
                                      write_inventory, write_registry)
from tests.firearm.fixtures import make_dataset


class FolderNameTests(unittest.TestCase):
    def test_positive_and_negative_names(self):
        p = parse_folder('PAH1_C1_P1_V1_HB_3')
        self.assertEqual((p['action'], p['camera'], p['camera_type'], p['location'], p['subject'], p['brightness'], p['take']),
                         ('PAH1', 'C1', 'cctv_ip', 'P1', 'V1', 'HB', 3))
        self.assertEqual(p['recording_group'], 'PAH1_C1_P1_V1_HB')
        n = parse_folder('N12_C2_P5_V4_LB_2')
        self.assertEqual((n['action'], n['camera_type'], n['take']), ('N12', 'phone', 2))

    def test_unrecognised_name_raises(self):
        for bad in ('PAH1_C3_P1_V1_HB_3', 'video', 'PAH1_C1_P1_V1_HB'):
            with self.assertRaises(ValueError):
                parse_folder(bad)

    def test_window_count_matches_anchor_range(self):
        self.assertEqual([expected_windows(n) for n in (1, 16, 17, 224, 225)], [1, 1, 2, 14, 15])
        self.assertEqual(expected_windows(225), len(range(0, 225, 16)))


class InventoryTests(unittest.TestCase):
    def test_inventory_stage_and_registry(self):
        with tempfile.TemporaryDirectory() as td:
            original = make_dataset(Path(td) / 'original', [
                ('Handgun', 'PAH1_C1_P1_V1_HB_1', 20, (64, 48), 25, {3: [1, 2, 10, 12]}),
                ('No_Gun', 'N1_C2_P3_V1_HB_1', 18, (96, 54), 30, None),
                # Same folder name in another category must still get a unique ID.
                ('Machine_Gun', 'PAH1_C1_P1_V1_HB_1', 17, (64, 48), 25, {}),
            ])
            rows = build_inventory(original, workers=2)
            self.assertEqual(sorted(r['id'] for r in rows), sorted([staged_id('Handgun', 'PAH1_C1_P1_V1_HB_1'),
                             staged_id('No_Gun', 'N1_C2_P3_V1_HB_1'), staged_id('Machine_Gun', 'PAH1_C1_P1_V1_HB_1')]))
            by = {r['id']: r for r in rows}
            hand = by['Handgun__PAH1_C1_P1_V1_HB_1']
            self.assertEqual((hand['frames_decoded'], hand['width'], hand['height'], hand['expected_windows'], hand['label_file']),
                             (20, 64, 48, 2, 'label.json'))
            self.assertIsNone(by['No_Gun__N1_C2_P3_V1_HB_1']['label_file'])
            staged = Path(td) / 'staged'
            stage(rows, original, staged)
            link = staged / 'Handgun__PAH1_C1_P1_V1_HB_1.mp4'
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (original / 'Handgun/PAH1_C1_P1_V1_HB_1/video.mp4').resolve())
            stage(rows, original, staged)  # idempotent for identical targets
            write_inventory(rows, td)
            self.assertEqual(len(json.loads((Path(td) / 'inventory.json').read_text())), 3)
            write_registry(Path(td) / 'registry.json', staged, Path(td) / 'features')
            from vera_runner.data import resolve
            cfg = {'registry': str(Path(td) / 'registry.json'), 'dataset': 'firearm_visilab', 'split': 'test',
                   'stages': ['preprocess', 'infer'], 'selection': {'video_ids': [], 'max_videos': None}}
            self.assertEqual(len(resolve(cfg)), 3)

    def test_stage_refuses_a_link_to_another_target(self):
        with tempfile.TemporaryDirectory() as td:
            original = make_dataset(Path(td) / 'original', [('Handgun', 'PAH1_C1_P1_V1_HB_1', 5, (32, 24), 25, {})])
            rows = build_inventory(original, workers=1)
            staged = Path(td) / 'staged'
            staged.mkdir()
            (staged / 'Handgun__PAH1_C1_P1_V1_HB_1.mp4').symlink_to(Path(td) / 'elsewhere.mp4')
            with self.assertRaises(FileExistsError):
                stage(rows, original, staged)

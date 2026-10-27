import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from studies.firearm_posttraining.grounding_analysis.statistics import grounding_test, claim
from studies.firearm_posttraining.grounding_analysis.run import counts_for_videos, freeze, verify
from studies.firearm_posttraining.data import geometry


class GroundingInferenceTests(unittest.TestCase):
    def arrays(self, groups=8, seeds=3):
        a = np.tile([8, 10, 10], (seeds, groups, 1))
        b = np.tile([4, 10, 10], (seeds, groups, 1))
        return [str(i) for i in range(groups)], a, b

    def test_exact_paired_inference_and_direction(self):
        groups, a, b = self.arrays()
        result = grounding_test(groups, a, b, bootstraps=100)
        self.assertAlmostEqual(result['difference'], .4)
        self.assertAlmostEqual(result['p_value_two_sided'], 2 / 256)
        self.assertEqual(result['permutation_mode'], 'exact')
        self.assertTrue(result['nominal_positive_evidence'])
        self.assertAlmostEqual(result['confidence_interval_95'][0], .4)
        reverse = grounding_test(groups, b, a, bootstraps=100)
        self.assertAlmostEqual(reverse['difference'], -.4)
        self.assertEqual(reverse['p_value_two_sided'], result['p_value_two_sided'])
        self.assertFalse(reverse['nominal_positive_evidence'])

    def test_identical_methods_have_p_one_and_zero_interval(self):
        groups, a, b = self.arrays()
        result = grounding_test(groups, a, a, bootstraps=100)
        self.assertEqual(result['p_value_two_sided'], 1)
        self.assertEqual(result['confidence_interval_95'], [0., 0.])
        self.assertFalse(result['nominal_positive_evidence'])

    def test_duplicate_videos_and_seeds_are_not_independent_samples(self):
        groups, a, b = self.arrays(seeds=1)
        base = grounding_test(groups, a, b, bootstraps=100)
        copied = grounding_test([x for x in groups for _ in range(2)],
                                np.repeat(a, 2, axis=1), np.repeat(b, 2, axis=1), bootstraps=100)
        seeds = grounding_test(groups, np.repeat(a, 3, axis=0), np.repeat(b, 3, axis=0), bootstraps=100)
        for result in (copied, seeds):
            self.assertEqual(result['recording_groups'], 8)
            self.assertEqual(result['permutations'], 256)
            self.assertEqual(result['p_value_two_sided'], base['p_value_two_sided'])
            self.assertAlmostEqual(result['difference'], base['difference'])

    def test_pooled_ratio_not_unweighted_average_video_f1(self):
        groups, a, b = self.arrays()
        a[:, 0] = [0, 1000, 1000]
        b[:, 0, 2] = 1000
        result = grounding_test(groups, a, b, bootstraps=100)
        self.assertAlmostEqual(result['rg_F1_by_seed'][0], 112 / 2140)
        self.assertFalse(result['nominal_positive_evidence'])

    def test_monte_carlo_is_deterministic_and_never_zero(self):
        groups, a, b = self.arrays(groups=17)
        first = grounding_test(groups, a, b, bootstraps=100, permutations=2000)
        second = grounding_test(groups, a, b, bootstraps=100, permutations=2000)
        self.assertEqual(first, second)
        self.assertGreater(first['p_value_two_sided'], 0)
        self.assertEqual(first['permutation_mode'], 'Monte_Carlo_plus_one')

    def test_sparse_ground_truth_is_inconclusive_and_bad_counts_fail(self):
        groups, a, b = self.arrays()
        empty = np.zeros_like(a)
        self.assertEqual(grounding_test(groups, empty, empty)['status'], 'inconclusive')
        sparse = empty.copy(); sparse[:, 0] = [1, 1, 1]
        self.assertEqual(grounding_test(groups, sparse, sparse)['status'], 'inconclusive')
        for bad in (a.astype(float) + .5, a * -1):
            with self.assertRaises(ValueError):
                grounding_test(groups, bad, b)
        mismatch = b.copy(); mismatch[:, 0, 2] += 1
        with self.assertRaises(ValueError):
            grounding_test(groups, a, mismatch)

    def test_gatekeeping_blocks_validation_and_failed_recognition_claims(self):
        groups, a, b = self.arrays()
        result = grounding_test(groups, a, b, bootstraps=100)
        self.assertFalse(claim(result, 'test', False)['confirmatory_grounding_improvement'])
        self.assertFalse(claim(result, 'val', True)['confirmatory_grounding_improvement'])
        self.assertTrue(claim(result, 'test', True)['confirmatory_grounding_improvement'])

    def test_unchanged_scorer_decomposes_known_unknown_whole_frame_and_invalid(self):
        book = json.loads(Path('studies/firearm_training/codebook.json').read_text())
        row = {'id': 'one', 'category': 'Handgun', 'geometry': geometry(640, 480),
               'views': [{'frame_id': i, 'boxes': [[.2, .3, .4, .5]] if i == 0 else ([] if i == 1 else None)} for i in range(8)]}
        value = {'category': 'Handgun', 'action_group': 'aim', 'action_code': 'PAH1',
                 'grounding': [{'frame_id': i, 'boxes': [[200, 300, 400, 500]] if i == 0 else ([[0, 0, 1000, 1000]] if i == 1 else [])} for i in range(8)]}
        other = copy.deepcopy(row); other['id'] = 'two'
        predictions = [{'video': 'one', 'text': json.dumps(value), 'truncated': False},
                       {'video': 'two', 'text': 'invalid', 'truncated': False}]
        counts, per_video, pooled = counts_for_videos([row, other], predictions, book)
        self.assertEqual(counts.tolist(), [[1, 2, 1], [0, 0, 1]])
        self.assertEqual(pooled['whole_image_boxes'], 1)
        self.assertEqual(pooled['scored_frames'], 4)
        with self.assertRaises(ValueError):
            counts_for_videos([row, other], [predictions[0], predictions[0]], book)

    def test_supplement_reproduces_frozen_counts_and_writes_separate_report(self):
        import csv
        from studies.firearm_posttraining.grounding_analysis.run import analyze
        from studies.firearm_posttraining.data import write_json, digest
        from studies.firearm_posttraining.report import localization
        from studies.firearm_posttraining.protocol import SEEDS
        book = json.loads(Path('studies/firearm_training/codebook.json').read_text())
        with tempfile.TemporaryDirectory() as td:
            root, source = Path(td) / 'root', Path(td) / 'source'
            source.mkdir(); root.mkdir()
            write_json(root / 'selection.json', {})
            write_json(root / 'budget.json', {})
            write_json(root / 'report/status.json', {'status': 'complete'})
            write_json(root / 'report/paired-contrasts.json', {f'{split}:RG-SG': {'positive_and_noninferior': True} for split in ('val', 'test')})
            rows = [{'id': str(i), 'category': 'Handgun', 'recording_group': str(i),
                     'geometry': geometry(640, 480),
                     'views': [{'frame_id': k, 'boxes': [[.2, .3, .4, .5]] if k == 0 else None} for k in range(8)]} for i in range(8)]
            base_rows = []; hashes = {}
            for split in ('val', 'test'):
                write_json(source / f'{split}.json', rows)
                write_json(source / f'boxes_{split}.json', {r['id']: {} for r in rows})
                hashes[f'{split}.json'] = digest(source / f'{split}.json')
                for method in ('SG', 'RG'):
                    box = [200, 300, 400, 500] if method == 'RG' else [0, 0, 50, 50]
                    value = {'category': 'Handgun', 'action_group': 'aim', 'action_code': 'PAH1',
                             'grounding': [{'frame_id': k, 'boxes': [box] if k == 0 else []} for k in range(8)]}
                    predictions = [{'video': r['id'], 'text': json.dumps(value), 'truncated': False} for r in rows]
                    for seed in SEEDS:
                        d = root / 'evaluation' / split / f'{method}-{seed}'
                        (d / 'evaluation').mkdir(parents=True)
                        pred = d / 'evaluation/predictions.jsonl'
                        pred.write_text('\n'.join(json.dumps(x) for x in predictions) + '\n')
                        write_json(d / 'result.json', {'status': 'complete', 'predictions_sha256': digest(pred)})
                        write_json(d / 'holdout_manifest.json', {'selection_hash': 'selection', 'annotation_sha256': digest(source / f'boxes_{split}.json')})
                        base_rows.append({'split': split, 'method': method, 'seed': seed, **localization(rows, predictions, book)})
            write_json(root / 'data/audit.json', {'source_hashes': hashes})
            with (root / 'report/grounding.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(base_rows[0])); writer.writeheader(); writer.writerows(base_rows)
            before = digest(root / 'report/grounding.csv')
            with patch('studies.firearm_posttraining.grounding_analysis.run.verify', return_value={'amendment_hash': 'amended', 'frozen_unix': 0}), \
                 patch('studies.firearm_posttraining.grounding_analysis.run.verify_selection', return_value=({'selection_hash': 'selection'}, {'protocol_hash': 'base'})), \
                 patch('studies.firearm_posttraining.grounding_analysis.run.enrich', side_effect=lambda r, a, f: r):
                result = analyze(root, root / 'amendment.json', source, 'unused')
            self.assertTrue(result['results']['test']['confirmatory_grounding_improvement'])
            self.assertFalse(result['results']['val']['confirmatory_grounding_improvement'])
            self.assertEqual(before, digest(root / 'report/grounding.csv'))
            self.assertEqual(json.loads((root / 'report/grounding-statistics-v1/status.json').read_text())['status'], 'complete')

    def test_freeze_rejects_existing_reporting_before_reading_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state = {'selection_exists': False, 'evaluation_directory_exists': True,
                     'report_directory_exists': False, 'evaluation_jobs': []}
            with patch('studies.firearm_posttraining.grounding_analysis.run.validate', return_value={}), \
                 patch('studies.firearm_posttraining.grounding_analysis.run.metadata', return_value=state):
                with self.assertRaisesRegex(ValueError, 'before selection'):
                    freeze(root, root / 'amendment.json')
            self.assertFalse((root / 'amendment.json').exists())

    def test_amendment_tampering_fails_before_result_access(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'amendment.json'
            path.write_text(json.dumps({'status': 'frozen_analysis_amendment', 'amendment_hash': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'modified'):
                verify(path, td)


if __name__ == '__main__':
    unittest.main()

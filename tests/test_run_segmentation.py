import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from scripts.run_segmentation import (aggregate_metrics, build_configurations, choose_best,
                                      prepare_image, run_experiment, iter_predictions)
from psib.preprocessing import preprocess_slice


class RunSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        self.output = self.root / 'results'
        self.output.mkdir()
        self.config = self.root / 'config.yaml'
        self.cfg = {'data_dir': 'data', 'output_dir': 'results',
                    'preprocessing': {'denoising': {'method': 'median', 'median_size': 1}}}
        self.config.write_text(yaml.safe_dump(self.cfg))
        ct = np.full((9, 9, 3), 60, dtype=np.int16)
        seg = np.zeros_like(ct, dtype=np.uint8)
        seg[1:8, 1:8, :2] = 1
        seg[2:4, 2:4, 0] = 2
        seg[2:5, 2:5, 1] = 2
        ct[seg == 2] = 0
        ct[5:7, 5:7, 0] = 0  # Four false positives in the first slice.
        nib.save(nib.Nifti1Image(ct, np.eye(4)), self.root / 'data/volume-1.nii.gz')
        nib.save(nib.Nifti1Image(seg, np.eye(4)), self.root / 'data/segmentation-1.nii.gz')
        self.selected = pd.DataFrame({'case_id': [1, 1, 1], 'slice_index': [0, 1, 2]})
        self.save_selection(self.selected)

    def save_selection(self, frame):
        frame.to_csv(self.output / 'selected_cases_slices.csv', index=False)

    def test_simple_run_outputs_aggregation_and_repeatability(self):
        first = run_experiment(self.config, 'first')
        second = run_experiment(self.config, 'second')
        per_slice = pd.read_csv(first / 'metrics_per_slice.csv')
        per_case = pd.read_csv(first / 'metrics_per_case.csv').iloc[0]
        self.assertAlmostEqual(per_case.dice, 26 / 30)
        self.assertAlmostEqual(per_case.iou, 13 / 17)
        self.assertNotAlmostEqual(per_case.dice, per_slice.dice.mean())
        self.assertEqual(per_slice.iloc[2].dice, 1.0)
        masks = sorted((first / 'predictions').rglob('*.npy'))
        self.assertEqual(len(masks), 3)
        self.assertEqual(len(list((first / 'predictions').rglob('*.png'))), 3)
        for path in masks:
            mask = np.load(path)
            self.assertEqual(mask.shape, (9, 9))
            self.assertEqual(mask.dtype, np.bool_)
            np.testing.assert_array_equal(mask, np.load(second / path.relative_to(first)))
        self.assertFalse(np.load(masks[2]).any())
        for name in ('metrics_per_slice.csv', 'metrics_per_case.csv', 'summary.csv', 'configurations.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(first / name), pd.read_csv(second / name))
        state = json.loads((first / 'run.json').read_text())
        self.assertEqual(state['status'], 'completed')
        self.assertIn('dependencies', state)
        effective = yaml.safe_load((first / 'effective_config.yaml').read_text())
        self.assertFalse(any(key.startswith('_') for key in effective))
        self.assertEqual(effective['segmentation']['threshold'], 0.45)
        with self.assertRaises(FileExistsError):
            run_experiment(self.config, 'first')

    def test_grid_and_hu_conversion(self):
        records = build_configurations({}, True)
        self.assertEqual(len(records), 252)
        self.assertEqual(len({r['configuration_id'] for r in records}), 252)
        global_records = [r for r in records if r['method'] == 'global']
        self.assertEqual({round(r['threshold'], 3) for r in global_records}, {0.425, 0.45, 0.475})
        self.assertEqual({r['threshold_hu'] for r in global_records}, {30, 40, 50})
        with self.assertRaisesRegex(ValueError, 'fuera de la ventana'):
            build_configurations({'preprocessing': {'hu_window': {'center': 0, 'width': 10}}}, True)

    def test_sweep_keeps_all_metrics_and_exports_winners(self):
        self.save_selection(self.selected.iloc[[0, 2]])
        # Keep real segmentation and NIfTI I/O; avoid rendering 12 redundant test PNGs.
        with patch('scripts.run_segmentation.export_prediction') as export:
            out = run_experiment(self.config, 'sweep', True)
            self.assertEqual(export.call_count, 6)
        slices = pd.read_csv(out / 'metrics_per_slice.csv')
        self.assertEqual(len(slices), 504)
        best = pd.read_csv(out / 'best_configurations.csv')
        self.assertEqual(set(best.method), {'global', 'otsu', 'statistical'})
        self.assertEqual(len(best), 3)
        configs = pd.read_csv(out / 'configurations.csv').to_dict('records')
        summary = pd.read_csv(out / 'summary.csv')
        expected = choose_best(summary.sample(frac=1, random_state=5), configs)
        self.assertEqual(list(best.configuration_id), list(expected.configuration_id))

    def test_cached_predictions_equal_direct_segmentations(self):
        from psib.segmentation import segment_slice
        image = np.random.default_rng(7).random((15, 16))
        liver = np.zeros_like(image, dtype=bool)
        liver[1:-1, 1:-1] = True
        records = build_configurations({}, True)
        for record, actual in iter_predictions(image, liver, records):
            settings = {key: value for key, value in record.items()
                        if key not in ('configuration_id', 'threshold_hu')}
            np.testing.assert_array_equal(actual, segment_slice(image, liver, settings))

    def test_tie_break_and_empty_case(self):
        summary = pd.DataFrame({'configuration_id': ['b', 'a', 'c'],
                                'dice_mean': [0.5, 0.5, 0.5], 'iou_mean': [0.4, 0.4, 0.3]})
        configs = [{'configuration_id': cid, 'method': 'global'} for cid in ('b', 'a', 'c')]
        self.assertEqual(choose_best(summary, configs).iloc[0].configuration_id, 'a')
        frame = pd.DataFrame([{'configuration_id': 'a', 'case_id': 1, 'slice_index': 0,
                               'pred_area': 0, 'reference_area': 0, 'intersection': 0, 'union': 0}])
        cases, totals = aggregate_metrics(frame)
        self.assertEqual(cases.iloc[0].dice, 1)
        self.assertEqual(totals.iloc[0].dice_std, 0)

    def test_preparation_matches_existing_preprocessing(self):
        hu = np.arange(100).reshape(10, 10)
        liver = np.ones_like(hu)
        for method in ('median', 'gaussian', 'bilateral'):
            cfg = {'preprocessing': {'denoising': {'method': method}}}
            np.testing.assert_array_equal(prepare_image(hu, cfg), preprocess_slice(hu, liver, cfg)['denoised'])

    def test_selection_errors_leave_failed_status(self):
        invalid = [pd.DataFrame(columns=['case_id', 'slice_index']),
                   pd.DataFrame({'other': [1]}),
                   pd.DataFrame({'case_id': [1, 1], 'slice_index': [0, 0]}),
                   pd.DataFrame({'case_id': [1], 'slice_index': [-1]}),
                   pd.DataFrame({'case_id': [1], 'slice_index': [0.5]}),
                   pd.DataFrame({'case_id': [1], 'slice_index': [np.nan]}),
                   pd.DataFrame({'case_id': [1], 'slice_index': [3]}),
                   pd.DataFrame({'case_id': [99], 'slice_index': [0]})]
        for i, frame in enumerate(invalid):
            self.save_selection(frame)
            with self.subTest(i=i), self.assertRaises(RuntimeError):
                run_experiment(self.config, f'bad_{i}')
            state = json.loads((self.output / f'segmentation/bad_{i}/run.json').read_text())
            self.assertEqual(state['status'], 'failed')
            if i >= 6:
                self.assertIn('caso', state['context'])
        (self.output / 'selected_cases_slices.csv').unlink()
        with self.assertRaisesRegex(RuntimeError, 'explore_dataset.py'):
            run_experiment(self.config, 'missing')
        for run_id in ('../escape', 'with space', ''):
            with self.assertRaises(ValueError):
                run_experiment(self.config, run_id)

    def test_export_failure_does_not_mark_complete(self):
        with patch('scripts.run_segmentation.export_prediction', side_effect=OSError('disk error')):
            with self.assertRaisesRegex(RuntimeError, 'caso 1, corte 0'):
                run_experiment(self.config, 'export_failure')
        state = json.loads((self.output / 'segmentation/export_failure/run.json').read_text())
        self.assertEqual(state['status'], 'failed')

    def test_local_configuration_precedence(self):
        self.cfg['segmentation'] = {'threshold': 0.0}
        (self.root / 'config.local.yaml').write_text(yaml.safe_dump(self.cfg))
        with patch('scripts.run_segmentation.export_prediction'):
            out = run_experiment(self.config, 'local')
        self.assertEqual(pd.read_csv(out / 'metrics_per_slice.csv').pred_area.sum(), 0)


if __name__ == '__main__':
    unittest.main()

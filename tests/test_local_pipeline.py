"""Configured growth input units, context and final morphology regression tests."""
import unittest
from unittest.mock import patch
import numpy as np
from psib.detection import configured_prediction
from psib.local_segmentation import detect_local
from psib.preprocessing import prepare_enhanced_slice
from psib.segmentation import postprocess_mask


class ConfiguredLocalTests(unittest.TestCase):
    def setUp(self):
        self.hu = np.full((40, 40), 90.)
        self.hu[14:26, 14:26] = 30.
        self.liver = np.ones((40, 40), bool)
        self.liver[:3] = False
        self.cfg = dict(local_pipeline='configured', detection=dict(
            enabled=True, method='local_growth', high=1., low=.33, context=False),
            preprocessing=dict(hu_window=dict(enabled=True, center=30., width=150.),
                               denoising=dict(method='gaussian', gaussian_sigma=1.),
                               enhancement=dict(method='none')),
            segmentation=dict(opening=True, closing=True, fill_holes=True, radius=2, min_area=50))

    def test_units_neighbors_and_no_second_smoothing(self):
        self.cfg['detection']['context'] = True
        def fake(image, liver, spacing, settings, *, context, preprocessed):
            expected = prepare_enhanced_slice(self.hu, self.cfg, liver_mask=self.liver)
            np.testing.assert_allclose(image[self.liver], (expected*150-45)[self.liver])
            neighbor = prepare_enhanced_slice(self.hu+10, self.cfg, liver_mask=self.liver)
            np.testing.assert_allclose(context[0][1][self.liver], (neighbor*150-45)[self.liver])
            self.assertTrue(preprocessed)
            return dict(mask=self.liver.copy())
        with patch('psib.local_pipeline.detect_local', side_effect=fake):
            result = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg,
                                          context=[(1., self.hu+10, self.liver)])
        self.assertFalse(result['mask'][~self.liver].any())

    def test_postprocessing_and_preview_override(self):
        raw = np.zeros_like(self.liver)
        raw[10:25, 10:25] = True
        raw[5, 5] = True
        with patch('psib.local_pipeline.detect_local', return_value=dict(mask=raw)):
            result = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg)
        np.testing.assert_array_equal(result['mask'], postprocess_mask(raw, self.liver, self.cfg['segmentation']))
        self.assertFalse(result['mask'][5, 5])
        with patch('psib.local_pipeline.detect_local', return_value=dict(mask=raw)):
            result = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg, dict(min_area=1000))
        self.assertFalse(result['mask'].any())

    def test_real_growth_determinism_and_area_control(self):
        first = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg)
        second = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg)
        self.assertTrue(first['mask'].any())
        np.testing.assert_array_equal(first['mask'], second['mask'])
        self.cfg['segmentation']['min_area'] = 10000
        self.assertFalse(configured_prediction(self.hu, self.liver, (1., 1.), self.cfg)['mask'].any())

    def test_old_config_remains_raw_and_empty_liver(self):
        self.cfg.pop('local_pipeline')
        expected = detect_local(self.hu, self.liver, (1., 1.),
                                {k:v for k,v in self.cfg['detection'].items() if k!='enabled'})
        actual = configured_prediction(self.hu, self.liver, (1., 1.), self.cfg)
        np.testing.assert_array_equal(expected['mask'], actual['mask'])
        self.cfg['local_pipeline'] = 'configured'
        actual = configured_prediction(self.hu, np.zeros_like(self.liver), (1., 1.), self.cfg)
        self.assertFalse(actual['mask'].any())
